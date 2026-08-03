"""Fine-tune exactly one assertion output while keeping NER/span predictions immutable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from .model_v2 import HybridNER
from .train_v2 import HybridDataset, collate, read_jsonl, targeted_sampler_weights


def _state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _focused_row_hash(model: HybridNER, index: int) -> str:
    digest = hashlib.sha256()
    digest.update(model.assertion_head.weight[index].detach().cpu().numpy().tobytes())
    digest.update(model.assertion_head.bias[index].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _sampler_weights(rows: list[dict], prefix: str, targeted_mass: float) -> list[float]:
    """Backward-compatible wrapper used by historical tests and old scripts."""
    try:
        weights, _ = targeted_sampler_weights(
            rows, prefixes=(prefix,), targeted_mass=targeted_mass
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return weights


@torch.no_grad()
def evaluate_focus(
    model: HybridNER,
    loader: DataLoader,
    assertion_index: int,
    device: str,
    threshold: float,
) -> dict[str, float | int]:
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        result = model(
            batch["input_ids"],
            batch["attention_mask"],
            span_starts=batch["span_starts"],
            span_ends=batch["span_ends"],
        )
        pred = torch.sigmoid(result["assertion_logits"][..., assertion_index]) >= threshold
        gold = batch["assertion_labels"][..., assertion_index].bool()
        mask = batch["assertion_mask"].bool()
        tp += int((pred & gold & mask).sum())
        fp += int((pred & ~gold & mask).sum())
        fn += int((~pred & gold & mask).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _loader(rows, tokenizer, label_to_id, max_len, batch_size, seed, *, sampler=None):
    dataset = HybridDataset(rows, tokenizer, label_to_id, max_len=max_len, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, dataset.pad),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Window dataset track_a")
    parser.add_argument("--init-checkpoint", required=True, help="Checkpoint HybridNER baseline")
    parser.add_argument("--historical-dev", required=True, help="Window JSONL dev independent")
    parser.add_argument(
        "--public-dev",
        help="Part 3 gán tay ở dạng window; mặc định dùng test.jsonl của dataset",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--focus", default="isHistorical")
    parser.add_argument("--targeted-id-prefix", required=True)
    parser.add_argument("--targeted-mass", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--independent-dev-tolerance", type=float, default=0.01,
        help="Mức giảm F1 tối đa trên dev độc lập khi chọn theo public dev",
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if not 0 < args.targeted_mass < 1:
        raise SystemExit("--targeted-mass phải nằm trong (0,1)")
    if not 0 <= args.independent_dev_tolerance <= 1:
        raise SystemExit("--independent-dev-tolerance phải nằm trong [0,1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = HybridNER.load(args.init_checkpoint, device)
    assertion_names = list(meta["assertions"])
    if args.focus not in assertion_names:
        raise SystemExit(f"Assertion {args.focus!r} không có trong checkpoint: {assertion_names}")
    assertion_index = assertion_names.index(args.focus)
    max_len = int(meta["max_len"])

    tokenizer = AutoTokenizer.from_pretrained(args.init_checkpoint, use_fast=True)
    data_dir = Path(args.data)
    labels = (data_dir / "labels.txt").read_text(encoding="utf-8").splitlines()
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_rows = read_jsonl(data_dir / "train.jsonl")
    validation_rows = read_jsonl(data_dir / "validation.jsonl")
    dev_rows = read_jsonl(Path(args.historical_dev))
    public_dev_path = Path(args.public_dev) if args.public_dev else data_dir / "test.jsonl"
    public_dev_rows = read_jsonl(public_dev_path)
    if not train_rows or not validation_rows or not dev_rows or not public_dev_rows:
        raise SystemExit("Train/validation/historical-dev/public-dev không được rỗng")

    weights = _sampler_weights(train_rows, args.targeted_id_prefix, args.targeted_mass)
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_rows), replacement=True, generator=sampler_generator
    )
    train_loader = _loader(
        train_rows, tokenizer, label_to_id, max_len, args.batch_size, args.seed, sampler=sampler
    )
    validation_loader = _loader(
        validation_rows, tokenizer, label_to_id, max_len, args.eval_batch_size, args.seed
    )
    dev_loader = _loader(
        dev_rows, tokenizer, label_to_id, max_len, args.eval_batch_size, args.seed
    )
    public_dev_loader = _loader(
        public_dev_rows, tokenizer, label_to_id, max_len, args.eval_batch_size, args.seed
    )

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.assertion_head.weight.requires_grad = True
    model.assertion_head.bias.requires_grad = True
    frozen_modules = {
        "encoder": _state_hash(model.encoder),
        "bio_head": _state_hash(model.bio_head),
        "span_projection": _state_hash(model.span_projection),
        "span_type_head": _state_hash(model.span_type_head),
    }
    nonfocus_before = {
        name: _focused_row_hash(model, index)
        for index, name in enumerate(assertion_names) if index != assertion_index
    }
    focus_before = _focused_row_hash(model, assertion_index)

    optimizer = torch.optim.AdamW(model.assertion_head.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device == "cuda")
    os.makedirs(args.out, exist_ok=True)
    history = []

    def save_checkpoint(path: str, epoch: int, metrics: dict) -> None:
        model.save(path, tokenizer, {
            "labels": labels,
            "types": list(meta["types"]),
            "assertions": assertion_names,
            "max_len": max_len,
            "assertion_head_only": True,
            "focus_assertion": args.focus,
            "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
            "selected_epoch": epoch,
            "selection_metrics": metrics,
        })

    initial_dev = evaluate_focus(model, dev_loader, assertion_index, device, args.threshold)
    initial_val = evaluate_focus(model, validation_loader, assertion_index, device, args.threshold)
    initial_public = evaluate_focus(
        model, public_dev_loader, assertion_index, device, args.threshold
    )
    initial_metrics = {
        "historical_dev": initial_dev,
        "public_dev": initial_public,
        "validation": initial_val,
    }
    best_key = (initial_public["f1"], initial_dev["f1"], initial_val["f1"])
    save_checkpoint(args.out + "/best", 0, initial_metrics)
    history.append({"epoch": 0, **initial_metrics, "eligible": True})

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.fp16 and device == "cuda"):
                result = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    span_starts=batch["span_starts"],
                    span_ends=batch["span_ends"],
                )
                logits = result["assertion_logits"][..., assertion_index]
                gold = batch["assertion_labels"][..., assertion_index]
                mask = batch["assertion_mask"].float()
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, gold, reduction="none"
                )
                loss = (raw * mask).sum() / mask.sum().clamp(min=1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        dev_metrics = evaluate_focus(model, dev_loader, assertion_index, device, args.threshold)
        validation_metrics = evaluate_focus(
            model, validation_loader, assertion_index, device, args.threshold
        )
        public_metrics = evaluate_focus(
            model, public_dev_loader, assertion_index, device, args.threshold
        )
        eligible = (
            dev_metrics["f1"]
            >= initial_dev["f1"] - args.independent_dev_tolerance
        )
        row = {
            "epoch": epoch,
            "historical_dev": dev_metrics,
            "public_dev": public_metrics,
            "validation": validation_metrics,
            "eligible": eligible,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.save_every_epoch:
            save_checkpoint(args.out + f"/epoch-{epoch:03d}", epoch, row)
        key = (public_metrics["f1"], dev_metrics["f1"], validation_metrics["f1"])
        if eligible and key > best_key:
            best_key = key
            save_checkpoint(args.out + "/best", epoch, row)

    frozen_after = {
        "encoder": _state_hash(model.encoder),
        "bio_head": _state_hash(model.bio_head),
        "span_projection": _state_hash(model.span_projection),
        "span_type_head": _state_hash(model.span_type_head),
    }
    nonfocus_after = {
        name: _focused_row_hash(model, index)
        for index, name in enumerate(assertion_names) if index != assertion_index
    }
    report = {
        "schema_version": 1,
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "focus": args.focus,
        "targeted_mass": args.targeted_mass,
        "targeted_windows": sum(
            str(row.get("record_id") or "").startswith(args.targeted_id_prefix)
            for row in train_rows
        ),
        "train_windows": len(train_rows),
        "historical_dev_windows": len(dev_rows),
        "public_dev": str(public_dev_path.resolve()),
        "public_dev_windows": len(public_dev_rows),
        "independent_dev_tolerance": args.independent_dev_tolerance,
        "frozen_modules_unchanged": frozen_modules == frozen_after,
        "nonfocus_assertion_rows_unchanged": nonfocus_before == nonfocus_after,
        "focus_row_changed": focus_before != _focused_row_hash(model, assertion_index),
        "history": history,
    }
    if not report["frozen_modules_unchanged"] or not report["nonfocus_assertion_rows_unchanged"]:
        raise RuntimeError("Invariant head-only bị vi phạm: tham số ngoài focus assertion đã đổi")
    (Path(args.out) / "assertion_head_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
