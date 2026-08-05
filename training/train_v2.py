from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .model_v2 import HybridNER
from .predict_v2 import encoder_capacity, predict_text
from .provenance import verify_dataset
from .score_local import score as score_raw_predictions


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def targeted_sampler_weights(
    rows: list[dict],
    *,
    prefixes: tuple[str, ...] = (),
    regexes: tuple[str, ...] = (),
    targeted_mass: float | None = None,
) -> tuple[list[float], dict[str, float | int | list[str]]]:
    """Override one targeted group's sampler mass while preserving within-group weights.

    ``targeted_mass=0`` is intentional: it supplies a rehearsal control with exactly the same
    corpus size/steps as a targeted run while never sampling the new records.
    """
    base = [float(row.get("sample_weight", 1.0)) for row in rows]
    if not base or any(not math.isfinite(value) or value < 0 for value in base):
        raise ValueError("sample_weight phải hữu hạn, không âm và corpus không được rỗng")
    if targeted_mass is None:
        return base, {
            "override_enabled": False,
            "targeted_windows": 0,
            "other_windows": len(rows),
            "targeted_mass": 0.0,
            "other_mass": 1.0,
            "prefixes": list(prefixes),
            "regexes": list(regexes),
        }
    if not 0 <= targeted_mass < 1:
        raise ValueError("--targeted-mass phải nằm trong [0,1)")
    if not prefixes and not regexes:
        raise ValueError("--targeted-mass cần ít nhất một --targeted-id-prefix/regex")
    compiled = [re.compile(pattern) for pattern in regexes]
    selected = []
    for index, row in enumerate(rows):
        record_id = str(row.get("record_id") or "")
        if any(record_id.startswith(prefix) for prefix in prefixes) or any(
            pattern.search(record_id) for pattern in compiled
        ):
            selected.append(index)
    selected_set = set(selected)
    other = [index for index in range(len(rows)) if index not in selected_set]
    if not selected:
        raise ValueError("Không tìm thấy targeted window theo selector đã truyền")
    if not other:
        raise ValueError("Corpus không còn window nền để rehearsal")

    def distribute(indexes: list[int], mass: float) -> None:
        source_sum = sum(base[index] for index in indexes)
        if source_sum > 0:
            for index in indexes:
                output[index] = mass * base[index] / source_sum
        else:
            for index in indexes:
                output[index] = mass / len(indexes)

    output = [0.0] * len(rows)
    distribute(selected, targeted_mass)
    distribute(other, 1.0 - targeted_mass)
    return output, {
        "override_enabled": True,
        "targeted_windows": len(selected),
        "other_windows": len(other),
        "targeted_mass": targeted_mass,
        "other_mass": 1.0 - targeted_mass,
        "prefixes": list(prefixes),
        "regexes": list(regexes),
    }


class HybridDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        label_to_id,
        max_len=256,
        negative_span_ratio=2.0,
        max_negative_words=5,
        seed=0,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len
        self.negative_span_ratio = negative_span_ratio
        self.max_negative_words = max_negative_words
        self.seed = seed
        self.bos = tokenizer.bos_token_id
        if self.bos is None:
            self.bos = tokenizer.cls_token_id
        self.eos = tokenizer.eos_token_id
        if self.eos is None:
            self.eos = tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id or 0
        self.unk = tokenizer.unk_token_id or self.pad

    def __len__(self):
        return len(self.rows)

    def _negative_spans(self, row, word_limit, index):
        positives = [(span["start"], span["end"]) for span in row["spans"] if span["end"] <= word_limit]
        wanted = max(2, int(math.ceil(len(positives) * self.negative_span_ratio)))
        rng = random.Random(f"{self.seed}:{row['id']}:{index}")
        output, seen = [], set(positives)
        for _ in range(wanted * 20):
            if len(output) >= wanted or word_limit <= row.get("n_prefix", 0):
                break
            start = rng.randrange(row.get("n_prefix", 0), word_limit)
            end = min(word_limit, start + rng.randint(1, self.max_negative_words))
            candidate = (start, end)
            if candidate in seen or any(not (end <= ps or start >= pe) for ps, pe in positives):
                continue
            seen.add(candidate)
            output.append(candidate)
        return output

    def __getitem__(self, index):
        row = self.rows[index]
        input_ids = [self.bos]
        bio_labels = [-100]
        loss_mask = [0]
        word_first, word_last = [], []
        for token, tag, keep in zip(row["tokens"], row["ner_tags"], row["token_loss_mask"]):
            subwords = self.tokenizer.encode(token, add_special_tokens=False) or [self.unk]
            if len(input_ids) + len(subwords) + 1 > self.max_len:
                break
            word_first.append(len(input_ids))
            input_ids.extend(subwords)
            word_last.append(len(input_ids) - 1)
            bio_labels.extend([self.label_to_id[tag]] + [-100] * (len(subwords) - 1))
            loss_mask.extend([keep] + [0] * (len(subwords) - 1))
        word_limit = len(word_first)
        input_ids.append(self.eos)
        bio_labels.append(-100)
        loss_mask.append(0)

        span_starts, span_ends, span_labels = [], [], []
        assertion_labels, assertion_mask = [], []
        for span in row["spans"]:
            if span["end"] > word_limit:
                continue
            span_starts.append(word_first[span["start"]])
            span_ends.append(word_last[span["end"] - 1])
            span_labels.append(span["type_id"])
            assertion_labels.append(span["assertions"])
            assertion_mask.append(span["assertion_mask"])
        for start, end in self._negative_spans(row, word_limit, index):
            span_starts.append(word_first[start])
            span_ends.append(word_last[end - 1])
            span_labels.append(0)
            assertion_labels.append([0, 0, 0])
            assertion_mask.append(0)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "bio_labels": bio_labels,
            "token_loss_mask": loss_mask,
            "span_starts": span_starts,
            "span_ends": span_ends,
            "span_labels": span_labels,
            "assertion_labels": assertion_labels,
            "assertion_mask": assertion_mask,
            "sample_weight": float(row.get("sample_weight", 1.0)),
        }


def collate(batch, pad):
    max_tokens = max(len(row["input_ids"]) for row in batch)
    max_spans = max(1, max(len(row["span_starts"]) for row in batch))
    output = {key: [] for key in (
        "input_ids", "attention_mask", "bio_labels", "token_loss_mask",
        "span_starts", "span_ends", "span_labels", "assertion_labels", "assertion_mask",
    )}
    for row in batch:
        token_pad = max_tokens - len(row["input_ids"])
        output["input_ids"].append(row["input_ids"] + [pad] * token_pad)
        output["attention_mask"].append(row["attention_mask"] + [0] * token_pad)
        output["bio_labels"].append(row["bio_labels"] + [-100] * token_pad)
        output["token_loss_mask"].append(row["token_loss_mask"] + [0] * token_pad)
        span_pad = max_spans - len(row["span_starts"])
        output["span_starts"].append(row["span_starts"] + [0] * span_pad)
        output["span_ends"].append(row["span_ends"] + [0] * span_pad)
        output["span_labels"].append(row["span_labels"] + [-100] * span_pad)
        output["assertion_labels"].append(row["assertion_labels"] + [[0, 0, 0]] * span_pad)
        output["assertion_mask"].append(row["assertion_mask"] + [0] * span_pad)
    return {
        key: torch.tensor(value, dtype=torch.float if key == "assertion_labels" else torch.long)
        for key, value in output.items()
    }


@torch.no_grad()
def evaluate(model, loader, id_to_label, device):
    model.eval()
    true_bio, pred_bio = [], []
    span_correct = span_total = 0
    a_tp = a_fp = a_fn = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        result = model(**batch)
        predicted = result["bio_logits"].argmax(-1).cpu()
        labels = batch["bio_labels"].cpu()
        for pred_row, label_row in zip(predicted, labels):
            true_seq, pred_seq = [], []
            for pred, label in zip(pred_row.tolist(), label_row.tolist()):
                if label == -100:
                    continue
                true_seq.append(id_to_label[label])
                pred_seq.append(id_to_label[pred])
            # seqeval raises ``Found input variables without list of list`` when a split only
            # contains fully masked/prefix windows.  Such rows contain no evaluable BIO target.
            if true_seq:
                true_bio.append(true_seq)
                pred_bio.append(pred_seq)
        valid_span = batch["span_labels"] > 0
        if valid_span.any():
            span_pred = result["span_logits"].argmax(-1)
            span_correct += int(((span_pred == batch["span_labels"]) & valid_span).sum())
            span_total += int(valid_span.sum())
            assertion_pred = (torch.sigmoid(result["assertion_logits"]) >= 0.5).long()
            mask = batch["assertion_mask"].unsqueeze(-1).bool()
            gold = batch["assertion_labels"].long()
            a_tp += int(((assertion_pred == 1) & (gold == 1) & mask).sum())
            a_fp += int(((assertion_pred == 1) & (gold == 0) & mask).sum())
            a_fn += int(((assertion_pred == 0) & (gold == 1) & mask).sum())
    ner_f1 = f1_score(true_bio, pred_bio) if true_bio else 0.0
    precision = a_tp / max(1, a_tp + a_fp)
    recall = a_tp / max(1, a_tp + a_fn)
    assertion_f1 = 2 * precision * recall / max(1e-9, precision + recall)
    report = (
        classification_report(true_bio, pred_bio, output_dict=True, zero_division=0)
        if true_bio else {}
    )
    return {
        "ner_f1": ner_f1,
        "per_type_f1": {
            name: values["f1-score"]
            for name, values in report.items()
            if isinstance(values, dict) and "f1-score" in values and name not in {"micro avg", "macro avg", "weighted avg"}
        },
        "span_type_accuracy": span_correct / max(1, span_total),
        "assertion_precision": precision,
        "assertion_recall": recall,
        "assertion_f1": assertion_f1,
        "selection_score": 0.8 * ner_f1 + 0.2 * assertion_f1,
    }


@torch.no_grad()
def evaluate_raw_documents(
    model,
    tokenizer,
    meta: dict,
    records: list[dict],
    device,
    *,
    max_len: int,
    max_words: int = 180,
    overlap_words: int = 45,
    prefix_subword_cap: int = 64,
    ner_type_thresholds: dict | None = None,
) -> dict:
    """Evaluate the real inference/merge path on complete documents."""
    gold = {}
    predictions = {}
    by_source: dict[str, list[str]] = {}
    for index, record in enumerate(records):
        record_id = str(record.get("id") or index)
        if record_id in gold:
            raise ValueError(f"Trùng raw validation id: {record_id}")
        gold[record_id] = record.get("entities") or []
        predictions[record_id] = predict_text(
            record["text"], model, tokenizer, meta, device,
            max_len=max_len,
            max_words=max_words,
            overlap_words=overlap_words,
            prefix_subword_cap=prefix_subword_cap,
            type_threshold=0.60,
            disagreement_threshold=0.85,
            assertion_threshold=0.60,
            assertion_policy="part3",
            assertion_aggregation="selected",
            tokenizer_aware_windows=True,
            ner_type_thresholds=ner_type_thresholds,
        )
        source = str(record.get("source") or record.get("record_kind") or "unknown")
        by_source.setdefault(source, []).append(record_id)

    raw = score_raw_predictions(gold, predictions)
    overlap = raw["overlap"]["f1"] / 100.0
    boundary = raw["boundary_f1_reference_only"] / 100.0
    assertion = raw["assertion_micro_on_matched"]["f1"] / 100.0
    overall_selection = 0.60 * overlap + 0.15 * boundary + 0.25 * assertion
    source_reports = {}
    for source, ids in sorted(by_source.items()):
        subset = score_raw_predictions(
            {key: gold[key] for key in ids},
            {key: predictions[key] for key in ids},
        )
        source_reports[source] = subset
    source_components = [
        {
            "overlap_f1": item["overlap"]["f1"] / 100.0,
            "boundary_f1": item["boundary_f1_reference_only"] / 100.0,
            "assertion_f1": item["assertion_micro_on_matched"]["f1"] / 100.0,
        }
        for item in source_reports.values()
    ]
    macro_by_source = {
        name: sum(item[name] for item in source_components) / max(1, len(source_components))
        for name in ("overlap_f1", "boundary_f1", "assertion_f1")
    }
    selection = (
        0.60 * macro_by_source["overlap_f1"]
        + 0.15 * macro_by_source["boundary_f1"]
        + 0.25 * macro_by_source["assertion_f1"]
    )
    return {
        "overlap_f1": overlap,
        "boundary_f1": boundary,
        "assertion_f1": assertion,
        "selection_score": selection,
        "overall_selection_score": overall_selection,
        "selection_aggregation": "macro_by_source",
        "weights": {"overlap": 0.60, "boundary": 0.15, "assertion": 0.25},
        "scorer_report": raw,
        "by_source": source_reports,
        "macro_by_source": macro_by_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Train hybrid BIO+span+assertion NER")
    parser.add_argument("--data-dir", "--data", dest="data_dir", required=True)
    parser.add_argument("--model", help="Backbone Hugging Face; không cần khi warm-start")
    parser.add_argument(
        "--init-checkpoint",
        help="Checkpoint HybridNER dùng để warm-start toàn bộ model thay vì khởi tạo backbone",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--encoder-lr", type=float,
        help="LR riêng backbone; mặc định dùng --lr",
    )
    parser.add_argument(
        "--head-lr", type=float,
        help="LR chung BIO/span/assertion heads; mặc định dùng --lr",
    )
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--span-weight", type=float, default=0.5)
    parser.add_argument("--assertion-weight", type=float, default=1.0)
    parser.add_argument("--negative-span-ratio", type=float, default=2.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument(
        "--max-updates", type=int,
        help="Dừng đúng số optimizer update; dùng để refit corpus khác kích thước",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Số epoch không cải thiện selection trước khi dừng; 0=tắt",
    )
    parser.add_argument(
        "--raw-validation",
        help="Raw JSONL để chọn checkpoint end-to-end; mặc định raw_validation.jsonl trong data",
    )
    parser.add_argument(
        "--raw-test",
        help="Raw JSONL để báo cáo end-to-end; mặc định raw_test.jsonl trong data",
    )
    parser.add_argument("--eval-max-words", type=int, default=180)
    parser.add_argument("--eval-overlap-words", type=int, default=45)
    parser.add_argument("--prefix-subword-cap", type=int, default=64)
    parser.add_argument(
        "--disable-raw-eval", action="store_true",
        help="Chỉ dùng metric window cũ; dành cho ablation/tương thích dataset cũ",
    )
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument("--require-dataset-variant", action="append", default=[])
    parser.add_argument("--require-part3-sha256")
    parser.add_argument(
        "--targeted-id-prefix", action="append", default=[],
        help="Có thể lặp; record_id khớp một prefix sẽ thuộc nhóm targeted",
    )
    parser.add_argument(
        "--targeted-id-regex", action="append", default=[],
        help="Có thể lặp; regex search trên record_id để tạo nhóm targeted",
    )
    parser.add_argument(
        "--targeted-mass", type=float,
        help="Khối lượng sampler của nhóm targeted; 0 tạo rehearsal control",
    )
    parser.add_argument(
        "--save-every-epoch", action="store_true",
        help="Lưu thêm epoch-XXX để phân tích warm-start, ngoài checkpoint best",
    )
    parser.add_argument(
        "--save-last", action="store_true",
        help="Lưu checkpoint last sau đúng số epoch; dùng cho private refit theo epoch đã chọn",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if not args.model and not args.init_checkpoint:
        parser.error("cần --model hoặc --init-checkpoint")
    if args.grad_accumulation < 1:
        parser.error("--grad-accumulation phải >= 1")
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates phải >= 1")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience phải >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    try:
        provenance = verify_dataset(
            data_dir,
            require_manifest=args.require_provenance,
            allowed_variants=args.require_dataset_variant,
            required_part3_sha256=args.require_part3_sha256,
        )
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Provenance preflight thất bại: {error}") from error
    labels = (data_dir / "labels.txt").read_text(encoding="utf-8").splitlines()
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer_source = args.init_checkpoint or args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    validation_rows = read_jsonl(data_dir / "validation.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    raw_validation_path = Path(args.raw_validation) if args.raw_validation else data_dir / "raw_validation.jsonl"
    raw_test_path = Path(args.raw_test) if args.raw_test else data_dir / "raw_test.jsonl"
    raw_validation_rows = (
        [] if args.disable_raw_eval or not raw_validation_path.is_file()
        else read_jsonl(raw_validation_path)
    )
    raw_test_rows = (
        [] if args.disable_raw_eval or not raw_test_path.is_file()
        else read_jsonl(raw_test_path)
    )
    train_data = HybridDataset(
        train_rows, tokenizer, label_to_id, args.max_len, args.negative_span_ratio, seed=args.seed
    )
    validation_data = HybridDataset(
        validation_rows, tokenizer, label_to_id, args.max_len, args.negative_span_ratio, seed=args.seed
    )
    test_data = HybridDataset(
        test_rows, tokenizer, label_to_id, args.max_len, args.negative_span_ratio, seed=args.seed
    )
    try:
        weights, sampler_report = targeted_sampler_weights(
            train_rows,
            prefixes=tuple(args.targeted_id_prefix),
            regexes=tuple(args.targeted_id_regex),
            targeted_mass=args.targeted_mass,
        )
    except (ValueError, re.error) as error:
        raise SystemExit(str(error)) from error
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=sampler_generator
    )
    collator = lambda rows: collate(rows, train_data.pad)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler, collate_fn=collator
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator
    )
    types = (data_dir / "types.txt").read_text(encoding="utf-8").splitlines()
    assertions = (data_dir / "assertions.txt").read_text(encoding="utf-8").splitlines()
    if args.init_checkpoint:
        model, init_meta = HybridNER.load(args.init_checkpoint, device)
        for key, actual, expected in (
            ("labels", list(init_meta["labels"]), labels),
            ("types", list(init_meta["types"]), types),
            ("assertions", list(init_meta["assertions"]), assertions),
        ):
            if actual != expected:
                raise SystemExit(
                    f"Schema {key} của checkpoint không khớp dataset: {actual!r} != {expected!r}"
                )
    else:
        model = HybridNER(
            args.model,
            len(labels),
            span_weight=args.span_weight,
            assertion_weight=args.assertion_weight,
        ).to(device)
    capacity = encoder_capacity(model)
    if args.max_len > capacity:
        # Cùng cái bẫy như predict_v2: vượt sức chứa position embedding thì CUDA chỉ báo
        # `device-side assert triggered`, không nói gì về độ dài.
        raise SystemExit(
            f"--max-len {args.max_len} vượt sức chứa của {args.model or args.init_checkpoint} "
            f"({capacity}); "
            f"dùng --max-len {capacity} trở xuống"
        )
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": encoder_lr},
        {"params": head_parameters, "lr": head_lr},
    ], weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accumulation)
    total_steps = args.max_updates or max(1, int(updates_per_epoch * args.epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device == "cuda")
    best = -1.0
    best_epoch = 0
    last_metrics: dict = {}
    os.makedirs(args.out, exist_ok=True)
    run_report = {
        "schema_version": 1,
        "data": str(data_dir.resolve()),
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        "backbone": model.base_model,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "encoder_lr": encoder_lr,
        "head_lr": head_lr,
        "max_len": args.max_len,
        "grad_accumulation": args.grad_accumulation,
        "updates_per_epoch": updates_per_epoch,
        "max_updates": args.max_updates,
        "early_stopping_patience": args.early_stopping_patience,
        "sampler": sampler_report,
        "provenance": provenance,
        "raw_evaluation": {
            "enabled": bool(raw_validation_rows),
            "validation": str(raw_validation_path.resolve()) if raw_validation_rows else None,
            "test": str(raw_test_path.resolve()) if raw_test_rows else None,
            "validation_reuses_test": provenance["validation_reuses_test"],
            "checkpoint_selection": (
                "raw_end_to_end_0.60_overlap_0.15_boundary_0.25_assertion"
                if raw_validation_rows else "legacy_window_0.80_bio_0.20_assertion"
            ),
        },
        "history": [],
    }

    def save_checkpoint(path: str, epoch: int, metrics: dict) -> None:
        model.save(path, tokenizer, {
            "labels": labels,
            "types": types,
            "assertions": assertions,
            "max_len": args.max_len,
            "init_checkpoint": run_report["init_checkpoint"],
            "selected_epoch": epoch,
            "selection_metrics": metrics,
            "sampler": sampler_report,
            "dataset_provenance": {
                "variant": (provenance.get("manifest") or {}).get("variant"),
                "part3_sha256": (
                    ((provenance.get("manifest") or {}).get("part3_artifact") or {}).get("sha256")
                ),
                "split_sha256": {
                    name: item["sha256"] for name, item in provenance["splits"].items()
                },
            },
        })

    global_updates = 0
    completed_epoch = 0
    stale_epochs = 0
    for epoch in range(int(math.ceil(args.epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.fp16 and device == "cuda"):
                result = model(**batch)
                loss = result["loss"] / args.grad_accumulation
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % args.grad_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_updates += 1
                if args.max_updates is not None and global_updates >= args.max_updates:
                    break
        metrics = evaluate(model, validation_loader, id_to_label, device)
        metrics["window_selection_score"] = metrics["selection_score"]
        if raw_validation_rows:
            end_to_end = evaluate_raw_documents(
                model, tokenizer,
                {"labels": labels, "types": types, "assertions": assertions},
                raw_validation_rows, device,
                max_len=args.max_len,
                max_words=args.eval_max_words,
                overlap_words=args.eval_overlap_words,
                prefix_subword_cap=args.prefix_subword_cap,
            )
            metrics["end_to_end"] = end_to_end
            metrics["selection_score"] = end_to_end["selection_score"]
        print(f"[epoch {epoch + 1}] {json.dumps(metrics)}", flush=True)
        run_report["history"].append({"epoch": epoch + 1, **metrics})
        last_metrics = metrics
        completed_epoch = epoch + 1
        if args.save_every_epoch:
            save_checkpoint(args.out + f"/epoch-{epoch + 1:03d}", epoch + 1, metrics)
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            best_epoch = epoch + 1
            save_checkpoint(args.out + "/best", epoch + 1, metrics)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.max_updates is not None and global_updates >= args.max_updates:
            break
        if args.early_stopping_patience and stale_epochs >= args.early_stopping_patience:
            print(
                f"[early-stop] {stale_epochs} epoch không cải thiện; best={best_epoch}",
                flush=True,
            )
            break
    if args.save_last:
        save_checkpoint(args.out + "/last", completed_epoch, last_metrics)
    run_report["best_epoch"] = best_epoch
    run_report["best_selection_score"] = best
    run_report["completed_epochs"] = completed_epoch
    run_report["completed_updates"] = global_updates
    run_report["stopped_early"] = completed_epoch < int(math.ceil(args.epochs))
    (Path(args.out) / "training_report.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    def test_checkpoint(name: str) -> dict:
        checkpoint = args.out + f"/{name}"
        evaluated_model, _ = HybridNER.load(checkpoint, device)
        metrics = evaluate(evaluated_model, test_loader, id_to_label, device)
        metrics["evaluated_checkpoint"] = name
        metrics["window_selection_score"] = metrics["selection_score"]
        if raw_test_rows:
            metrics["end_to_end"] = evaluate_raw_documents(
                evaluated_model, tokenizer,
                {"labels": labels, "types": types, "assertions": assertions},
                raw_test_rows, device,
                max_len=args.max_len,
                max_words=args.eval_max_words,
                overlap_words=args.eval_overlap_words,
                prefix_subword_cap=args.prefix_subword_cap,
            )
        (Path(args.out) / f"test_metrics_{name}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metrics

    best_metrics = test_checkpoint("best")
    # Backward-compatible alias, now explicitly self-identifying.
    (Path(args.out) / "test_metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.save_last:
        test_checkpoint("last")
    print(f"[test:best] {json.dumps(best_metrics)}")


if __name__ == "__main__":
    main()
