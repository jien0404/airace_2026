from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .model_v2 import HybridNER


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


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
    ner_f1 = f1_score(true_bio, pred_bio)
    precision = a_tp / max(1, a_tp + a_fp)
    recall = a_tp / max(1, a_tp + a_fn)
    assertion_f1 = 2 * precision * recall / max(1e-9, precision + recall)
    report = classification_report(true_bio, pred_bio, output_dict=True, zero_division=0)
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


def main():
    parser = argparse.ArgumentParser(description="Train hybrid BIO+span+assertion NER")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--span-weight", type=float, default=0.5)
    parser.add_argument("--assertion-weight", type=float, default=1.0)
    parser.add_argument("--negative-span-ratio", type=float, default=2.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    labels = (data_dir / "labels.txt").read_text(encoding="utf-8").splitlines()
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    validation_rows = read_jsonl(data_dir / "validation.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    train_data = HybridDataset(
        train_rows, tokenizer, label_to_id, args.max_len, args.negative_span_ratio, seed=args.seed
    )
    validation_data = HybridDataset(
        validation_rows, tokenizer, label_to_id, args.max_len, args.negative_span_ratio, seed=args.seed
    )
    test_data = HybridDataset(
        test_rows, tokenizer, label_to_id, args.max_len, args.negative_span_ratio, seed=args.seed
    )
    weights = [float(row.get("sample_weight", 1.0)) for row in train_rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
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
    model = HybridNER(
        args.model,
        len(labels),
        span_weight=args.span_weight,
        assertion_weight=args.assertion_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, int(len(train_loader) * args.epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    best = -1.0
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(int(math.ceil(args.epochs))):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.fp16):
                result = model(**batch)
            scaler.scale(result["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        metrics = evaluate(model, validation_loader, id_to_label, device)
        print(f"[epoch {epoch + 1}] {json.dumps(metrics)}", flush=True)
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            model.save(args.out + "/best", tokenizer, {
                "labels": labels,
                "types": (data_dir / "types.txt").read_text(encoding="utf-8").splitlines(),
                "assertions": (data_dir / "assertions.txt").read_text(encoding="utf-8").splitlines(),
                "max_len": args.max_len,
            })
    best_model, meta = HybridNER.load(args.out + "/best", device)
    metrics = evaluate(best_model, test_loader, id_to_label, device)
    (Path(args.out) / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[test] {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
