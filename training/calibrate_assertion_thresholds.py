"""Calibrate sparse assertion thresholds by entity type on an independent window dev set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .model_v2 import HybridNER
from .train_v2 import HybridDataset, collate, read_jsonl


def _metrics(probabilities: list[float], gold: list[int], threshold: float) -> dict[str, float | int]:
    tp = fp = fn = 0
    for probability, expected in zip(probabilities, gold):
        predicted = probability >= threshold
        tp += int(predicted and expected == 1)
        fp += int(predicted and expected == 0)
        fn += int(not predicted and expected == 1)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _grid(low: float, high: float, step: float) -> list[float]:
    values = []
    current = low
    while current <= high + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


@torch.no_grad()
def collect(
    model: HybridNER,
    loader: DataLoader,
    assertion_names: list[str],
    type_names: list[str],
    device: str,
) -> dict[tuple[str, str], tuple[list[float], list[int]]]:
    values: dict[tuple[str, str], tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    model.eval()
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        result = model(
            batch["input_ids"], batch["attention_mask"],
            span_starts=batch["span_starts"], span_ends=batch["span_ends"],
        )
        probabilities = torch.sigmoid(result["assertion_logits"]).cpu()
        labels = batch["assertion_labels"].long().cpu()
        masks = batch["assertion_mask"].bool().cpu()
        span_types = batch["span_labels"].long().cpu()
        for row in range(probabilities.shape[0]):
            for span in range(probabilities.shape[1]):
                type_id = int(span_types[row, span])
                if not masks[row, span] or not 1 <= type_id <= len(type_names):
                    continue
                entity_type = type_names[type_id - 1]
                for assertion_index, assertion in enumerate(assertion_names):
                    pair = values[(assertion, entity_type)]
                    pair[0].append(float(probabilities[row, span, assertion_index]))
                    pair[1].append(int(labels[row, span, assertion_index]))
    return dict(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev", required=True, help="Reviewed analogue-only window JSONL")
    parser.add_argument("--out-map", required=True)
    parser.add_argument("--out-report")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--global-threshold", type=float, default=0.60)
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.85)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if not (0 <= args.low <= args.high <= 1 and args.step > 0):
        raise SystemExit("Khoảng threshold không hợp lệ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = HybridNER.load(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    labels = list(meta["labels"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows = read_jsonl(Path(args.dev))
    dataset = HybridDataset(
        rows, tokenizer, label_to_id, max_len=int(meta["max_len"]), seed=args.seed
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: collate(batch, dataset.pad),
    )
    assertion_names = list(meta["assertions"])
    type_names = list(meta["types"])
    collected = collect(model, loader, assertion_names, type_names, device)
    thresholds: dict[str, dict[str, float]] = defaultdict(dict)
    pair_reports = []
    candidates = _grid(args.low, args.high, args.step)
    for (assertion, entity_type), (probabilities, gold) in sorted(collected.items()):
        positives = sum(gold)
        allowed = not (
            assertion == "isFamily"
            or assertion == "isNegated" and entity_type == "CHẨN_ĐOÁN"
        )
        rows_for_pair = [
            {"threshold": threshold, **_metrics(probabilities, gold, threshold)}
            for threshold in candidates
        ]
        selected = max(
            rows_for_pair,
            key=lambda row: (
                row["f1"], row["precision"],
                -abs(float(row["threshold"]) - args.global_threshold),
            ),
        )
        eligible = allowed and positives >= args.min_positives
        if eligible:
            thresholds[assertion][entity_type] = float(selected["threshold"])
        pair_reports.append({
            "assertion": assertion,
            "entity_type": entity_type,
            "examples": len(gold),
            "positives": positives,
            "eligible": eligible,
            "selected": selected if eligible else None,
            "reason_skipped": None if eligible else (
                "blocked_by_part3_firewall" if not allowed else "too_few_positives"
            ),
            "global_metrics": _metrics(probabilities, gold, args.global_threshold),
            "grid": rows_for_pair,
        })
    sparse_map = {name: mapping for name, mapping in thresholds.items() if mapping}
    out_map = Path(args.out_map)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    out_map.write_text(json.dumps(sparse_map, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dev": str(Path(args.dev).resolve()),
        "dev_windows": len(rows),
        "global_threshold": args.global_threshold,
        "min_positives": args.min_positives,
        "selected_sparse_map": sparse_map,
        "selection_rule": "max F1, then precision, then nearest global threshold",
        "part3_firewall_applied_at_inference": True,
        "candidates_optimized": False,
        "pairs": pair_reports,
    }
    out_report = Path(args.out_report) if args.out_report else out_map.with_suffix(".report.json")
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
