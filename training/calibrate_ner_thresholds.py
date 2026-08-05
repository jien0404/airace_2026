"""Coordinate-calibrate NER span-veto/disagreement thresholds on independent raw dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .model_v2 import HybridNER
from .provenance import verify_dataset
from .schema_v2 import TYPES
from .train_v2 import evaluate_raw_documents, read_jsonl


PART3_SHA256 = "e803003cee5a43d740bf4024e65b82d7a6a7214389e3338c0d83352333959b9b"


def values(low: float, high: float, step: float) -> list[float]:
    output = []
    current = low
    while current <= high + 1e-9:
        output.append(round(current, 6))
        current += step
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--out-map", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--low", type=float, default=0.50)
    parser.add_argument("--high", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.10)
    parser.add_argument("--type", action="append", choices=TYPES)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    provenance = verify_dataset(
        args.dataset.resolve(),
        require_manifest=True,
        allowed_variants=("n1",),
        required_part3_sha256=PART3_SHA256,
    )
    dev_path = args.dev.resolve() if args.dev else args.dataset.resolve() / "raw_validation.jsonl"
    records = read_jsonl(dev_path)
    exposed = [row.get("id") for row in records if row.get("part3_exposed") or row.get("source") == "part3"]
    if exposed:
        raise SystemExit(f"Calibration dev không độc lập Part3: {exposed[:5]}")
    if provenance["validation_reuses_test"]:
        raise SystemExit("Calibration bị chặn: window validation và test trùng SHA")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    checkpoint = args.checkpoint.resolve()
    model, meta = HybridNER.load(str(checkpoint), device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=False)
    candidates = values(args.low, args.high, args.step)
    thresholds = {"span_veto": {}, "disagreement": {}}
    trials = []

    def evaluate(mapping: dict) -> dict:
        return evaluate_raw_documents(
            model, tokenizer, meta, records, device,
            max_len=int(meta.get("max_len") or 256),
            overlap_words=45,
            ner_type_thresholds=mapping,
        )

    baseline = evaluate(thresholds)
    trials.append({"group": "baseline", "selection_score": baseline["selection_score"]})
    for group in ("span_veto", "disagreement"):
        for entity_type in args.type or list(TYPES):
            best_value = 0.60 if group == "span_veto" else 0.85
            best_metrics = None
            for candidate in candidates:
                trial_map = {name: dict(items) for name, items in thresholds.items()}
                trial_map[group][entity_type] = candidate
                metrics = evaluate(trial_map)
                trials.append({
                    "group": group,
                    "entity_type": entity_type,
                    "threshold": candidate,
                    "selection_score": metrics["selection_score"],
                    "overall_selection_score": metrics["overall_selection_score"],
                })
                if best_metrics is None or metrics["selection_score"] > best_metrics["selection_score"]:
                    best_value, best_metrics = candidate, metrics
            thresholds[group][entity_type] = best_value

    final = evaluate(thresholds)
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_map.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_report.write_text(json.dumps({
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "dataset_provenance": provenance,
        "dev": str(dev_path),
        "part3_exposed_records": 0,
        "objective": "raw_end_to_end_macro_by_source",
        "baseline": baseline,
        "selected_thresholds": thresholds,
        "final": final,
        "trials": trials,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_map": str(args.out_map),
        "baseline": baseline["selection_score"],
        "final": final["selection_score"],
        "thresholds": thresholds,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
