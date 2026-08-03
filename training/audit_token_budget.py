"""Measure train/infer subword truncation before changing the model window length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .train_v2 import read_jsonl


def _subword_length(token: str, tokenizer, cache: dict[str, int]) -> int:
    if token not in cache:
        cache[token] = max(1, len(tokenizer.encode(token, add_special_tokens=False)))
    return cache[token]


def word_limit_for_row(
    row: dict, tokenizer, max_len: int, token_lengths: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Return (number of complete words retained, encoded length including BOS/EOS)."""
    used = 2
    retained = 0
    token_lengths = token_lengths if token_lengths is not None else {}
    for token in row["tokens"]:
        length = _subword_length(token, tokenizer, token_lengths)
        if used + length > max_len:
            break
        used += length
        retained += 1
    return retained, used


def audit_rows(
    rows: list[dict], tokenizer, max_len: int, token_lengths: dict[str, int] | None = None,
) -> dict:
    token_lengths = token_lengths if token_lengths is not None else {}
    truncated = clipped_spans = total_spans = prefix_truncated = 0
    examples = []
    encoded_lengths = []
    for row in rows:
        word_limit, encoded_length = word_limit_for_row(
            row, tokenizer, max_len, token_lengths
        )
        full_length = 2 + sum(
            _subword_length(token, tokenizer, token_lengths)
            for token in row["tokens"]
        )
        encoded_lengths.append(full_length)
        is_truncated = word_limit < len(row["tokens"])
        truncated += int(is_truncated)
        prefix_truncated += int(word_limit < int(row.get("n_prefix", 0)))
        row_clipped = 0
        for span in row.get("spans") or []:
            if int(span.get("type_id", 0)) <= 0:
                continue
            total_spans += 1
            if int(span["end"]) > word_limit:
                clipped_spans += 1
                row_clipped += 1
        if (is_truncated or row_clipped) and len(examples) < 25:
            examples.append({
                "id": row.get("id"),
                "record_id": row.get("record_id"),
                "words": len(row["tokens"]),
                "retained_words": word_limit,
                "subwords_with_specials": full_length,
                "retained_subwords_with_specials": encoded_length,
                "clipped_spans": row_clipped,
            })
    lengths = sorted(encoded_lengths)

    def percentile(fraction: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, int((len(lengths) - 1) * fraction))]

    return {
        "max_len": max_len,
        "windows": len(rows),
        "truncated_windows": truncated,
        "truncated_window_ratio": truncated / max(1, len(rows)),
        "prefix_truncated_windows": prefix_truncated,
        "labeled_spans": total_spans,
        "clipped_labeled_spans": clipped_spans,
        "clipped_labeled_span_ratio": clipped_spans / max(1, total_spans),
        "encoded_length_p50": percentile(0.50),
        "encoded_length_p90": percentile(0.90),
        "encoded_length_p99": percentile(0.99),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Thư mục window dataset track_a")
    parser.add_argument("--model", required=True, help="Tokenizer/checkpoint tương ứng")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--candidate-max-len", type=int, default=384)
    parser.add_argument("--truncated-window-gate", type=float, default=0.10)
    parser.add_argument("--clipped-span-gate", type=float, default=0.005)
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    data_dir = Path(args.data)
    report = {
        "schema_version": 1,
        "data": str(data_dir.resolve()),
        "model": args.model,
        "baseline": {},
        "candidate": {},
        "gates": {
            "truncated_window_ratio": args.truncated_window_gate,
            "clipped_labeled_span_ratio": args.clipped_span_gate,
        },
    }
    token_lengths: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        report["baseline"][split] = audit_rows(
            rows, tokenizer, args.max_len, token_lengths
        )
        report["candidate"][split] = audit_rows(
            rows, tokenizer, args.candidate_max_len, token_lengths
        )
    baseline = report["baseline"]
    report["recommend_matched_long_window"] = any(
        stats["truncated_window_ratio"] > args.truncated_window_gate
        or stats["clipped_labeled_span_ratio"] > args.clipped_span_gate
        for stats in baseline.values()
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
