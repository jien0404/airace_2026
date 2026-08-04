"""Build independent diagnostic windows from an analogue-only targeted pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .augment_ner_corpus import _pilot_gate, _targeted_records
from .build_dataset_v2 import record_to_windows


def build(pilot: Path, out: Path, *, max_words: int, overlap_words: int, seed: int) -> dict:
    checked = _pilot_gate(pilot)
    mode = checked["manifest"].get("mode")
    if mode not in {"historical", "k_wer", "k_assertion", "l_assertion", "m_assertion"}:
        raise RuntimeError(f"Pilot dev không phải targeted diagnostic mode: {pilot}")
    if checked["manifest"].get("direct_replay", 0):
        raise RuntimeError("Historical dev phải independent: direct_replay bắt buộc bằng 0")
    records, corrections = _targeted_records(
        pilot, checked["drafts"], mask_assertions=mode == "k_wer"
    )
    windows = []
    coverage_errors = []
    for record in records:
        built, errors = record_to_windows(
            record,
            max_words=max_words,
            overlap_words=overlap_words,
            header_dropout=0.0,
            o_keep=1.0,
            seed=seed,
        )
        windows.extend(built)
        coverage_errors.extend(errors)
    if coverage_errors:
        raise RuntimeError(f"Historical dev có entity không được cover: {coverage_errors[:10]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for row in windows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": 1,
        "pilot": str(pilot.resolve()),
        "records": len(records),
        "windows": len(windows),
        "entities": sum(len(row["spans"]) for row in windows),
        "direct_replay": 0,
        "mode": mode,
        "assertion_supervision_masked": mode == "k_wer",
        "assertion_corrections_at_ingest": dict(corrections),
        "max_words": max_words,
        "overlap_words": overlap_words,
    }
    out.with_suffix(out.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    report = build(
        Path(args.pilot),
        Path(args.out),
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
