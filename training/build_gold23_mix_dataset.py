"""Build the mixed official-data experiment for a possible Part 2 + Part 3 private test.

This variant is deliberately different from N1/N2/N3:

* Part 1 is train-only;
* Part 2 and Part 3 are each split by document, with 80% in train and 20% in validation;
* all synthetic sources are train-only, after exact-text de-duplication against validation;
* the original ``groundtruth_part2`` labels are used for Part 2 (mechanical offset/schema
  cleaning only), while the current v67 artifact is used for Part 3;
* ``test.jsonl`` intentionally reuses the mixed validation slice because ``train_v2`` expects a
  test split.  The validation metrics are the selection signal; the duplicate test is not an
  independent estimate.

The purpose is to test the user's private-test hypothesis, not to replace N2/N3.
Candidates remain empty and are outside the optimization scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from business_rules.artifacts import current_labels_zip, manifest as artifact_manifest
from training.assertion_policy import filter_assertions
from training.build_dataset_v2 import build_track, record_to_windows
from training.build_final_n_datasets import (
    SEED,
    _describe,
    _drop_unwindowable,
    _split,
    _tag,
    read_jsonl,
    sha256,
)
from training.build_ner_corpus import load_labeled_dir, load_part3


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "datasets/final_n_sources_v67_20260804"
INDEPENDENT_AUGMENT_DIR = (
    ROOT / "datasets/final_n_augment/final_generalization_independent_v3_v67_9200_20260804"
)
PART1_ROOT = ROOT / "annotation/data/label_final/part1"
PART2_GROUNDTRUTH_ROOT = ROOT / "annotation/data/groundtruth_part2"
RAW_NAME = "ner_N4_gold23_mix_v3_v67_20260804"
WINDOW_NAME = "ner_N4_gold23_mix_v3_v67_windows_20260804"

# The masses are window-sampling masses, not raw-record proportions.  Official Part 2/3 get
# enough weight to make this experiment useful for a mixed private test, while synthetic data
# still supplies surface/context diversity.
SOURCE_MASSES = {
    "official_part1": 0.15,
    "official_part2": 0.20,
    "official_part3": 0.20,
    "broad": 0.15,
    "final_llm": 0.10,
    "augment": 0.15,
    "analogue": 0.05,
}


def sanitize(record: dict[str, Any], policy: str = "part3") -> dict[str, Any]:
    """Keep the representable schema and apply the chosen assertion convention."""
    output = deepcopy(record)
    output["entities"] = []
    for entity in record.get("entities") or ():
        output["entities"].append({
            "text": entity["text"],
            "position": list(entity["position"]),
            "type": entity["type"],
            "assertions": filter_assertions(
                entity["type"], entity.get("assertions") or (), policy
            ),
            "candidates": [],
            "_assertion_supervision": entity.get("_assertion_supervision", True),
        })
    output["candidates_included"] = False
    return output


def _normalised_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _dedupe_against_validation(
    records: Iterable[dict[str, Any]], validation: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove exact normalized-text copies of validation records from train sources."""
    heldout = {_normalised_text(row["text"]) for row in validation}
    kept, removed = [], []
    for row in records:
        if _normalised_text(row["text"]) in heldout:
            removed.append(row["id"])
        else:
            kept.append(row)
    return kept, removed


def _load_official() -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    stats: Counter = Counter()
    part1 = load_labeled_dir(
        PART1_ROOT / "notes", PART1_ROOT / "labels", "part1",
        mask_assertions=False, stats=stats,
    )
    # This is intentionally the BTC groundtruth directory, not label_final/gt2.  It lets this
    # experiment test the hypothesis that private data follows the native Part 2 convention.
    part2 = load_labeled_dir(
        PART2_GROUNDTRUTH_ROOT / "notes", PART2_GROUNDTRUTH_ROOT / "labels",
        "part2_groundtruth", mask_assertions=False, stats=stats,
    )
    part3 = load_part3(ROOT / "input_turn2", current_labels_zip(), stats)
    if (len(part1), len(part2), len(part3)) != (100, 100, 100):
        raise RuntimeError(
            f"Official split sai: part1={len(part1)}, part2_gt={len(part2)}, part3={len(part3)}"
        )
    clean_part3 = [sanitize(row, "part3") for row in part3]
    for row in clean_part3:
        row["part3_exposed"] = True
    return (
        [sanitize(row, "part3") for row in part1],
        [sanitize(row, "legacy") for row in part2],
        clean_part3,
        dict(stats),
    )


def _load_synthetic() -> tuple[list[dict], dict[str, int]]:
    """Load every v67 synthetic family; official heldout text is removed later."""
    broad = [sanitize(row) for row in read_jsonl(SOURCE_DIR / "broad_h.jsonl")]
    broad += [sanitize(row) for row in read_jsonl(SOURCE_DIR / "broad_independent.jsonl")]
    for row in broad:
        row["_sampling_group"] = "broad"

    llm = [sanitize(row) for row in read_jsonl(SOURCE_DIR / "final_llm_generalization.jsonl")]
    for row in llm:
        row["_sampling_group"] = "final_llm"

    augment = [
        sanitize(row) for row in read_jsonl(INDEPENDENT_AUGMENT_DIR / "augment.jsonl")
    ]
    for row in augment:
        row["_sampling_group"] = "augment"

    analogues = [
        sanitize(row) for row in read_jsonl(SOURCE_DIR / "selected_analogues.jsonl")
    ]
    for row in analogues:
        row["_sampling_group"] = "analogue"

    all_rows = broad + llm + augment + analogues
    all_rows, dropped = _drop_unwindowable(all_rows)
    by_group = Counter(row["_sampling_group"] for row in all_rows)
    return all_rows, {"loaded": len(broad) + len(llm) + len(augment) + len(analogues),
                      "kept": len(all_rows), "dropped_unwindowable": len(dropped),
                      "by_group": dict(by_group), "dropped_ids": dropped}


def build(out_root: Path = ROOT / "datasets") -> dict[str, Any]:
    part1, part2, part3, cleaning_stats = _load_official()
    part2_validation, part2_train = _split(part2, 20, "n4-gold23-part2-validation")
    part3_validation, part3_train = _split(part3, 20, "n4-gold23-part3-validation")
    validation = _tag(part2_validation, "validation_part2") + _tag(
        part3_validation, "validation_part3"
    )

    synthetic, synthetic_stats = _load_synthetic()
    train_official = (
        _tag(part1, "official_part1")
        + _tag(part2_train, "official_part2")
        + _tag(part3_train, "official_part3")
    )
    train_synthetic = synthetic
    train_all = train_official + train_synthetic
    train_all, duplicate_ids = _dedupe_against_validation(train_all, validation)
    # Keep the test input required by train_v2 while explicitly documenting that it is not an
    # independent test estimate.
    splits = {"train": train_all, "validation": validation, "test": validation}

    from training.build_final_n_datasets import _assert_clean_splits
    _assert_clean_splits(splits)

    raw_root = out_root / RAW_NAME / "track_a"
    for split, rows in splits.items():
        write_path = raw_root / f"{split}.jsonl"
        write_path.parent.mkdir(parents=True, exist_ok=True)
        with write_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "variant": "n4_gold23_mix",
        "purpose": "private_test_hypothesis_part2_plus_part3",
        "scope": "WER_and_assertion_only",
        "candidates_optimized": False,
        "seed": SEED,
        "split_policy": {
            "part1": "all_train_no_validation",
            "part2": "groundtruth_part2_80_train_20_validation",
            "part3": "v67_artifact_80_train_20_validation",
            "synthetic": "train_only_exact_normalized_text_dedup_against_validation",
        },
        "part2_label_source": str(PART2_GROUNDTRUTH_ROOT),
        "part2_assertion_convention": "legacy_native_after_mechanical_cleaning",
        "part3_artifact": artifact_manifest()["current"],
        "sampling_masses": SOURCE_MASSES,
        "synthetic_stats": synthetic_stats,
        "dedup_against_validation": {
            "removed_count": len(duplicate_ids),
            "removed_ids": duplicate_ids,
        },
        "cleaning_stats": cleaning_stats,
        "splits": {name: _describe(rows) for name, rows in splits.items()},
        "test_reuses_validation": True,
        "source_sha256": {
            "part1_label_manifest": sha256(ROOT / "annotation/data/label_final/manifest.json"),
            "part2_reviewed_manifest": sha256(PART2_GROUNDTRUTH_ROOT / "_reviewed.json"),
            "part3_labels": sha256(current_labels_zip()),
            "broad_h": sha256(SOURCE_DIR / "broad_h.jsonl"),
            "broad_independent": sha256(SOURCE_DIR / "broad_independent.jsonl"),
            "final_llm": sha256(SOURCE_DIR / "final_llm_generalization.jsonl"),
            "augment": sha256(INDEPENDENT_AUGMENT_DIR / "augment.jsonl"),
            "analogues": sha256(SOURCE_DIR / "selected_analogues.jsonl"),
        },
        "notes": [
            "Part 2 and Part 3 are intentionally present in train and validation; this is not a Part-3-heldout-only corpus.",
            "Part 2 uses original groundtruth_part2 rather than label_final/gt2.",
            "Validation is mixed and record-disjoint from train; test duplicates validation only because train_v2 requires test.jsonl.",
            "Part3-exposed synthetic analogues are retained as train data after exact normalized-text de-duplication; they are not validation records.",
        ],
    }
    (raw_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    windows_root = out_root / WINDOW_NAME / "track_a"
    report = build_track(
        raw_root, windows_root, 180, 45, 0.35, 0.35, 0.60, SEED,
        source_masses=SOURCE_MASSES,
    )
    unexpected = [
        error for error in report["coverage_errors"]
        if error.split(":", 1)[0] not in {"part1", "part2_groundtruth", "part3", "C"}
    ]
    if unexpected:
        raise RuntimeError(f"N4 có coverage error mới: {unexpected[:5]}")
    manifest["window_coverage"] = {
        "known_legacy_errors": len(report["coverage_errors"]),
        "new_final_source_errors": 0,
    }
    (windows_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    zip_path = ROOT / f"{WINDOW_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((out_root / WINDOW_NAME).rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_root / WINDOW_NAME))
    return {"manifest": manifest, "build_report": report, "zip": str(zip_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    print(json.dumps(build(args.out_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
