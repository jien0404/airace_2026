"""Assemble the final N source bank and three private/generalization corpora.

This build is intentionally separate from the historical H--M optimization stream.  N1 keeps
Part 3 test-only for an independent diagnostic; N2 uses a fixed 80/20 Part-3 split to select an
epoch; N3 puts all available official records in train and is refit for exactly that epoch.
Only WER/NER and assertion are in scope; candidates are always empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from business_rules.artifacts import current_labels_zip, manifest as artifact_manifest
from training.assertion_policy import filter_assertions
from training.build_dataset_v2 import build_track, record_to_windows
from training.build_ner_corpus import load_labeled_dir, load_part3


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260804
SOURCE_DIR = ROOT / "datasets/final_n_sources"
AUGMENT_DIR = ROOT / "datasets/final_n_augment/final_generalization_v1_8200_20260804"
INDEPENDENT_AUGMENT_DIR = (
    ROOT / "datasets/final_n_augment/final_generalization_independent_v1_8200_20260804"
)
H_TRAIN = ROOT / "datasets/ner_H_education_fpfix_20260803/track_a/train.jsonl"
INDEPENDENT_H_TRAIN = ROOT / "datasets/ner_H/track_a/train.jsonl"
M_TRAIN = ROOT / "datasets/ner_M_assertion_hardneg_20260804/track_a/train.jsonl"
LLM_SOURCE = SOURCE_DIR / "final_llm_generalization.jsonl"

TARGETED_PILOTS = (
    "part3_targeted_wer_patch_240_20260803",
    "part3_historical_scope_v2_train_reviewed_160_20260803",
    "part3_l_assertion_balanced_train_reviewed_232_20260803",
    "part3_m_assertion_hardneg_train_reviewed_166_20260804",
)

VARIANTS = {
    "n1": {
        "raw": "ner_N1_final_generalization_part3_heldout_20260804",
        "windows": "ner_N1_final_generalization_part3_heldout_windows_20260804",
        "part3": "test_only",
        "masses": {
            "official_independent": 0.40, "broad": 0.30,
            "final_llm": 0.10, "augment": 0.20,
        },
    },
    "n2": {
        "raw": "ner_N2_final_generalization_part3_split_20260804",
        "windows": "ner_N2_final_generalization_part3_split_windows_20260804",
        "part3": "train80_validation20",
        "masses": {
            "official_independent": 0.25, "official_part3": 0.25,
            "broad": 0.20, "final_llm": 0.10, "augment": 0.15, "analogue": 0.05,
        },
    },
    "n3": {
        "raw": "ner_N3_final_generalization_private_refit_20260804",
        "windows": "ner_N3_final_generalization_private_refit_windows_20260804",
        "part3": "all_train",
        "masses": {
            "official_independent": 0.25, "official_part3": 0.25,
            "broad": 0.20, "final_llm": 0.10, "augment": 0.15, "analogue": 0.05,
        },
    },
}


def variant_specs(suffix: str = "") -> dict[str, dict[str, Any]]:
    """Return artifact names, optionally inserting a version suffix in Final N names."""
    if not suffix:
        return deepcopy(VARIANTS)
    marker = f"final_generalization_{suffix}"
    output = {}
    for key, spec in VARIANTS.items():
        item = deepcopy(spec)
        item["raw"] = item["raw"].replace("final_generalization", marker)
        item["windows"] = item["windows"].replace("final_generalization", marker)
        output[key] = item
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(record: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(record)
    output["entities"] = []
    for entity in record.get("entities") or ():
        item = {
            "text": entity["text"], "position": list(entity["position"]),
            "type": entity["type"],
            "assertions": filter_assertions(
                entity["type"], entity.get("assertions") or (), "part3"
            ),
            "candidates": [],
            "_assertion_supervision": entity.get("_assertion_supervision", True),
        }
        output["entities"].append(item)
    output["candidates_included"] = False
    return output


def _direct_request_ids() -> set[str]:
    direct = set()
    for pilot_name in TARGETED_PILOTS:
        path = ROOT / "datasets/pilots" / pilot_name / "requests.jsonl"
        for request in read_jsonl(path):
            targeted = (
                (request.get("generation_brief") or {}).get("targeted_error")
                or request.get("targeted_error") or {}
            )
            if targeted.get("direct_replay"):
                direct.add(f"{pilot_name}:{request['request_id']}")
    return direct


def prepare_sources(out_dir: Path = SOURCE_DIR) -> dict[str, Any]:
    broad = [
        sanitize(row) for row in read_jsonl(H_TRAIN)
        if row.get("source") == "synthetic_track_a"
    ]
    for row in broad:
        row.update({
            "source": "final_broad_h", "part3_exposed": False,
            "donor_group": row.get("id"), "_sampling_group": "broad",
        })
    broad_independent = [
        sanitize(row) for row in read_jsonl(INDEPENDENT_H_TRAIN)
        if row.get("source") == "synthetic_track_a"
    ]
    for row in broad_independent:
        row.update({
            "source": "final_broad_independent", "part3_exposed": False,
            "donor_group": row.get("id"), "_sampling_group": "broad",
        })

    direct_ids = _direct_request_ids()
    targeted_all = [
        sanitize(row) for row in read_jsonl(M_TRAIN)
        if row.get("source") == "synthetic_targeted_part3"
    ]
    analogues = []
    for row in targeted_all:
        if row["id"] in direct_ids:
            continue
        row.update({
            "source": "final_selected_analogue", "part3_exposed": True,
            "donor_group": row.get("id"), "_sampling_group": "analogue",
        })
        analogues.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "broad_h.jsonl", broad)
    write_jsonl(out_dir / "broad_independent.jsonl", broad_independent)
    write_jsonl(out_dir / "selected_analogues.jsonl", analogues)
    manifest = {
        "schema_version": 1,
        "stage": "final_n_source_bank_prepared",
        "broad_h": len(broad),
        "broad_independent": len(broad_independent),
        "targeted_all": len(targeted_all),
        "selected_analogues": len(analogues),
        "direct_replay_excluded": len(targeted_all) - len(analogues),
        "candidates_included": False,
        "scope": "WER_and_assertion_only",
        "inputs": {
            "H_private": str(H_TRAIN), "H_independent": str(INDEPENDENT_H_TRAIN),
            "M": str(M_TRAIN),
        },
    }
    (out_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_official() -> tuple[list[dict], list[dict], list[dict]]:
    stats: Counter = Counter()
    final_root = ROOT / "annotation/data/label_final"
    part1 = load_labeled_dir(
        final_root / "part1/notes", final_root / "part1/labels", "part1",
        mask_assertions=False, stats=stats,
    )
    part2 = load_labeled_dir(
        final_root / "gt2/notes", final_root / "gt2/labels", "part2",
        mask_assertions=False, stats=stats,
    )
    part3 = load_part3(ROOT / "input_turn2", current_labels_zip(), stats)
    if (len(part1), len(part2), len(part3)) != (100, 100, 100):
        raise RuntimeError(f"Official split sai: {len(part1)}/{len(part2)}/{len(part3)}")
    clean_part3 = [sanitize(row) for row in part3]
    for row in clean_part3:
        row["part3_exposed"] = True
    return [sanitize(row) for row in part1], [sanitize(row) for row in part2], clean_part3


def _split(records: list[dict], count: int, salt: str) -> tuple[list[dict], list[dict]]:
    rows = sorted(records, key=lambda row: row["id"])
    random.Random(f"{SEED}:{salt}").shuffle(rows)
    return rows[:count], rows[count:]


def _tag(records: Iterable[dict], group: str) -> list[dict]:
    output = []
    for record in records:
        row = deepcopy(record)
        row["_sampling_group"] = group
        output.append(row)
    return output


def _drop_unwindowable(records: list[dict]) -> tuple[list[dict], list[str]]:
    kept, dropped = [], []
    for record in records:
        _, errors = record_to_windows(record, 180, 45, 0.35, 0.35, SEED)
        if errors:
            dropped.append(record["id"])
        else:
            kept.append(record)
    return kept, dropped


def _assert_clean_splits(splits: dict[str, list[dict]]) -> None:
    id_sets = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    for left, right in (("train", "validation"), ("train", "test")):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            raise RuntimeError(f"Leak ID {left}/{right}: {sorted(overlap)[:3]}")
    for name, rows in splits.items():
        for row in rows:
            text = row["text"]
            for entity in row.get("entities") or ():
                start, end = entity["position"]
                if text[start:end] != entity["text"]:
                    raise RuntimeError(f"Offset lệch {name}:{row['id']}:{entity}")
                if entity.get("candidates"):
                    raise RuntimeError(f"Candidate lọt vào {name}:{row['id']}")


def _describe(rows: list[dict]) -> dict[str, Any]:
    entities = [entity for row in rows for entity in row.get("entities") or ()]
    return {
        "records": len(rows), "entities": len(entities),
        "asserted": sum(bool(entity.get("assertions")) for entity in entities),
        "by_group": dict(Counter(row.get("_sampling_group") for row in rows)),
        "by_source": dict(Counter(row.get("source") for row in rows)),
        "part3_exposed": sum(bool(row.get("part3_exposed")) for row in rows),
    }


def build_corpora(
    source_dir: Path = SOURCE_DIR,
    augment_dir: Path = AUGMENT_DIR,
    independent_augment_dir: Path = INDEPENDENT_AUGMENT_DIR,
    datasets_dir: Path = ROOT / "datasets",
    variants: dict[str, dict[str, Any]] | None = None,
    summary_name: str = "final_n_build_summary.json",
) -> dict[str, Any]:
    broad = read_jsonl(source_dir / "broad_h.jsonl")
    broad_independent = read_jsonl(source_dir / "broad_independent.jsonl")
    analogues = read_jsonl(source_dir / "selected_analogues.jsonl")
    # The source bank is versioned together with the artifact/data update.  Do
    # not silently fall back to the old global source when ``--source-dir`` is
    # a new directory; that would make the manifest claim one source bank while
    # the records came from another one.
    llm_source = source_dir / "final_llm_generalization.jsonl"
    if not llm_source.exists():
        raise FileNotFoundError(
            f"Thiếu final_llm_generalization.jsonl trong source bank: {llm_source}"
        )
    llm = [sanitize(row) for row in read_jsonl(llm_source)]
    for row in llm:
        row["_sampling_group"] = "final_llm"
    augment = [sanitize(row) for row in read_jsonl(augment_dir / "augment.jsonl")]
    for row in augment:
        row["_sampling_group"] = "augment"
    independent_augment = [
        sanitize(row) for row in read_jsonl(independent_augment_dir / "augment.jsonl")
    ]
    for row in independent_augment:
        row["_sampling_group"] = "augment"
    llm, dropped_llm = _drop_unwindowable(llm)
    augment, dropped_augment = _drop_unwindowable(augment)
    independent_augment, dropped_independent_augment = _drop_unwindowable(
        independent_augment
    )
    part1, part2, part3 = load_official()

    llm_validation, llm_train = _split(llm, 160, "llm-validation")
    heldout_groups = {row.get("donor_group") for row in llm_validation}
    augment_train_safe = [row for row in augment if row.get("donor_group") not in heldout_groups]
    independent_augment_train_safe = [
        row for row in independent_augment if row.get("donor_group") not in heldout_groups
    ]
    part2_validation, part2_train = _split(part2, 15, "part2-validation")
    part3_validation, part3_train = _split(part3, 20, "part3-validation")

    outputs = {}
    for key, spec in (variants or VARIANTS).items():
        if key == "n1":
            train = (
                _tag(part1 + part2_train, "official_independent")
                + _tag(broad_independent, "broad") + _tag(llm_train, "final_llm")
                + _tag(independent_augment_train_safe, "augment")
            )
            validation = part2_validation + llm_validation
            test = part3
        elif key == "n2":
            train = (
                _tag(part1 + part2, "official_independent")
                + _tag(part3_train, "official_part3")
                + _tag(broad, "broad") + _tag(llm, "final_llm")
                + _tag(augment, "augment") + _tag(analogues, "analogue")
            )
            validation = part3_validation
            test = part3_validation
        else:
            train = (
                _tag(part1 + part2, "official_independent")
                + _tag(part3, "official_part3")
                + _tag(broad, "broad") + _tag(llm_train, "final_llm")
                + _tag(augment_train_safe, "augment") + _tag(analogues, "analogue")
            )
            validation = llm_validation
            test = llm_validation
        splits = {"train": train, "validation": validation, "test": test}
        _assert_clean_splits(splits)
        raw_root = datasets_dir / spec["raw"] / "track_a"
        for split, rows in splits.items():
            write_jsonl(raw_root / f"{split}.jsonl", rows)
        manifest = {
            "schema_version": 1, "variant": key,
            "independent_from_H_to_M_optimization": True,
            "scope": "WER_and_assertion_only", "candidates_optimized": False,
            "part3_policy": spec["part3"], "seed": SEED,
            "dropped_unwindowable_final_records": {
                "llm": dropped_llm,
                "augment": (
                    dropped_independent_augment if key == "n1" else dropped_augment
                ),
            },
            "sampling_masses": spec["masses"],
            "splits": {name: _describe(rows) for name, rows in splits.items()},
            "source_sha256": {
                "broad": sha256(source_dir / "broad_h.jsonl"),
                "llm": sha256(llm_source),
                "augment": sha256(augment_dir / "augment.jsonl"),
                "analogues": sha256(source_dir / "selected_analogues.jsonl"),
                "part3_labels": sha256(current_labels_zip()),
                "part1_part2_label_manifest": sha256(
                    ROOT / "annotation/data/label_final/manifest.json"
                ),
            },
            "part3_artifact": artifact_manifest()["current"],
            "augment_manifest": str(
                (independent_augment_dir if key == "n1" else augment_dir) / "manifest.json"
            ),
        }
        if key == "n1":
            manifest["source_sha256"]["broad"] = sha256(
                source_dir / "broad_independent.jsonl"
            )
            manifest["source_sha256"]["augment"] = sha256(
                independent_augment_dir / "augment.jsonl"
            )
        (raw_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        windows_root = datasets_dir / spec["windows"] / "track_a"
        report = build_track(
            raw_root, windows_root, 180, 45, 0.35, 0.35, 0.60, SEED,
            source_masses=spec["masses"],
        )
        # H--M đã có đúng 82 omission do một số span official/quá dài không nằm trọn trong
        # sliding window và 7 record broad C cũ. Final source mới tuyệt đối không được tạo lỗi
        # coverage mới; không giả vờ rằng lỗi di sản đã biến mất.
        unexpected_coverage = [
            error for error in report["coverage_errors"]
            if error.split(":", 1)[0] not in {"part1", "part2", "part3", "C"}
        ]
        if unexpected_coverage:
            raise RuntimeError(f"{key} có coverage error mới: {unexpected_coverage[:5]}")
        manifest["window_coverage"] = {
            "known_legacy_errors": len(report["coverage_errors"]),
            "new_final_source_errors": 0,
        }
        (windows_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        zip_path = ROOT / f"{spec['windows']}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((datasets_dir / spec["windows"]).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(datasets_dir / spec["windows"]))
        outputs[key] = {"manifest": manifest, "build_report": report, "zip": str(zip_path)}
    summary_path = datasets_dir / summary_name
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("prepare-sources")
    sources.add_argument("--out", type=Path, default=SOURCE_DIR)
    corpora = sub.add_parser("build-corpora")
    corpora.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    corpora.add_argument("--augment-dir", type=Path, default=AUGMENT_DIR)
    corpora.add_argument(
        "--independent-augment-dir", type=Path, default=INDEPENDENT_AUGMENT_DIR,
    )
    corpora.add_argument("--datasets-dir", type=Path, default=ROOT / "datasets")
    corpora.add_argument(
        "--variant-suffix", default="",
        help="suffix chen vao final_generalization trong ten artifact, vi du v2",
    )
    args = parser.parse_args()
    if args.command == "prepare-sources":
        result = prepare_sources(args.out.resolve())
    else:
        result = build_corpora(
            args.source_dir.resolve(), args.augment_dir.resolve(),
            args.independent_augment_dir.resolve(), args.datasets_dir.resolve(),
            variants=variant_specs(args.variant_suffix),
            summary_name=(
                "final_n_build_summary.json"
                if not args.variant_suffix else f"final_n_build_summary_{args.variant_suffix}.json"
            ),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
