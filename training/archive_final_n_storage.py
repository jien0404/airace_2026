"""Archive raw Final N v2 data without train/validation/test splits.

The archive is a storage/provenance artifact only. It keeps the accepted LLM data, both
provenance-specific deterministic augment branches, and the exact sanitized Part 1/Part 2
records used by the Final N builder. No model dataset split is created here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from training.build_ner_corpus import load_labeled_dir
from training.build_final_n_datasets import sanitize
from training.windowing_v2 import tokenize_raw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "datasets/archives/final_n_v2_storage_20260804"
LLM_SOURCE = ROOT / "datasets/final_n_sources/final_llm_generalization.jsonl"
LLM_PILOT = ROOT / "datasets/pilots/final_generalization_v1_800_20260804"
AUGMENT_PRIVATE = ROOT / "datasets/final_n_augment/final_generalization_v2_9200_20260804"
AUGMENT_INDEPENDENT = (
    ROOT / "datasets/final_n_augment/final_generalization_independent_v2_9200_20260804"
)
PART1_ROOT = ROOT / "annotation/data/label_final/part1"
PART2_ROOT = ROOT / "annotation/data/label_final/gt2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def quantile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def record_stats(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    chars = []
    raw_tokens = []
    entity_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    augment_counts: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    texts: Counter[str] = Counter()
    offset_errors = []
    overlap_errors = []
    candidates = []
    asserted_entities = 0
    entity_total = 0
    negative_total = 0
    part3_exposed = 0

    for row in rows:
        record_id = str(row.get("id") or "")
        text = str(row.get("text") or "")
        ids[record_id] += 1
        texts[text] += 1
        chars.append(len(text))
        raw_tokens.append(len(tokenize_raw(text)))
        source_counts[str(row.get("source") or "unknown")] += 1
        context_counts[str(
            row.get("context_family") or row.get("genre") or "unknown"
        )] += 1
        augment_counts[str(row.get("primary_augmentation") or "none")] += 1
        part3_exposed += int(bool(row.get("part3_exposed")))
        if row.get("candidates_included") or any(
            entity.get("candidates") for entity in row.get("entities") or ()
        ):
            candidates.append(record_id)

        spans = []
        for index, entity in enumerate(row.get("entities") or ()):
            entity_total += 1
            entity_type = str(entity.get("type") or "unknown")
            entity_counts[entity_type] += 1
            assertions = list(entity.get("assertions") or ())
            if assertions:
                asserted_entities += 1
            for assertion in assertions:
                assertion_counts[str(assertion)] += 1
            position = entity.get("position")
            if not (
                isinstance(position, list) and len(position) == 2
                and all(isinstance(value, int) for value in position)
            ):
                offset_errors.append(f"{record_id}:entity[{index}]:position")
                continue
            start, end = position
            if not (0 <= start < end <= len(text)) or text[start:end] != entity.get("text"):
                offset_errors.append(f"{record_id}:entity[{index}]:offset")
            spans.append((start, end, index))
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if left[1] > right[0]:
                overlap_errors.append(f"{record_id}:{left[2]}/{right[2]}")

        for negative in row.get("intentional_negatives") or ():
            negative_total += 1
            negative_counts[str(negative.get("semantic_type") or "unknown")] += 1

    return {
        "name": name,
        "records": len(rows),
        "entities": entity_total,
        "asserted_entities": asserted_entities,
        "assertion_labels": dict(sorted(assertion_counts.items())),
        "entities_by_type": dict(sorted(entity_counts.items())),
        "intentional_negatives": negative_total,
        "intentional_negatives_by_type": dict(sorted(negative_counts.items())),
        "part3_exposed_records": part3_exposed,
        "candidates_records": len(candidates),
        "candidates_record_ids_sample": candidates[:20],
        "offset_errors": len(offset_errors),
        "offset_error_sample": offset_errors[:20],
        "overlap_errors": len(overlap_errors),
        "overlap_error_sample": overlap_errors[:20],
        "duplicate_ids": sum(count - 1 for count in ids.values() if count > 1),
        "duplicate_texts": sum(count - 1 for count in texts.values() if count > 1),
        "sources": dict(sorted(source_counts.items())),
        "context_families": dict(sorted(context_counts.items())),
        "augmentation_kinds": dict(sorted(augment_counts.items())),
        "text_length_chars": {
            "min": min(chars) if chars else None,
            "median": quantile(chars, 0.50),
            "p90": quantile(chars, 0.90),
            "max": max(chars) if chars else None,
        },
        "text_length_raw_tokens": {
            "min": min(raw_tokens) if raw_tokens else None,
            "median": quantile(raw_tokens, 0.50),
            "p90": quantile(raw_tokens, 0.90),
            "max": max(raw_tokens) if raw_tokens else None,
        },
    }


def load_official_part12() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats: Counter = Counter()
    part1 = load_labeled_dir(
        PART1_ROOT / "notes", PART1_ROOT / "labels", "part1",
        mask_assertions=False, stats=stats,
    )
    part2 = load_labeled_dir(
        PART2_ROOT / "notes", PART2_ROOT / "labels", "part2",
        mask_assertions=False, stats=stats,
    )
    return [sanitize(row) for row in part1], [sanitize(row) for row in part2]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_directory_files(source: Path, target: Path) -> list[str]:
    copied = []
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = target / path.name
        shutil.copy2(path, destination)
        copied.append(path.name)
    return copied


def write_stats(path: Path, rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    stats = record_stats(rows, name)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def build_archive(out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Archive đã tồn tại: {out_dir}; dùng --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = out_dir / "generated_llm"
    generated_dir.mkdir(parents=True, exist_ok=True)
    llm_rows = read_jsonl(LLM_SOURCE)
    shutil.copy2(LLM_SOURCE, generated_dir / "data.jsonl")
    for name in (
        "requests.jsonl", "drafts.jsonl", "audit_report.json",
        "semantic_audit_report.json", "semantic_manual_rejections.json",
        "pilot_manifest.json",
    ):
        copy_if_exists(LLM_PILOT / name, generated_dir / name)
    llm_stats = write_stats(generated_dir / "stats.json", llm_rows, "generated_llm")

    augment_dir = out_dir / "deterministic_augment"
    augment_dir.mkdir(parents=True, exist_ok=True)
    augment_stats = {}
    augment_sources = {
        "private_v2": AUGMENT_PRIVATE,
        "independent_v2": AUGMENT_INDEPENDENT,
    }
    for branch, source_dir in augment_sources.items():
        branch_dir = augment_dir / branch
        copied = copy_directory_files(source_dir, branch_dir)
        rows = read_jsonl(source_dir / "augment.jsonl")
        augment_stats[branch] = write_stats(branch_dir / "stats.json", rows, branch)
        (branch_dir / "archive_manifest.json").write_text(
            json.dumps({
                "source_directory": str(source_dir),
                "copied_files": copied,
                "data_file": "augment.jsonl",
                "data_sha256": sha256(branch_dir / "augment.jsonl"),
                "stats_file": "stats.json",
            }, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    official_dir = out_dir / "official_part1_part2"
    official_dir.mkdir(parents=True, exist_ok=True)
    part1, part2 = load_official_part12()
    write_jsonl(official_dir / "part1.jsonl", part1)
    write_jsonl(official_dir / "part2.jsonl", part2)
    write_jsonl(official_dir / "combined_part1_part2.jsonl", part1 + part2)
    part1_stats = write_stats(official_dir / "part1_stats.json", part1, "official_part1")
    part2_stats = write_stats(official_dir / "part2_stats.json", part2, "official_part2")
    official_stats = write_stats(
        official_dir / "combined_stats.json", part1 + part2, "official_part1_part2"
    )
    (official_dir / "source_manifest.json").write_text(
        json.dumps({
            "part1_notes": str(PART1_ROOT / "notes"),
            "part1_labels": str(PART1_ROOT / "labels"),
            "part2_notes": str(PART2_ROOT / "notes"),
            "part2_labels": str(PART2_ROOT / "labels"),
            "sanitization": "training.build_final_n_datasets.sanitize",
            "records": {"part1": len(part1), "part2": len(part2)},
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    common_rows = part1 + part2 + llm_rows
    route_stats = {
        "common_part1_part2_plus_llm": record_stats(common_rows, "common"),
        "private_route_part1_part2_plus_llm_plus_private_augment": record_stats(
            common_rows + read_jsonl(AUGMENT_PRIVATE / "augment.jsonl"), "private_route"
        ),
        "independent_route_part1_part2_plus_llm_plus_independent_augment": record_stats(
            common_rows + read_jsonl(AUGMENT_INDEPENDENT / "augment.jsonl"), "independent_route"
        ),
    }
    aggregate = {
        "note": "private_v2 và independent_v2 là hai route augment thay thế, không cộng chung.",
        "common_records": len(common_rows),
        "generated_llm": llm_stats,
        "augment": augment_stats,
        "official_part1": part1_stats,
        "official_part2": part2_stats,
        "official_part1_part2": official_stats,
        "routes": route_stats,
    }
    (out_dir / "aggregate_stats.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "archive_id": out_dir.name,
        "storage_only": True,
        "contains_train_validation_test_splits": False,
        "generated_llm_records": len(llm_rows),
        "augment_branches": {branch: 9200 for branch in augment_sources},
        "official_part1_records": len(part1),
        "official_part2_records": len(part2),
        "common_records": len(common_rows),
        "route_records": {
            "private": len(common_rows) + 9200,
            "independent": len(common_rows) + 9200,
        },
        "files": {
            "generated_llm": {
                "data": "generated_llm/data.jsonl",
                "sha256": sha256(generated_dir / "data.jsonl"),
            },
            "private_augment": {
                "data": "deterministic_augment/private_v2/augment.jsonl",
                "sha256": sha256(augment_dir / "private_v2/augment.jsonl"),
            },
            "independent_augment": {
                "data": "deterministic_augment/independent_v2/augment.jsonl",
                "sha256": sha256(augment_dir / "independent_v2/augment.jsonl"),
            },
            "part1": {
                "data": "official_part1_part2/part1.jsonl",
                "sha256": sha256(official_dir / "part1.jsonl"),
            },
            "part2": {
                "data": "official_part1_part2/part2.jsonl",
                "sha256": sha256(official_dir / "part2.jsonl"),
            },
        },
        "aggregate_stats": "aggregate_stats.json",
    }
    (out_dir / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    readme = (
        f"# {out_dir.name}\n\n"
        "Archive storage-only, không chia train/validation/test.\n\n"
        "## Ba thư mục\n\n"
        "- `generated_llm/`: 780 LLM context accepted và audit trail.\n"
        "- `deterministic_augment/`: hai nhánh augment v2 thay thế nhau: private và independent.\n"
        "- `official_part1_part2/`: bản sanitize chính xác của Part 1/Part 2 dùng khi build N.\n\n"
        "## Thống kê\n\n"
        "Xem `aggregate_stats.json`; stats riêng nằm trong từng thư mục.\n"
        "Hai nhánh augment không được cộng chung: mỗi route có 200 official + 780 LLM + 9.200 augment = 10.180 record lưu trữ logic.\n\n"
        "Không có model checkpoint, prediction, candidate hoặc split train/validation/test trong archive này.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_archive(args.out.resolve(), overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
