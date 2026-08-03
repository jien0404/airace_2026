"""Build dataset độc lập để train checkpoint nộp trên private test.

Khác với :mod:`training.build_ner_corpus`, module này cố ý đưa public Part 3 vào
train/validation. Bối cảnh là Part 3 chỉ là public test và private test của BTC là
một bộ note khác cùng nghiệp vụ. Đây là một nhánh checkpoint riêng, không dùng để
đánh giá hiệu quả của các pilot synthetic targeted H/I/J.

Hai biến thể được dựng cùng lúc:

``official_p3split_v1``
    Part 3 chia 80/20 ở mức tài liệu; Part 1 + Part 2 vào train. Đây là biến thể
    khuyến nghị để chọn checkpoint vì validation bám đúng phân phối public/private.

``official_p3split_part1valid_v1``
    Giống trên, nhưng giữ thêm 15 tài liệu Part 1 ở validation. Đây chỉ là ablation:
    Part 1 có overlap với Part 3 nên validation hỗn hợp có thể lạc quan hơn.

Không có ``test`` cục bộ: toàn bộ 100 tài liệu Part 3 đều được phân bổ vào train hoặc
validation. Private test thật không có nhãn ở local.

Ví dụ:

    python -m training.build_private_checkpoint_dataset

Kết quả gồm record-level corpus, window-level corpus và ZIP sẵn để upload máy GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from business_rules.artifacts import current_labels_zip, manifest as artifact_manifest

from .build_dataset_v2 import build_track
from .build_ner_corpus import REPO_ROOT, load_labeled_dir, load_part3, describe


DEFAULT_RECORD_ROOT = REPO_ROOT / "datasets/private_checkpoint_official_20260803"
DEFAULT_WINDOW_ROOT = REPO_ROOT / "datasets/ner_private_checkpoint_official_20260803"
DEFAULT_ZIP_ROOT = REPO_ROOT / "datasets"

PART1_ROOT = REPO_ROOT / "annotation/data/label_final/part1"
PART2_ROOT = REPO_ROOT / "annotation/data/label_final/gt2"
PART3_NOTES = REPO_ROOT / "input_turn2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Hash file names and bytes, stable across directory traversal order."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _split_records(
    records: list[dict[str, Any]], train_ratio: float, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio phải nằm giữa 0 và 1, nhận {train_ratio}")
    ordered = sorted(records, key=lambda record: record["id"])
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    train_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * train_ratio)))
    return (
        sorted(shuffled[:train_count], key=lambda record: record["id"]),
        sorted(shuffled[train_count:], key=lambda record: record["id"]),
    )


def _validate_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Recheck the invariants that matter before exporting a training corpus."""
    stats = Counter()
    seen_ids: set[str] = set()
    allowed_types = {
        "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM", "THUỐC",
    }
    for record in records:
        record_id = record["id"]
        if record_id in seen_ids:
            stats["duplicate_record_id"] += 1
        seen_ids.add(record_id)
        intervals: list[tuple[int, int]] = []
        for entity in record.get("entities") or []:
            position = entity.get("position")
            if not (isinstance(position, list) and len(position) == 2):
                stats["invalid_position"] += 1
                continue
            start, end = position
            if not (isinstance(start, int) and isinstance(end, int)):
                stats["invalid_position"] += 1
                continue
            if not (0 <= start < end <= len(record["text"])):
                stats["offset_out_of_range"] += 1
            elif record["text"][start:end] != entity.get("text"):
                stats["offset_text_mismatch"] += 1
            if entity.get("type") not in allowed_types:
                stats["unknown_type"] += 1
            if "\n" in str(entity.get("text", "")):
                stats["newline_span"] += 1
            intervals.append((start, end))
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if right[0] < left[1]:
                stats["overlap"] += 1
    return dict(stats)


def _zip_track(track_dir: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Đã tồn tại {output}; dùng --overwrite nếu muốn dựng lại artifact này"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Giữ track_a ở cấp đầu ZIP để lệnh `unzip -d datasets/...` dùng trực tiếp.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in track_dir.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(track_dir.parent).as_posix())


def _source_manifest(
    part1: list[dict[str, Any]],
    part2: list[dict[str, Any]],
    part3: list[dict[str, Any]],
    source_stats: dict[str, dict[str, int]],
    labels_zip: Path,
) -> dict[str, Any]:
    artifact = artifact_manifest()
    return {
        "part1": {
            "notes": _relative(PART1_ROOT / "notes"),
            "labels": _relative(PART1_ROOT / "labels"),
            "root_manifest": _relative(PART1_ROOT.parent / "manifest.json"),
            "root_manifest_sha256": _sha256(PART1_ROOT.parent / "manifest.json"),
            "tree_sha256": _tree_sha256(PART1_ROOT),
            "records": len(part1),
            "description": "Part 1 nhãn chính thống bản label_final hậu kiểm",
            "cleaning_audit": source_stats["part1"],
        },
        "part2": {
            "notes": _relative(PART2_ROOT / "notes"),
            "labels": _relative(PART2_ROOT / "labels"),
            "root_manifest": _relative(PART2_ROOT.parent / "manifest.json"),
            "root_manifest_sha256": _sha256(PART2_ROOT.parent / "manifest.json"),
            "tree_sha256": _tree_sha256(PART2_ROOT),
            "records": len(part2),
            "description": "Part 2 nhãn chính thống bản label_final hậu kiểm assertion/span",
            "cleaning_audit": source_stats["part2"],
        },
        "part3": {
            "notes": _relative(PART3_NOTES),
            "labels_zip": _relative(labels_zip),
            "labels_sha256": _sha256(labels_zip),
            "artifact_manifest": _relative(REPO_ROOT / "business_rules/artifacts/MANIFEST.json"),
            "artifact_manifest_sha256": _sha256(
                REPO_ROOT / "business_rules/artifacts/MANIFEST.json"
            ),
            "artifact_name": artifact["current"]["name"],
            "artifact_manifest_entry": artifact["current"],
            "records": len(part3),
            "description": (
                "Public Part 3 hiện hành dùng làm pseudo-gold local; artifact là bản nhãn tốt "
                "nhất/probed của dự án, không phải gold file BTC phát hành riêng"
            ),
            "cleaning_audit": source_stats["part3"],
        },
    }


def build_variant(
    *,
    name: str,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    record_root: Path,
    window_root: Path,
    zip_root: Path,
    seed: int,
    split_policy: dict[str, Any],
    max_words: int,
    overlap_words: int,
    header_dropout: float,
    o_keep: float,
    standalone_ratio: float,
    overwrite: bool,
    skip_windows: bool,
) -> dict[str, Any]:
    variant_root = record_root / name
    track_dir = variant_root / "track_a"
    if variant_root.exists() and not overwrite:
        raise FileExistsError(
            f"Đã tồn tại {variant_root}; dùng --overwrite nếu muốn dựng lại artifact này"
        )
    track_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train, "validation": validation, "test": []}
    all_ids: set[str] = set()
    for split_name, records in splits.items():
        ids = {record["id"] for record in records}
        if all_ids & ids:
            raise RuntimeError(f"Record rò giữa split {name}/{split_name}")
        all_ids |= ids
        _write_jsonl(track_dir / f"{split_name}.jsonl", records)

    record_audit = _validate_records(record for rows in splits.values() for record in rows)
    if record_audit:
        raise RuntimeError(f"Corpus {name} có lỗi sau clean: {record_audit}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_name": name,
        "purpose": "private_checkpoint",
        "independent_branch": True,
        "independence_note": (
            "Nhánh này phục vụ checkpoint nộp private test. Nó độc lập với luồng tối ưu "
            "synthetic targeted và không dùng các pilot H/I/J, education FP-fix hay historical."
        ),
        "track": "A",
        "seed": seed,
        "part3_used_for": "train_and_validation",
        "private_test_note": (
            "Part 3 là public test; private test thật không có ở local. Không có test split "
            "cục bộ để tránh giả vờ đo unbiased private performance."
        ),
        "source_policy": {
            "official_documents_only": True,
            "part3_labels_are_local_pseudo_gold": True,
            "synthetic_included": False,
            "targeted_pilots_included": [],
            "candidates_training": False,
        },
        "sources": source_manifest,
        "split_policy": split_policy,
        "record_audit": record_audit,
        "record_root": _relative(variant_root),
        "record_ids": {
            split: [record["id"] for record in records]
            for split, records in splits.items()
        },
        "splits": {split: describe(records) for split, records in splits.items()},
    }

    window_report: dict[str, Any] | None = None
    zip_path: Path | None = None
    if not skip_windows:
        window_track_dir = window_root / name / "track_a"
        window_report = build_track(
            track_dir,
            window_track_dir,
            max_words,
            overlap_words,
            header_dropout,
            o_keep,
            standalone_ratio,
            seed,
            gold_mass=1.0,
        )
        if window_report["coverage_errors"]:
            # Một số RAW Part 3 có lỗi dính hai từ quanh span, ví dụ
            # `thuốcVastarel`, `dạithường`, `5tăng`. Word-level windowing không thể
            # biểu diễn một entity là substring của một token; tăng overlap không
            # giải quyết được. Không âm thầm xoá entity/record, nhưng ghi cảnh báo
            # đầy đủ vào build_report + manifest để tránh coi đây là coverage hoàn hảo.
            print(
                f"[warning] {name}: {len(window_report['coverage_errors'])} "
                "entity không biểu diễn được ở word-level windowing; xem manifest"
            )
            manifest["window_quality_warning"] = {
                "kind": "unrepresentable_entity_inside_raw_token",
                "count": len(window_report["coverage_errors"]),
                "examples": window_report["coverage_errors"][:20],
                "action": "giữ RAW và nhãn trong record corpus; không giả vờ đã cover",
            }
        zip_path = zip_root / f"ner_private_checkpoint_{name}_windows_20260803.zip"
        _zip_track(window_track_dir, zip_path, overwrite)
        manifest["window_root"] = _relative(window_root / name)
        manifest["window_build"] = window_report
        manifest["zip"] = {
            "path": _relative(zip_path),
            "sha256": _sha256(zip_path),
        }

    (variant_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (variant_root / "README.md").write_text(
        "# " + name + "\n\n"
        "Dataset checkpoint độc lập cho private test. Xem `manifest.json` để biết "
        "provenance, split và cleaning audit.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--window-root", type=Path, default=DEFAULT_WINDOW_ROOT)
    parser.add_argument("--zip-root", type=Path, default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--part3-train-ratio", type=float, default=0.80)
    parser.add_argument(
        "--part1-validation-documents", type=int, default=15,
        help="Số Part 1 giữ ở validation của biến thể ablation; 0 để bỏ biến thể này",
    )
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=45)
    parser.add_argument("--header-dropout", type=float, default=0.35)
    parser.add_argument("--o-keep", type=float, default=0.35)
    parser.add_argument("--standalone-ratio", type=float, default=0.60)
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (PART1_ROOT, PART2_ROOT, PART3_NOTES):
        if not path.exists():
            raise SystemExit(f"Thiếu nguồn bắt buộc: {path}")
    labels_zip = current_labels_zip()

    source_stats: dict[str, Counter] = {
        "part1": Counter(), "part2": Counter(), "part3": Counter(),
    }
    part1 = load_labeled_dir(
        PART1_ROOT / "notes", PART1_ROOT / "labels", "part1",
        mask_assertions=False, stats=source_stats["part1"],
    )
    part2 = load_labeled_dir(
        PART2_ROOT / "notes", PART2_ROOT / "labels", "part2",
        mask_assertions=False, stats=source_stats["part2"],
    )
    part3 = load_part3(PART3_NOTES, labels_zip, source_stats["part3"])
    if len(part1) != 100 or len(part2) != 100 or len(part3) != 100:
        raise SystemExit(
            f"Số tài liệu không đúng kỳ vọng: part1={len(part1)} "
            f"part2={len(part2)} part3={len(part3)}"
        )

    source_audit = {
        "part1": dict(source_stats["part1"]),
        "part2": dict(source_stats["part2"]),
        "part3": dict(source_stats["part3"]),
    }
    source_manifest = _source_manifest(part1, part2, part3, source_audit, labels_zip)

    if args.part1_validation_documents < 0 or args.part1_validation_documents >= len(part1):
        raise SystemExit("--part1-validation-documents phải từ 0 đến 99")
    part3_train, part3_validation = _split_records(
        part3, args.part3_train_ratio, f"{args.seed}:part3",
    )
    part1_train, part1_validation = _split_records(
        part1, 1.0 - args.part1_validation_documents / len(part1),
        f"{args.seed}:part1",
    ) if args.part1_validation_documents else (part1, [])
    if args.part1_validation_documents:
        # `_split_records` dùng ratio; kiểm lại đúng số tài liệu user yêu cầu.
        if len(part1_validation) != args.part1_validation_documents:
            raise RuntimeError("Không tạo đúng số Part 1 validation yêu cầu")

    base_train = part3_train + part1 + part2
    base_validation = part3_validation
    manifests = []
    manifests.append(build_variant(
        name="official_p3split_v1",
        train=base_train,
        validation=base_validation,
        source_manifest=source_manifest,
        record_root=args.record_root,
        window_root=args.window_root,
        zip_root=args.zip_root,
        seed=args.seed,
        split_policy={
            "part3": {"train_documents": len(part3_train), "validation_documents": len(part3_validation)},
            "part1": {"train_documents": len(part1), "validation_documents": 0},
            "part2": {"train_documents": len(part2), "validation_documents": 0},
            "rationale": "Validation thuần Part 3 là proxy gần nhất cho private cùng nghiệp vụ.",
        },
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        header_dropout=args.header_dropout,
        o_keep=args.o_keep,
        standalone_ratio=args.standalone_ratio,
        overwrite=args.overwrite,
        skip_windows=args.skip_windows,
    ))
    if args.part1_validation_documents:
        manifests.append(build_variant(
            name="official_p3split_part1valid_v1",
            train=part3_train + part1_train + part2,
            validation=part3_validation + part1_validation,
            source_manifest=source_manifest,
            record_root=args.record_root,
            window_root=args.window_root,
            zip_root=args.zip_root,
            seed=args.seed,
            split_policy={
                "part3": {"train_documents": len(part3_train), "validation_documents": len(part3_validation)},
                "part1": {"train_documents": len(part1_train), "validation_documents": len(part1_validation)},
                "part2": {"train_documents": len(part2), "validation_documents": 0},
                "rationale": (
                    "Ablation thêm Part 1 validation; Part 1 có overlap với Part 3 nên "
                    "không dùng làm validation duy nhất cho quyết định private."
                ),
            },
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            header_dropout=args.header_dropout,
            o_keep=args.o_keep,
            standalone_ratio=args.standalone_ratio,
            overwrite=args.overwrite,
            skip_windows=args.skip_windows,
        ))

    print(json.dumps({"variants": manifests}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
