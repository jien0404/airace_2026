from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from .schema_v2 import ASSERTIONS, ASSERTION_TYPES, TYPES

from .windowing_v2 import entity_centered_window, entity_word_span, sliding_windows

LABELS = ["O"] + [f"{prefix}-{typ}" for typ in TYPES for prefix in ("B", "I")]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TYPE_TO_ID = {typ: index + 1 for index, typ in enumerate(TYPES)}  # 0=None cho span head


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def record_to_windows(
    record: dict,
    max_words: int,
    overlap_words: int,
    header_dropout: float,
    o_keep: float,
    seed: int,
    tokenizer=None,
    max_len: int = 256,
    prefix_subword_cap: int = 64,
    subword_cache: dict[str, int] | None = None,
) -> tuple[list[dict], list[str]]:
    rng = random.Random(f"{seed}:{record['id']}")
    include_header = rng.random() >= header_dropout
    output, errors = [], []
    covered = set()
    normal_windows = sliding_windows(
        record["text"], max_words, overlap_words, include_header,
        tokenizer=tokenizer, max_len=max_len, prefix_subword_cap=prefix_subword_cap,
        subword_cache=subword_cache,
    )
    all_windows = list(normal_windows)
    if tokenizer is not None:
        for entity in record.get("entities") or []:
            if any(
                entity_word_span(window["content"], *entity["position"]) is not None
                for window in normal_windows
            ):
                continue
            rescue = entity_centered_window(
                record["text"], entity["position"], tokenizer,
                max_len=max_len, max_words=max_words,
                subword_cache=subword_cache,
            )
            if rescue is not None:
                all_windows.append(rescue)
    for window_index, window in enumerate(all_windows):
        prefix = window["prefix"]
        content = window["content"]
        tokens = prefix + content
        tags = ["O"] * len(tokens)
        token_loss_mask = [0] * len(prefix) + [
            1 if rng.random() < o_keep else 0 for _ in content
        ]
        spans = []
        for entity_index, entity in enumerate(record.get("entities") or []):
            span = entity_word_span(content, *entity["position"])
            if span is None:
                continue
            start, end = span
            covered.add(entity_index)
            start += len(prefix)
            end += len(prefix)
            typ = entity["type"]
            tags[start] = f"B-{typ}"
            for pos in range(start + 1, end):
                tags[pos] = f"I-{typ}"
            for pos in range(start, end):
                token_loss_mask[pos] = 1
            assertions = entity.get("assertions") or []
            assertion_supervision = entity.get("_assertion_supervision", True)
            spans.append({
                "start": start,
                "end": end,
                "type_id": TYPE_TO_ID[typ],
                "assertions": [int(name in assertions) for name in ASSERTIONS],
                "assertion_mask": int(assertion_supervision and typ in ASSERTION_TYPES),
            })
        if not spans and rng.random() > 0.5:
            continue
        output.append({
            "id": f"{record['id']}:w{window_index}",
            "record_id": record["id"],
            "tokens": [token for token, _, _ in tokens],
            "char_offsets": [[start, end] for _, start, end in tokens],
            "n_prefix": len(prefix),
            "ner_tags": tags,
            "token_loss_mask": token_loss_mask,
            "spans": spans,
            "genre": record.get("genre", "unknown"),
            "source": record.get("source", record.get("record_kind", "unknown")),
            "record_kind": record.get("record_kind", "unknown"),
            "sampling_group": record.get("_sampling_group", "default"),
            "sample_weight": 1.0,
            "subword_length": window.get("subword_length"),
            "coverage_rescue": bool(window.get("coverage_rescue")),
        })
    for index in range(len(record.get("entities") or [])):
        if index not in covered:
            errors.append(f"{record['id']}:entity[{index}]:not_covered")
    return output, errors


def build_track(
    dataset_dir: Path,
    out_dir: Path,
    max_words: int,
    overlap_words: int,
    header_dropout: float,
    o_keep: float,
    standalone_ratio: float,
    seed: int,
    gold_mass: float = 0.5,
    source_masses: dict[str, float] | None = None,
    tokenizer=None,
    max_len: int = 256,
    prefix_subword_cap: int = 64,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.txt").write_text("\n".join(LABELS), encoding="utf-8")
    (out_dir / "types.txt").write_text("\n".join(TYPES), encoding="utf-8")
    (out_dir / "assertions.txt").write_text("\n".join(ASSERTIONS), encoding="utf-8")
    report = {
        "schema_version": 2,
        "raw_dataset_root": str(dataset_dir.resolve()),
        "tokenizer_aware": tokenizer is not None,
        "windowing": {
            "max_words": max_words,
            "overlap_words": overlap_words,
            "max_len": max_len if tokenizer is not None else None,
            "prefix_subword_cap": prefix_subword_cap if tokenizer is not None else None,
        },
        "splits": {},
        "coverage_errors": [],
    }
    subword_cache: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        records = read_jsonl(dataset_dir / f"{split}.jsonl")
        windows = []
        for record in records:
            built, errors = record_to_windows(
                record, max_words, overlap_words, header_dropout, o_keep, seed,
                tokenizer, max_len, prefix_subword_cap, subword_cache,
            )
            windows.extend(built)
            report["coverage_errors"].extend(errors)
        if split == "train" and windows:
            if source_masses:
                unknown_groups = {
                    row["sampling_group"] for row in windows
                } - set(source_masses)
                if unknown_groups:
                    raise RuntimeError(
                        f"Thiếu source mass cho sampling group: {sorted(unknown_groups)}"
                    )
                if abs(sum(source_masses.values()) - 1.0) > 1e-9:
                    raise RuntimeError("Tổng source_masses phải bằng 1")
                for group, mass in source_masses.items():
                    indexes = [
                        index for index, row in enumerate(windows)
                        if row["sampling_group"] == group
                    ]
                    if not indexes and mass > 0:
                        raise RuntimeError(f"Sampling group {group!r} rỗng nhưng mass={mass}")
                    for index in indexes:
                        windows[index]["sample_weight"] = mass / len(indexes)
            else:
                segment_indexes = [
                    index for index, row in enumerate(windows)
                    if row["record_kind"] == "segment"
                ]
                document_indexes = [
                    index for index, row in enumerate(windows)
                    if row["record_kind"] == "document"
                ]
                synthetic_mass = 1.0 - gold_mass
                for indexes, mass in (
                    (segment_indexes, synthetic_mass * standalone_ratio),
                    (document_indexes, synthetic_mass * (1.0 - standalone_ratio)),
                ):
                    if indexes:
                        per_window = mass / len(indexes)
                        for index in indexes:
                            windows[index]["sample_weight"] = per_window
                direct_part3_indexes = [
                    index for index, row in enumerate(windows)
                    if row["record_kind"].startswith("gold") and row["source"] == "part3"
                ]
                independent_gold_indexes = [
                    index for index, row in enumerate(windows)
                    if row["record_kind"].startswith("gold") and row["source"] != "part3"
                ]
                # C: Part 3 đúng 25% sampling mass; independent gold 25%; synthetic 50%.
                # A/B không có direct Part 3 nên independent gold nhận trọn 50%.
                direct_mass = gold_mass / 2 if direct_part3_indexes else 0.0
                independent_mass = gold_mass - direct_mass
                for indexes, mass in (
                    (independent_gold_indexes, independent_mass),
                    (direct_part3_indexes, direct_mass),
                ):
                    if indexes:
                        for index in indexes:
                            windows[index]["sample_weight"] = mass / len(indexes)
            total = sum(row["sample_weight"] for row in windows)
            scale = len(windows) / max(total, 1e-9)
            for row in windows:
                row["sample_weight"] *= scale
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in windows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split] = {
            "records": len(records),
            "windows": len(windows),
            "entities": sum(len(window["spans"]) for window in windows),
            "max_subword_length": max(
                (row.get("subword_length") or 0 for row in windows), default=0
            ),
            "subword_overflow_windows": sum(
                (row.get("subword_length") or 0) > max_len for row in windows
            ) if tokenizer is not None else None,
            "coverage_rescue_windows": sum(
                bool(row.get("coverage_rescue")) for row in windows
            ),
        }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if (dataset_dir / "manifest.json").is_file():
        shutil.copy2(dataset_dir / "manifest.json", out_dir / "manifest.json")
    # Raw document-level validation is required to select checkpoints with the same merge and
    # overlap scorer as inference.  Keep it inside the portable window archive.
    for split in ("validation", "test"):
        source = dataset_dir / f"{split}.jsonl"
        if source.is_file():
            shutil.copy2(source, out_dir / f"raw_{split}.jsonl")
    return report


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset_factory output cho model v2")
    parser.add_argument("--dataset-root", default="datasets/v1")
    parser.add_argument("--out-root", default="training/dataset_v2")
    parser.add_argument("--tracks", default="A,B,C")
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=45)
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer HF/checkpoint; bật window theo ngân sách subword dùng chung với infer",
    )
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--prefix-subword-cap", type=int, default=64)
    parser.add_argument("--header-dropout", type=float, default=0.35)
    parser.add_argument("--o-keep", type=float, default=0.35)
    parser.add_argument("--standalone-ratio", type=float, default=0.60)
    parser.add_argument(
        "--gold-mass", type=float, default=0.5,
        help="Tỷ lệ khối lượng lấy mẫu dành cho gold thật; phần còn lại cho synthetic. "
             "Bỏ gt2 mà giữ 0.5 thì 249 cửa sổ part1 bị lặp ~14 lần mỗi epoch.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--source-mass", action="append", default=[], metavar="GROUP=FLOAT",
        help="Override sampler mass theo record._sampling_group; phải cấp đủ group và tổng=1",
    )
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    source_masses = None
    if args.source_mass:
        source_masses = {}
        for spec in args.source_mass:
            group, separator, value = spec.partition("=")
            if not separator or not group:
                parser.error(f"--source-mass không hợp lệ: {spec!r}")
            source_masses[group] = float(value)
    for track in [value.strip().upper() for value in args.tracks.split(",")]:
        report = build_track(
            Path(args.dataset_root) / f"track_{track.lower()}",
            Path(args.out_root) / f"track_{track.lower()}",
            args.max_words,
            args.overlap_words,
            args.header_dropout,
            args.o_keep,
            args.standalone_ratio,
            args.seed,
            args.gold_mass,
            source_masses,
            tokenizer,
            args.max_len,
            args.prefix_subword_cap,
        )
        print(
            f"[{track}] "
            + " ".join(f"{name}={stats['windows']}w" for name, stats in report["splits"].items())
            + f" coverage_errors={len(report['coverage_errors'])}"
        )


if __name__ == "__main__":
    main()
