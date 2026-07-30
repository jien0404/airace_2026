"""Gộp các nguồn thành corpus NER mức bản ghi cho thử nghiệm train đầu tiên.

Phân vai (Track A — Part 3 KHÔNG bao giờ vào train):

| split      | nguồn                                                    |
|------------|----------------------------------------------------------|
| train      | synthetic `dataset_factory` + gt2 (phần giữ lại) + part1 |
| validation | gt2 giữ riêng theo tài liệu — dữ liệu THẬT, không synthetic |
| test       | Part 3 (`input_turn2`) + nhãn artifact tốt nhất           |

Assertion: chỉ part1 được coi là trusted. gt2 bị mask theo `configs/dataset_v1.json`, còn
synthetic bị mask vì mẻ hiện tại chỉ có 0,5% entity mang assertion và có nhãn historical sai.
Entity bị mask mang `_assertion_supervision=false` nên `build_dataset_v2` không tính loss
assertion trên chúng — head assertion không bị dạy sai trong lúc ta đo NER.

Bản ghi ra: {id, text, entities[{text,position,type,assertions,candidates,_assertion_supervision}],
record_kind, source, genre}. `build_dataset_v2` đọc trực tiếp định dạng này.
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

from .schema_v2 import ASSERTIONS, ASSERTION_TYPES, TYPES

REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_entities(
    text: str,
    entities: Iterable[dict[str, Any]],
    *,
    mask_assertions: bool,
    stats: Counter,
) -> list[dict[str, Any]]:
    """Giữ lại entity biểu diễn được bằng BIO; verify offset trên RAW, không NFC."""
    kept: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in entities:
        start, end = entity["position"]
        surface = entity["text"]
        # Gold thật có ~1.170 span kèm khoảng trắng đầu/cuối. Ranh giới không được tính điểm,
        # nhưng span lệch token thì cửa sổ BIO không biểu diễn được và entity biến mất khỏi
        # nhãn train. Cắt khoảng trắng là thao tác an toàn duy nhất ở đây.
        trimmed = surface.strip()
        if trimmed and trimmed != surface:
            offset = surface.index(trimmed)
            start, end = start + offset, start + offset + len(trimmed)
            surface = trimmed
            stats["sửa:cắt khoảng trắng ở span"] += 1
        if entity["type"] not in TYPES:
            stats["bỏ:type lạ"] += 1
            continue
        if not (0 <= start < end <= len(text)) or text[start:end] != surface:
            stats["bỏ:offset lệch"] += 1
            continue
        if "\n" in surface:
            stats["bỏ:span vắt dòng"] += 1
            continue
        key = (start, end, entity["type"])
        if key in seen:
            stats["bỏ:trùng span"] += 1
            continue
        seen.add(key)
        assertions = [] if mask_assertions else [
            name for name in entity.get("assertions") or [] if name in ASSERTIONS
        ]
        if assertions and entity["type"] not in ASSERTION_TYPES:
            assertions = []
            stats["sửa:assertion sai type"] += 1
        kept.append({
            "text": surface,
            "position": [start, end],
            "type": entity["type"],
            "assertions": assertions,
            "candidates": [],
            "_assertion_supervision": not mask_assertions,
        })
    kept.sort(key=lambda row: (row["position"][0], row["position"][1]))
    # Overlap làm hỏng BIO: giữ span dài hơn, bỏ span chồng lên nó.
    accepted: list[dict[str, Any]] = []
    for row in kept:
        if accepted and row["position"][0] < accepted[-1]["position"][1]:
            previous = accepted[-1]
            if (row["position"][1] - row["position"][0]) > (
                previous["position"][1] - previous["position"][0]
            ):
                accepted[-1] = row
            stats["bỏ:span chồng lấn"] += 1
            continue
        accepted.append(row)
    return accepted


def load_labeled_dir(
    notes_dir: Path,
    labels_dir: Path,
    source: str,
    *,
    mask_assertions: bool,
    stats: Counter,
) -> list[dict[str, Any]]:
    records = []
    for note_path in sorted(notes_dir.glob("*.txt"), key=lambda p: (len(p.stem), p.stem)):
        label_path = labels_dir / f"{note_path.stem}.json"
        if not label_path.exists():
            continue
        text = note_path.read_text(encoding="utf-8")
        entities = json.loads(label_path.read_text(encoding="utf-8"))
        records.append({
            "id": f"{source}:{note_path.stem}",
            "text": text,
            "entities": clean_entities(
                text, entities, mask_assertions=mask_assertions, stats=stats
            ),
            "record_kind": "gold_document",
            "source": source,
            "genre": "clinical_document_unknown_layout",
        })
    return records


def load_part3(notes_dir: Path, labels_zip: Path, stats: Counter) -> list[dict[str, Any]]:
    records = []
    with zipfile.ZipFile(labels_zip) as archive:
        members = {Path(name).stem: name for name in archive.namelist() if name.endswith(".json")}
        for note_path in sorted(notes_dir.glob("*.txt"), key=lambda p: (len(p.stem), p.stem)):
            member = members.get(note_path.stem)
            if member is None:
                continue
            text = note_path.read_text(encoding="utf-8")
            entities = json.loads(archive.read(member))
            records.append({
                "id": f"part3:{note_path.stem}",
                "text": text,
                "entities": clean_entities(
                    text, entities, mask_assertions=False, stats=stats
                ),
                "record_kind": "gold_document",
                "source": "part3",
                "genre": "clinical_document_unknown_layout",
            })
    return records


def load_synthetic(pilot_dir: Path, stats: Counter) -> list[dict[str, Any]]:
    records = []
    drafts_path = pilot_dir / "drafts.jsonl"
    for line in drafts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        draft = json.loads(line)
        text = draft["text"]
        entities = [
            {
                "text": mention["text"],
                "position": mention["position"],
                "type": mention["type"],
                "assertions": mention.get("assertions") or [],
            }
            for mention in draft["mentions"]
            if mention.get("should_label")
        ]
        records.append({
            "id": draft["draft_id"],
            "text": text,
            # mask: mẻ synthetic hiện tại không đủ tin cậy về assertion
            "entities": clean_entities(text, entities, mask_assertions=True, stats=stats),
            "record_kind": "segment",
            "source": "synthetic_track_a",
            "genre": (draft.get("reference_case") or {}).get("genre", "unknown"),
            "source_case_id": draft.get("source_case_id"),
        })
    return records


def describe(records: list[dict[str, Any]]) -> dict[str, Any]:
    chars = sum(len(record["text"]) for record in records)
    entities = [entity for record in records for entity in record["entities"]]
    asserted = sum(1 for entity in entities if entity["assertions"])
    return {
        "records": len(records),
        "characters": chars,
        "entities": len(entities),
        "entities_per_1k_chars": round(1000 * len(entities) / max(chars, 1), 2),
        "assertion_rate": round(100 * asserted / max(len(entities), 1), 2),
        "types": {
            typ: round(
                100 * sum(1 for entity in entities if entity["type"] == typ)
                / max(len(entities), 1), 1
            )
            for typ in TYPES
        },
        "by_source": dict(Counter(record["source"] for record in records)),
    }


def build(
    pilot_dir: Path,
    out_dir: Path,
    validation_documents: int,
    seed: int,
) -> dict[str, Any]:
    stats: Counter = Counter()
    synthetic = load_synthetic(pilot_dir, stats)
    gt2 = load_labeled_dir(
        _repo("annotation/data/groundtruth_part2/notes"),
        _repo("annotation/data/groundtruth_part2/labels"),
        "gt2", mask_assertions=True, stats=stats,
    )
    part1 = load_labeled_dir(
        _repo("annotation/data/best_54.92/notes"),
        _repo("annotation/data/best_54.92/labels"),
        "part1", mask_assertions=False, stats=stats,
    )
    part3_notes = _repo("input_turn2")
    part3_labels = _repo(
        "business_rules/artifacts/current/"
        "v66ab_hyperbilirubinemia_plus_alopecia_winners.zip"
    )
    test = load_part3(part3_notes, part3_labels, stats)
    if not test:
        raise RuntimeError("Không dựng được test Part 3; kiểm tra input_turn2 và artifact")

    # Validation là tài liệu gt2 GIỮ RIÊNG: cùng phân phối thật với test, và không có
    # document nào vừa ở train vừa ở validation.
    rng = random.Random(seed)
    shuffled = sorted(gt2, key=lambda record: record["id"])
    rng.shuffle(shuffled)
    validation = shuffled[:validation_documents]
    gt2_train = shuffled[validation_documents:]

    splits = {
        "train": synthetic + gt2_train + part1,
        "validation": validation,
        "test": test,
    }
    train_ids = {record["id"] for record in splits["train"]}
    for name in ("validation", "test"):
        overlap = train_ids & {record["id"] for record in splits[name]}
        if overlap:
            raise RuntimeError(f"{name} rò vào train: {sorted(overlap)[:5]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "track": "A",
        "part3_used_for": "test_only",
        "seed": seed,
        "sources": {
            "synthetic": {
                "pilot": str(pilot_dir),
                "drafts_sha256": _sha256(pilot_dir / "drafts.jsonl"),
                "assertions": "masked_unreliable",
            },
            "gt2": {"path": "annotation/data/groundtruth_part2", "assertions": "masked"},
            "part1": {"path": "annotation/data/best_54.92", "assertions": "trusted"},
            "part3_test": {
                "notes": "input_turn2",
                "labels": str(part3_labels.relative_to(REPO_ROOT)),
                "labels_sha256": _sha256(part3_labels),
                "note": "nhãn bài nộp tốt nhất, KHÔNG phải gold do ban tổ chức công bố",
            },
        },
        "dropped_or_fixed": dict(stats),
        "splits": {name: describe(records) for name, records in splits.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", default="datasets/pilots/train_trial_3000_a_v1")
    parser.add_argument("--out", default="datasets/ner_v1/track_a")
    parser.add_argument("--validation-documents", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    manifest = build(
        _repo(args.pilot), _repo(args.out), args.validation_documents, args.seed
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
