"""Gộp các nguồn thành corpus NER mức bản ghi cho thử nghiệm train đầu tiên.

Phân vai (Track A — Part 3 KHÔNG bao giờ vào train):

| split      | nguồn                                                    |
|------------|----------------------------------------------------------|
| train      | synthetic `dataset_factory` + gt2 (phần giữ lại) + part1 |
| validation | gt2 giữ riêng (NER) + lát synthetic giữ riêng (đo được assertion) |
| test       | Part 3 (`input_turn2`) + nhãn artifact tốt nhất           |

Assertion: part1 và synthetic được coi là trusted, gt2 bị mask.

- synthetic: mẻ v1 bị mask vì chỉ có 0,5% entity mang assertion. Mẻ v2 sinh theo spec mới đạt
  19,5% (gold 16,6%), assertion phải có cue đỡ theo hai chế độ support/conflict và được auto-repair,
  nên **đã mở supervision**. Không mở thì cả tập train chỉ còn part1 (2.726 entity) dạy assertion
  và head assertion không học được gì — đo được `assertion_f1 = 0.0` ở lần train đầu.
- gt2: `annotation/label_audit` đếm được 3.036 nghi ngờ trên 15.444 entity, trong đó 330 ca
  KẾT_QUẢ_XÉT_NGHIỆM mang assertion là vi phạm luật tuyệt đối. Giữ mask cho tới khi bản sửa được
  người duyệt.

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
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from business_rules.artifacts import current_labels_zip

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


GIANT_RUN = re.compile(r"\S{100,}")
GIANT_RUN_KEEP = 30

LEADING_NEGATOR = re.compile(r"^(không|chưa|phủ nhận|ko)\s+", re.IGNORECASE)

# Quy ước đo được trên CẢ BỐN nguồn: từ phủ định nằm NGOÀI span, entity mang `isNegated`.
# 782 ca gold theo quy ước này (gt2 663, part1 72, part3 47) so với 2 ca ngược lại.
#
# Nhưng KHÔNG được cắt bừa: `không thể tự đứng dậy`, `không nhấc chân phải khỏi mặt giường`,
# `chưa phát hiện bất thường` là gold part3 nguyên văn — ở đó phủ định CHÍNH LÀ phát hiện, cắt
# ra thì span còn lại vô nghĩa. Phân biệt bằng bằng chứng chứ không bằng cảm tính: chỉ cắt khi
# phần lõi tự nó đã là một entity đứng riêng trong gold ≥5 lần. Tập dưới đây là kết quả của
# phép đo đó trên ner_v4 (8 surface ứng viên, đã loại `tự chủ đại tiện` vì gold part3 gán
# `tiểu tiện không tự chủ` NGUYÊN CỤM — cắt nó là lật ngược nghĩa).
#
# Đo lại khi đổi corpus:
#   python -m training.negation_span_audit --data datasets/ner_v4/track_a
NEGATABLE_CORES = frozenset({"chướng", "ho", "ngủ", "nôn", "sốt", "đau", "đau đầu"})


def strip_leading_negator(
    surface: str, entity_type: str, assertions: list[str],
) -> tuple[str, list[str]] | None:
    """Cắt từ phủ định khỏi đầu span và gán `isNegated`, hoặc None nếu không đủ bằng chứng."""
    if entity_type not in ASSERTION_TYPES:
        return None
    match = LEADING_NEGATOR.match(surface)
    if not match:
        return None
    core = surface[match.end():]
    if core.strip().lower() not in NEGATABLE_CORES:
        return None
    return core, sorted(set(assertions) | {"isNegated"})


def collapse_giant_runs(
    text: str, entities: list[dict[str, Any]], stats: Counter,
) -> tuple[str, list[dict[str, Any]]]:
    """Rút gọn chuỗi không-khoảng-trắng dài bất thường, dời offset entity theo.

    gt2 có 8 chuỗi rác kiểu `nông농농농…` dài tới 18.019 ký tự. `windowing_v2` tách từ theo khoảng
    trắng nên cả chuỗi thành MỘT từ; tokenizer sinh 18.016 subword (cảnh báo `18016 > 512`), và
    `train_v2` gặp từ dài quá `max_len` là `break` ngay — mọi từ phía sau trong cửa sổ đó mất
    trắng. Không entity nào nằm trong các chuỗi này nên rút gọn là an toàn.
    """
    matches = [
        match for match in GIANT_RUN.finditer(text)
        if not any(
            not (entity["position"][1] <= match.start() or entity["position"][0] >= match.end())
            for entity in entities
        )
    ]
    if not matches:
        return text, entities
    pieces: list[str] = []
    shifts: list[tuple[int, int]] = []  # (vị trí gốc, tổng số ký tự đã bỏ tính tới đó)
    cursor = removed = 0
    for match in matches:
        pieces.append(text[cursor:match.start()])
        pieces.append(match.group()[:GIANT_RUN_KEEP])
        removed += len(match.group()) - GIANT_RUN_KEEP
        shifts.append((match.end(), removed))
        cursor = match.end()
        stats["sửa:rút gọn chuỗi rác dài"] += 1
    pieces.append(text[cursor:])
    new_text = "".join(pieces)

    def shift(position: int) -> int:
        delta = 0
        for boundary, total in shifts:
            if position >= boundary:
                delta = total
        return position - delta

    moved = []
    for entity in entities:
        start, end = entity["position"]
        moved.append({**entity, "position": [shift(start), shift(end)]})
    return new_text, moved


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
        forced: list[str] = []
        stripped = strip_leading_negator(
            surface, entity["type"], list(entity.get("assertions") or []),
        )
        if stripped is not None:
            core, forced = stripped
            start = end - len(core)
            surface = core
            stats["sửa:cắt từ phủ định khỏi span"] += 1
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
            name for name in (forced or entity.get("assertions") or []) if name in ASSERTIONS
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
        text, entities = collapse_giant_runs(text, entities, stats)
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


def load_synthetic(
    pilot_dir: Path, stats: Counter, mask_assertions: bool = False,
) -> list[dict[str, Any]]:
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
            "entities": clean_entities(
                text, entities, mask_assertions=mask_assertions, stats=stats
            ),
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
    pilot_dirs: list[Path],
    out_dir: Path,
    validation_documents: int,
    seed: int,
    mask_synthetic_assertions: bool = False,
    validation_synthetic_records: int = 200,
    exclude: tuple[str, ...] = (),
    use_fixed: bool = False,
    real_root: str | None = None,
    validation_sources: tuple[str, ...] = ("gt2", "synthetic"),
    validation_part1_documents: int = 15,
) -> dict[str, Any]:
    stats: Counter = Counter()
    # Nhiều pilot gộp lại: `id` của draft đã mang tiền tố mẻ nên không đụng nhau, và mỗi mẻ vào
    # train nguyên khối. Gộp KHÔNG thêm từ vựng — cả bốn mẻ dùng chung đúng ~2.100 surface do bộ
    # lọc `min_source_count=2` — nên đây là phép đo về KHỐI LƯỢNG và phong cách, không phải về
    # đa dạng từ vựng.
    synthetic = []
    seen_ids: set[str] = set()
    for pilot_dir in pilot_dirs:
        for record in load_synthetic(pilot_dir, stats, mask_synthetic_assertions):
            if record["id"] in seen_ids:
                record["id"] = f"{pilot_dir.name}:{record['id']}"
            seen_ids.add(record["id"])
            synthetic.append(record)
    # Bản đã rà bởi `annotation/label_audit`. Khi dùng bản này thì assertion của gt2 được MỞ:
    # lý do mask trước đây là 3.036 nghi ngờ trên 15.444 entity, mà chính chúng vừa được sửa.
    # Sau khi sửa, gt2 đạt assertion 16,8% (gold Part 3 16,6%) so với 10,7% của bản gốc.
    # `--real-root` cho phép trỏ vào bản nhãn thật khác (vd `label_final` đã hậu xử lý) mà không
    # phải ghi đè `label_fixed` — cần thiết để so hai bản nhãn trong cùng một lượt thử nghiệm.
    if real_root:
        gt2_root, part1_root = _repo(f"{real_root}/gt2"), _repo(f"{real_root}/part1")
    else:
        gt2_root = _repo("annotation/data/label_fixed/gt2" if use_fixed else "annotation/data/groundtruth_part2")
        part1_root = _repo("annotation/data/label_fixed/part1" if use_fixed else "annotation/data/best_54.92")
    gt2 = load_labeled_dir(
        gt2_root / "notes", gt2_root / "labels",
        "gt2", mask_assertions=not use_fixed, stats=stats,
    )
    part1 = load_labeled_dir(
        part1_root / "notes", part1_root / "labels",
        "part1", mask_assertions=False, stats=stats,
    )
    part3_notes = _repo("input_turn2")
    part3_labels = current_labels_zip()
    test = load_part3(part3_notes, part3_labels, stats)
    if not test:
        raise RuntimeError("Không dựng được test Part 3; kiểm tra input_turn2 và artifact")

    # THÀNH PHẦN VALIDATION LÀ MỘT BIẾN, không phải hằng số.
    #
    # Mỗi lựa chọn có một khuyết điểm đã biết, và không có phương án nào đúng hiển nhiên:
    #
    # - chỉ `gt2`: cùng phân phối văn bản thật với test, nhưng khi gt2 bị mask assertion thì
    #   `assertion_f1` luôn bằng 0 nên `selection_score` mù với một phần ba bài toán;
    # - chỉ `synthetic`: đo được assertion, nhưng chọn checkpoint theo chính phân phối mình sinh
    #   ra — model hợp với dữ liệu giả nhất chưa chắc hợp với Part 3 nhất;
    # - có `part1`: nhãn thật, assertion trusted, nhưng `part1-part3-text-overlap` đo được 40,5%
    #   văn bản Part 1 trùng Part 3, nên điểm validation sẽ lạc quan giả.
    #
    # Part 3 KHÔNG bao giờ được làm validation (`DATASET_CONTRACT` §7).
    rng = random.Random(seed)

    def hold_out(records: list[dict[str, Any]], count: int) -> tuple[list, list]:
        ordered = sorted(records, key=lambda record: record["id"])
        rng.shuffle(ordered)
        return ordered[:count], ordered[count:]

    unknown = set(validation_sources) - {"gt2", "synthetic", "part1"}
    if unknown:
        raise RuntimeError(f"--validation-sources không hợp lệ: {sorted(unknown)}")
    if not validation_sources:
        raise RuntimeError("validation không được rỗng")

    validation: list[dict[str, Any]] = []
    if "gt2" in validation_sources:
        held, gt2_train = hold_out(gt2, validation_documents)
        validation += held
    else:
        gt2_train = gt2
    if "synthetic" in validation_sources:
        held, synthetic_train = hold_out(synthetic, validation_synthetic_records)
        validation += held
    else:
        synthetic_train = synthetic
    if "part1" in validation_sources:
        held, part1_train = hold_out(part1, validation_part1_documents)
        validation += held
    else:
        part1_train = part1

    # Ablation nguồn: gt2 chiếm 53,6% khối lượng lấy mẫu nhưng phân bố type lệch hẳn so với
    # Part 3 (TRIỆU_CHỨNG 67% vs 39%), nên cần đo được phương án bỏ/giữ nó.
    train_by_source = {
        "synthetic": synthetic_train, "gt2": gt2_train, "part1": part1_train,
    }
    for name in exclude:
        if name not in train_by_source:
            raise RuntimeError(f"--exclude {name!r} không phải nguồn train hợp lệ")
        train_by_source[name] = []

    splits = {
        "train": train_by_source["synthetic"] + train_by_source["gt2"] + train_by_source["part1"],
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
        # Kế thừa cờ nhiễm từ pilot: nếu mẻ synthetic sinh ở Track C thì corpus cũng nhiễm.
        "contaminated": any(
            (json.loads((p / "pilot_manifest.json").read_text(encoding="utf-8"))
             .get("contaminated") if (p / "pilot_manifest.json").exists() else False)
            for p in pilot_dirs
        ),
        "seed": seed,
        "sources": {
            "synthetic": {
                "pilot": [str(p) for p in pilot_dirs],
                "drafts_sha256": [_sha256(p / "drafts.jsonl") for p in pilot_dirs],
                "assertions": "masked" if mask_synthetic_assertions else "trusted",
            },
            "gt2": {
                "path": str(gt2_root.relative_to(REPO_ROOT)),
                "assertions": "masked" if not use_fixed else "trusted",
            },
            "part1": {"path": str(part1_root.relative_to(REPO_ROOT)), "assertions": "trusted"},
            "part3_test": {
                "notes": "input_turn2",
                "labels": str(part3_labels.relative_to(REPO_ROOT)),
                "labels_sha256": _sha256(part3_labels),
                "note": "nhãn bài nộp tốt nhất, KHÔNG phải gold do ban tổ chức công bố",
            },
        },
        "excluded_from_train": list(exclude),
        "labels": "label_fixed (đã rà)" if use_fixed else "gốc",
        "dropped_or_fixed": dict(stats),
        "splits": {name: describe(records) for name, records in splits.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pilot", default="datasets/pilots/train_v2_5000_kept",
        help="Một hoặc nhiều thư mục pilot, ngăn cách bởi dấu phẩy",
    )
    parser.add_argument("--out", default="datasets/ner_v2/track_a")
    parser.add_argument("--validation-documents", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--use-fixed", action="store_true",
        help="Dùng nhãn đã rà ở annotation/data/label_fixed và MỞ assertion của gt2",
    )
    parser.add_argument(
        "--real-root",
        help="Thư mục nhãn THẬT thay cho label_fixed, vd annotation/data/label_final",
    )
    parser.add_argument(
        "--exclude", default="",
        help="Bỏ nguồn khỏi TRAIN, ngăn cách bởi dấu phẩy: synthetic,gt2,part1",
    )
    parser.add_argument(
        "--validation-synthetic", type=int, default=200,
        help="Số bản ghi synthetic giữ riêng cho validation để đo được assertion",
    )
    parser.add_argument(
        "--validation-sources", default="gt2,synthetic",
        help="Nguồn cho validation, ngăn cách bởi dấu phẩy: gt2,synthetic,part1",
    )
    parser.add_argument(
        "--validation-part1", type=int, default=15,
        help="Số tài liệu part1 giữ riêng khi part1 nằm trong --validation-sources",
    )
    parser.add_argument(
        "--mask-synthetic-assertions", action="store_true",
        help="Tắt supervision assertion của synthetic (mặc định BẬT từ mẻ v2)",
    )
    args = parser.parse_args()
    manifest = build(
        [_repo(x.strip()) for x in args.pilot.split(",") if x.strip()], _repo(args.out), args.validation_documents, args.seed,
        args.mask_synthetic_assertions, args.validation_synthetic,
        tuple(name for name in args.exclude.split(',') if name.strip()),
        args.use_fixed,
        args.real_root,
        tuple(name.strip() for name in args.validation_sources.split(',') if name.strip()),
        args.validation_part1,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
