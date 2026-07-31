"""Báo cáo occurrence — thước đo cho occurrence gate (DATASET_CONTRACT §3b, CANONICAL §1.2).

Với mỗi (file, surface) đếm: số occurrence xuất hiện trong RAW, số được gán, số bị bỏ, số bị một
entity dài hơn phủ. Từ đó suy ra tỷ lệ nhóm "lấy tất cả" vs nhóm "mixed" — chỉ số quyết định
xem dataset có dạy được gate context hay chỉ dạy nhận diện surface.

Occurrence được đếm theo **ranh giới từ**, và occurrence nằm trong một entity dài hơn được tách
riêng thành `covered` chứ không tính là "bị bỏ", vì quy ước cấm gán entity ngắn lồng trong entity
dài đã cover nó.

Dùng:

    python -m dataset_factory.occurrence_report --jsonl datasets/ner_v1/track_a/train.jsonl
    python -m dataset_factory.occurrence_report --labels-dir annotation/data/groundtruth_part2
    python -m dataset_factory.occurrence_report --pilot datasets/pilots/<tên>
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .casebook import is_word_bounded

MIN_SURFACE_LEN = 3
NEAR_REPEAT_CHARS = 100
SENTENCE_SPLIT = re.compile(r"[.!?…]\s|\n")


def all_occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    cursor = 0
    while surface:
        start = text.find(surface, cursor)
        if start < 0:
            break
        end = start + len(surface)
        if is_word_bounded(text, start, end):
            positions.append((start, end))
        cursor = start + max(1, len(surface))
    return positions


def _sentence_index(text: str) -> list[int]:
    """Trả về chỉ số câu cho từng ký tự, để xét occurrence lặp trong CÙNG câu."""
    index = [0] * (len(text) + 1)
    current = 0
    position = 0
    for match in SENTENCE_SPLIT.finditer(text):
        for offset in range(position, match.end()):
            index[offset] = current
        position = match.end()
        current += 1
    for offset in range(position, len(text) + 1):
        index[offset] = current
    return index


def analyse_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = record["text"]
    entities = record.get("entities") or []
    spans = {(entity["position"][0], entity["position"][1]) for entity in entities}
    labeled_starts: dict[str, set[int]] = defaultdict(set)
    type_by_surface: dict[str, str] = {}
    for entity in entities:
        labeled_starts[entity["text"]].add(entity["position"][0])
        type_by_surface.setdefault(entity["text"], entity["type"])
    sentences = _sentence_index(text)

    rows = []
    for surface, starts in labeled_starts.items():
        if len(surface) < MIN_SURFACE_LEN:
            continue
        occurrences = all_occurrences(text, surface)
        covered = [
            (start, end) for start, end in occurrences
            if any(
                other_start <= start and end <= other_end and (other_start, other_end) != (start, end)
                for other_start, other_end in spans
            )
        ]
        covered_set = set(covered)
        candidates = [item for item in occurrences if item not in covered_set]
        selected = [item for item in candidates if item[0] in starts]
        omitted = [item for item in candidates if item[0] not in starts]
        near_selected = sum(
            1 for start, _ in omitted
            if any(
                0 < start - other < NEAR_REPEAT_CHARS
                and sentences[start] == sentences[other]
                for other, _ in selected
            )
        )
        near_taken = sum(
            1 for start, _ in selected
            if any(
                0 < start - other < NEAR_REPEAT_CHARS
                and sentences[start] == sentences[other]
                for other, _ in selected
            )
        )
        rows.append({
            "file": record.get("id", "?"),
            "surface": surface,
            "type": type_by_surface.get(surface, "?"),
            "occurrence_count": len(candidates),
            "selected_count": len(selected),
            "omitted_count": len(omitted),
            "covered_count": len(covered),
            "near_same_sentence_taken": near_taken,
            "near_same_sentence_omitted": near_selected,
            # Nhóm mixed mà occurrence được gán đúng là occurrence ĐẦU TIÊN: dấu hiệu dataset
            # đang dạy luật cũ "chỉ lấy lần đầu" thay vì đọc ngữ cảnh từng occurrence.
            "first_only": bool(omitted) and selected == candidates[:1],
        })
    return rows


def summarise(rows: Iterable[dict[str, Any]], records: int) -> dict[str, Any]:
    rows = [row for row in rows if row["occurrence_count"] >= 2]
    take_all = [row for row in rows if row["omitted_count"] == 0]
    mixed = [row for row in rows if row["omitted_count"] > 0]
    per_type: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per_type[row["type"]]["selected"] += row["selected_count"]
        per_type[row["type"]]["omitted"] += row["omitted_count"]
        per_type[row["type"]]["groups"] += 1
        if row["omitted_count"]:
            per_type[row["type"]]["mixed_groups"] += 1
    near_taken = sum(row["near_same_sentence_taken"] for row in rows)
    near_omitted = sum(row["near_same_sentence_omitted"] for row in rows)
    return {
        "records": records,
        "repeated_surface_groups": len(rows),
        "groups_per_record": round(len(rows) / max(records, 1), 2),
        "take_all_pct": round(100 * len(take_all) / max(len(rows), 1), 1),
        "mixed_pct": round(100 * len(mixed) / max(len(rows), 1), 1),
        "selected": sum(row["selected_count"] for row in rows),
        "omitted": sum(row["omitted_count"] for row in rows),
        "covered_occurrences": sum(row["covered_count"] for row in rows),
        # Tỷ lệ occurrence lặp TRONG CÙNG CÂU, cách occurrence đã chọn < 100 ký tự,
        # mà vẫn được gán. Spec Part 2 đo được 16,7%.
        "near_same_sentence_taken_pct": round(
            100 * near_taken / max(near_taken + near_omitted, 1), 1
        ),
        # Trong các nhóm mixed, bao nhiêu % chỉ gán occurrence đầu. Gold gt2 = 27,6%.
        "mixed_first_only_pct": round(
            100 * sum(1 for row in mixed if row["first_only"]) / max(len(mixed), 1), 1
        ),
        "per_type": {
            typ: {
                "groups": counts["groups"],
                "mixed_pct": round(100 * counts["mixed_groups"] / max(counts["groups"], 1), 1),
                "selected": counts["selected"],
                "omitted": counts["omitted"],
                "take_rate_pct": round(
                    100 * counts["selected"] / max(counts["selected"] + counts["omitted"], 1), 1
                ),
            }
            for typ, counts in sorted(per_type.items())
        },
    }


def warnings(rows: list[dict[str, Any]], summary: dict[str, Any], records: list[dict]) -> list[str]:
    flags = []
    if summary["repeated_surface_groups"] and summary["mixed_pct"] < 15:
        flags.append(
            f"mixed_pct={summary['mixed_pct']}% quá thấp — dataset gần như không dạy "
            "occurrence gate (tham chiếu: gold ~20%, train nên 35-40%)"
        )
    diversity = summary.get("surface_diversity") or {}
    if diversity.get("singleton_pct", 100) < 25:
        flags.append(
            f"singleton_pct={diversity['singleton_pct']}% (Part 3: 56%) — từ vựng lặp quá nhiều, "
            f"lặp trung bình {diversity.get('mean_reuse')}×; cân nhắc --surface-reuse-cap"
        )
    if summary["groups_per_record"] < 0.5:
        flags.append(
            f"chỉ {summary['groups_per_record']} nhóm surface lặp/bản ghi — hầu như không có "
            "cơ hội học gate"
        )
    empty = sum(1 for record in records if not (record.get("entities") or []))
    if not empty:
        flags.append("không có bản ghi nào rỗng entity; spec yêu cầu có note không chứa entity")
    if not summary["covered_occurrences"]:
        flags.append(
            "không có occurrence nào bị entity dài hơn phủ; thiếu mẫu 'entity dài cover surface ngắn'"
        )
    # Mục tiêu là 45-55% (vị trí vô nghĩa), không phải 27,6% của gold — chỉ cảnh báo khi vượt hẳn.
    if summary["mixed_first_only_pct"] > 60:
        flags.append(
            f"mixed_first_only_pct={summary['mixed_first_only_pct']}% — occurrence được gán hầu "
            "như luôn là lần xuất hiện ĐẦU TIÊN, tức dataset dạy luật cũ 'chỉ lấy lần đầu' chứ "
            "không dạy đọc ngữ cảnh (mục tiêu 45-55%, gold gt2 = 27,6%)"
        )
    return flags


def _records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _records_from_labels_dir(root: Path) -> list[dict[str, Any]]:
    notes, labels = root / "notes", root / "labels"
    records = []
    for note in sorted(notes.glob("*.txt")):
        label_path = labels / f"{note.stem}.json"
        if label_path.exists():
            records.append({
                "id": f"{root.name}:{note.stem}",
                "text": note.read_text(encoding="utf-8"),
                "entities": json.loads(label_path.read_text(encoding="utf-8")),
            })
    return records


def _records_from_zip(notes_dir: Path, archive_path: Path) -> list[dict[str, Any]]:
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {Path(name).stem: name for name in archive.namelist() if name.endswith(".json")}
        for note in sorted(notes_dir.glob("*.txt")):
            if note.stem in members:
                records.append({
                    "id": f"{archive_path.stem}:{note.stem}",
                    "text": note.read_text(encoding="utf-8"),
                    "entities": json.loads(archive.read(members[note.stem])),
                })
    return records


def _records_from_pilot(pilot_dir: Path) -> list[dict[str, Any]]:
    """Draft pilot: giữ thêm các occurrence CỐ Ý KHÔNG GÁN để báo cáo được gate."""
    records = []
    for line in (pilot_dir / "drafts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        draft = json.loads(line)
        records.append({
            "id": draft["draft_id"],
            "text": draft["text"],
            "entities": [
                {
                    "text": mention["text"],
                    "position": mention["position"],
                    "type": mention["type"],
                    "assertions": mention.get("assertions") or [],
                }
                for mention in draft["mentions"] if mention.get("should_label")
            ],
            "declared_roles": [
                (plan.get(mention["seed_id"]) or {}).get("occurrence_role")
                for mention in draft["mentions"]
                if not mention.get("should_label")
                for plan in [{
                    item["seed_id"]: item
                    for item in (draft.get("entity_contract") or {}).get("entity_plan", [])
                }]
            ],
        })
    return records


def report(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for record in records for row in analyse_record(record)]
    summary = summarise(rows, len(records))
    declared = Counter(
        role for record in records for role in record.get("declared_roles") or [] if role
    )
    if declared:
        # Draft pilot khai báo sẵn vai trò của occurrence không gán; đo trực tiếp thay vì
        # suy ngược từ nhãn (surface không gán ở đâu cả thì analyse_record không thấy).
        summary["declared_omitted_roles"] = dict(declared)
        summary["covered_occurrences"] += declared.get("covered_by_longer", 0)
    summary["surface_diversity"] = surface_diversity(records)
    return {"summary": summary, "warnings": warnings(rows, summary, records), "rows": rows}


def surface_diversity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Đa dạng từ vựng — đây là CẢNH BÁO, không phải quota.

    Mẻ v3: 2.111 surface cho 39.777 ca, lặp 18,8×, singleton 0,1%. Part 3 gold: 1.179 surface cho
    2.711 ca, singleton 56%. Đo được 323 ca gold (11,9%) có surface nằm sẵn trong kho mà mẻ v3
    chưa bao giờ bốc — đó là phần `--surface-reuse-cap` mở ra.

    `CANONICAL` §6: phân phối phải nảy ra từ luật gán và thể loại văn bản, không ép hậu kỳ. Nên
    các số này chỉ để soi, không được dùng để lọc/bù draft sau khi sinh.
    """
    counts: Counter = Counter()
    for record in records:
        for entity in record.get("entities") or []:
            counts[(entity["text"].strip().casefold(), entity["type"])] += 1
    total = sum(counts.values())
    if not counts:
        return {"distinct_surfaces": 0, "mean_reuse": 0.0, "singleton_pct": 0.0}
    return {
        "distinct_surfaces": len(counts),
        "mean_reuse": round(total / len(counts), 1),
        "singleton_pct": round(
            100 * sum(1 for value in counts.values() if value == 1) / len(counts), 1
        ),
        "part3_reference": {"distinct_surfaces": 1179, "mean_reuse": 2.3, "singleton_pct": 56.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jsonl", help="file jsonl {id,text,entities}")
    group.add_argument("--labels-dir", help="thư mục có notes/ và labels/")
    group.add_argument("--pilot", help="thư mục pilot có drafts.jsonl")
    group.add_argument("--zip-labels", help="zip nhãn, dùng cùng --notes")
    parser.add_argument("--notes", help="thư mục RAW đi kèm --zip-labels")
    parser.add_argument("--source", help="chỉ lấy bản ghi có source này (với --jsonl)")
    parser.add_argument("--rows-out", help="ghi báo cáo từng dòng ra CSV")
    args = parser.parse_args()

    if args.jsonl:
        records = _records_from_jsonl(Path(args.jsonl))
        if args.source:
            records = [row for row in records if row.get("source") == args.source]
    elif args.labels_dir:
        records = _records_from_labels_dir(Path(args.labels_dir))
    elif args.pilot:
        records = _records_from_pilot(Path(args.pilot))
    else:
        if not args.notes:
            raise SystemExit("--zip-labels cần đi kèm --notes")
        records = _records_from_zip(Path(args.notes), Path(args.zip_labels))

    result = report(records)
    if args.rows_out:
        header = [
            "surface", "type", "file", "occurrence_count",
            "selected_count", "omitted_count", "covered_count",
        ]
        lines = [",".join(header)]
        for row in result["rows"]:
            lines.append(",".join(str(row[key]).replace(",", " ") for key in header))
        Path(args.rows_out).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(
        {"summary": result["summary"], "warnings": result["warnings"]},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
