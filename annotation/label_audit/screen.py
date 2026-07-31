"""Sàng lọc bằng LUẬT trước khi hỏi LLM.

Gửi cả 18.000 entity của part1 + gt2 cho LLM vừa đắt vừa nhiễu. Ở đây chỉ giữ lại các entity
mà một luật CƠ HỌC trong `business_rules/CANONICAL.md` đã đủ để nghi ngờ, rồi mới đưa LLM phân
xử. Mỗi loại nghi ngờ nêu rõ luật bị đụng, để người review đọc được lý do.

Bốn nhóm nghi ngờ:

1. `assertion_without_cue` — entity mang assertion nhưng câu không có cue nào đỡ (CANONICAL §4.1:
   assertion phải dựa vào bằng chứng trong cùng mệnh đề/context).
2. `cue_without_assertion` — cue phủ định/tiền sử đứng ngay trước entity nhưng assertions rỗng.
3. `assertion_on_forbidden_type` — TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM có assertion. Đây là luật
   tuyệt đối, không cần LLM phân xử.
4. `historical_drug` — THUỐC mang isHistorical, nghiệp vụ còn OPEN (CANONICAL §4.2).

5. `span_swallows_negation` — span mang `isNegated` nhưng NUỐT LUÔN từ phủ định vào trong
   (`"Không đau đầu"` thay vì `"đau đầu"`). Phủ định là bằng chứng, không phải phần của concept.
6. `negation_inside_span` — span bắt đầu bằng từ phủ định mà assertions lại rỗng: hoặc thiếu
   `isNegated`, hoặc span đang lấn sang phần phủ định, thường là cả hai.
7. `type_minority_vs_corpus` — cùng surface được gán type khác với đa số áp đảo trong chính
   part1+gt2 (đa số ≥85% trên ≥4 occurrence). Type quyết định theo occurrence nên đây KHÔNG phải
   luật tuyệt đối; LLM phải đọc context rồi mới kết luận.

Nhóm 3 được đánh dấu `mechanical=True`: đúng/sai không phụ thuộc ngữ cảnh.

Ba nhóm 5-7 có thể đề xuất đổi `type` hoặc `position`, không chỉ `assertions`.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dataset_factory.pilot import _cues_before, _cues_support
from dataset_factory.schema import ASSERTION_TYPES, ASSERTIONS

# ±400 phủ trọn CÂU chứa entity 99,5% số ca (đo trên part1+gt2: p95 = 170 ký tự tới đầu câu,
# 156 tới cuối câu). Nới thêm nữa vô ích: ±700 cũng chỉ đưa được heading vào 49% số ca, nên
# heading được gắn RIÊNG thay vì trông chờ cửa sổ với tới.
CONTEXT_CHARS = 400
# Dòng heading: hoặc `Tên mục:` đứng riêng, hoặc dòng viết hoa không có chữ thường.
HEADING_LINE = re.compile(
    r"^[ \t]*(?:[^\n:]{3,45}:[ \t]*|[A-ZĐÀ-ỸÁÂĂÊÔƠƯ][^\na-zà-ỹáâăêôơư]{3,45})$", re.MULTILINE
)
# Từ phủ định đứng đầu span. `\S` cuối để không bắt span chỉ có mỗi từ phủ định.
LEADING_NEGATION = re.compile(r"^\s*(không|chưa|phủ nhận|ko)\s+(?=\S)", re.IGNORECASE)
# Ngưỡng cho `type_minority_vs_corpus`: đa số phải áp đảo và đủ số bản sao thì thiểu số mới đáng ngờ.
TYPE_MAJORITY_RATIO = 0.85
TYPE_MAJORITY_MIN_COUNT = 4


def _nearest_heading(text: str, start: int) -> str:
    """Heading/tên mục gần nhất phía trước entity.

    Quyết định gán hay không phụ thuộc thể thức phát ngôn của mục (bệnh sử vs giáo dục vs dặn dò),
    mà heading chỉ nằm trong cửa sổ ±320 ở 29,5% số ca. Nới cửa sổ không giải quyết được — ±1000
    cũng mới đạt 62% — nên lấy thẳng heading và gửi kèm.
    """
    best = ""
    for match in HEADING_LINE.finditer(text, 0, start):
        best = match.group().strip()
    return best


def _context(text: str, start: int, end: int) -> dict[str, Any]:
    left = max(0, start - CONTEXT_CHARS)
    right = min(len(text), end + CONTEXT_CHARS)
    return {
        "before": text[left:start],
        "surface": text[start:end],
        "after": text[end:right],
        "window_start": left,
        "heading": _nearest_heading(text, start),
    }


def screen_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = record["text"]
    findings: list[dict[str, Any]] = []
    for index, entity in enumerate(record["entities"]):
        start, end = entity["position"]
        if not (0 <= start < end <= len(text)) or text[start:end] != entity["text"]:
            findings.append({
                "kind": "offset_mismatch",
                "mechanical": True,
                "rule": "CANONICAL §3.1 — raw[start:end] phải bằng text",
                "current": {"assertions": entity.get("assertions") or [], "type": entity["type"]},
                "proposed": None,
                "reason": "Offset không khớp RAW; nhãn này chết khi chấm.",
                "file": record["id"],
                "entity_index": index,
                "entity": entity,
                "context": {"before": "", "surface": entity["text"], "after": "", "window_start": start},
            })
            continue
        assertions = list(entity.get("assertions") or [])
        typ = entity["type"]
        context = _context(text, start, end)
        base = {
            "file": record["id"],
            "entity_index": index,
            "entity": entity,
            "context": context,
        }
        if assertions and typ not in ASSERTION_TYPES:
            findings.append({
                **base,
                "kind": "assertion_on_forbidden_type",
                "mechanical": True,
                "rule": f"CANONICAL §4.1 — {typ} luôn có assertions rỗng",
                "current": {"assertions": assertions, "type": typ},
                "proposed": {"assertions": [], "type": typ},
                "reason": f"{typ} không được mang assertion trong bất kỳ ngữ cảnh nào.",
            })
            continue
        bad = sorted(set(assertions) - set(ASSERTIONS))
        if bad:
            findings.append({
                **base,
                "kind": "unknown_assertion",
                "mechanical": True,
                "rule": "DE_BAI — chỉ có isNegated/isFamily/isHistorical",
                "current": {"assertions": assertions, "type": typ},
                "proposed": {
                    "assertions": [a for a in assertions if a in ASSERTIONS], "type": typ,
                },
                "reason": f"Assertion không tồn tại trong schema: {bad}",
            })
            continue
        if typ not in ASSERTION_TYPES:
            continue
        support = _cues_support(text, [start, end])
        conflict = _cues_before(text, start)
        missing = [name for name in assertions if name not in support]
        if missing:
            findings.append({
                **base,
                "kind": "assertion_without_cue",
                "mechanical": False,
                "rule": "CANONICAL §4.1 — assertion phải dựa vào cue trong cùng mệnh đề",
                "current": {"assertions": assertions, "type": typ},
                "proposed": {
                    "assertions": [a for a in assertions if a in support], "type": typ,
                },
                "reason": (
                    f"Không thấy cue nào đỡ cho {missing} trong phạm vi câu chứa entity."
                ),
            })
            continue
        extra = sorted(conflict - set(assertions))
        if extra and not (typ == "THUỐC" and extra == ["isHistorical"]):
            findings.append({
                **base,
                "kind": "cue_without_assertion",
                "mechanical": False,
                "rule": "CANONICAL §4.1 — cue trực tiếp thì phải gán assertion tương ứng",
                "current": {"assertions": assertions, "type": typ},
                "proposed": {"assertions": sorted(set(assertions) | set(extra)), "type": typ},
                "reason": f"Cue {extra} đứng ngay trước entity nhưng nhãn để trống.",
            })
            continue
        negation = LEADING_NEGATION.match(entity["text"])
        if negation and typ in ASSERTION_TYPES:
            trimmed = entity["text"][negation.end():]
            proposed_position = [start + negation.end(), end]
            if "isNegated" in assertions:
                findings.append({
                    **base,
                    "kind": "span_swallows_negation",
                    "mechanical": False,
                    "rule": "CANONICAL §3.2 — span lấy concept lõi; từ phủ định là bằng chứng, "
                            "không phải phần của concept",
                    "current": {"assertions": assertions, "type": typ},
                    "proposed": {
                        "assertions": assertions, "type": typ,
                        "text": trimmed, "position": proposed_position,
                    },
                    "reason": (
                        f"Span đã mang isNegated mà vẫn chứa từ phủ định {negation.group().strip()!r}; "
                        f"concept lõi là {trimmed!r}."
                    ),
                })
            else:
                findings.append({
                    **base,
                    "kind": "negation_inside_span",
                    "mechanical": False,
                    "rule": "CANONICAL §3.2 + §4.1 — cắt từ phủ định khỏi span và gán isNegated",
                    "current": {"assertions": assertions, "type": typ},
                    "proposed": {
                        "assertions": sorted(set(assertions) | {"isNegated"}), "type": typ,
                        "text": trimmed, "position": proposed_position,
                    },
                    "reason": (
                        f"Span bắt đầu bằng {negation.group().strip()!r} nhưng assertions rỗng."
                    ),
                })
            continue
        if typ == "THUỐC" and "isHistorical" in assertions:
            findings.append({
                **base,
                "kind": "historical_drug",
                "mechanical": False,
                "rule": "CANONICAL §4.2 — isHistorical cho THUỐC cần thận trọng cao, nghiệp vụ còn OPEN",
                "current": {"assertions": assertions, "type": typ},
                "proposed": {
                    "assertions": [a for a in assertions if a != "isHistorical"], "type": typ,
                },
                "reason": "Thuốc đã ngừng/đã hết không mặc nhiên là historical.",
            })
    return findings


def type_majority(records: Iterable[dict[str, Any]]) -> dict[str, tuple[str, int, int]]:
    """`surface -> (type đa số, số lần của type đó, tổng số lần)` trên toàn corpus."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for entity in record["entities"]:
            counts[(entity.get("text") or "").strip().casefold()][entity["type"]] += 1
    table = {}
    for surface, per_type in counts.items():
        total = sum(per_type.values())
        winner, top = per_type.most_common(1)[0]
        table[surface] = (winner, top, total)
    return table


def screen_type_minority(
    record: dict[str, Any], table: dict[str, tuple[str, int, int]],
) -> list[dict[str, Any]]:
    """Occurrence mang type thiểu số so với đa số áp đảo của chính corpus nhãn tay.

    KHÔNG phải luật tuyệt đối: CANONICAL §2.1 nói type quyết định theo occurrence, và part2 có
    165 surface đổi type theo ngữ cảnh. Ngưỡng đa số ≥85% trên ≥4 occurrence để loại nhóm mơ hồ
    thật sự; phần còn lại vẫn phải để LLM đọc context.
    """
    text = record["text"]
    findings = []
    for index, entity in enumerate(record["entities"]):
        key = (entity.get("text") or "").strip().casefold()
        row = table.get(key)
        if not row:
            continue
        winner, top, total = row
        if entity["type"] == winner or total < TYPE_MAJORITY_MIN_COUNT:
            continue
        if top / total < TYPE_MAJORITY_RATIO:
            continue
        start, end = entity["position"]
        if not (0 <= start < end <= len(text)) or text[start:end] != entity["text"]:
            continue
        assertions = list(entity.get("assertions") or [])
        if winner not in ASSERTION_TYPES:
            assertions = []
        findings.append({
            "file": record["id"],
            "entity_index": index,
            "entity": entity,
            "context": _context(text, start, end),
            "kind": "type_minority_vs_corpus",
            "mechanical": False,
            "rule": "CANONICAL §2.1 — type theo occurrence, nhưng thiểu số cực đoan thường là nhầm",
            "current": {"assertions": list(entity.get("assertions") or []), "type": entity["type"]},
            "proposed": {"assertions": assertions, "type": winner},
            "reason": (
                f"{entity['text']!r} được gán {winner} {top}/{total} lần trong part1+gt2, "
                f"ở đây lại là {entity['type']}."
            ),
        })
    return findings


def screen_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(records)
    rows = [row for record in records for row in screen_record(record)]
    table = type_majority(records)
    seen = {(row["file"], row["entity_index"]) for row in rows}
    for record in records:
        for row in screen_type_minority(record, table):
            if (row["file"], row["entity_index"]) not in seen:
                rows.append(row)
    return rows


def load_labeled_dir(root: Path, source: str) -> list[dict[str, Any]]:
    notes, labels = root / "notes", root / "labels"
    records = []
    for note in sorted(notes.glob("*.txt"), key=lambda p: (len(p.stem), p.stem)):
        label_path = labels / f"{note.stem}.json"
        if not label_path.exists():
            continue
        records.append({
            "id": f"{source}:{note.stem}",
            "source": source,
            "file_stem": note.stem,
            "text": note.read_text(encoding="utf-8"),
            "entities": json.loads(label_path.read_text(encoding="utf-8")),
        })
    return records


def screen_delete_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Đưa MỌI entity ra soát lại quyết định gán/không gán.

    Không lọc bằng từ khoá: thử rồi và tỷ lệ trúng gần bằng 0 — `gợi ý`, `điển hình`, `cần` xuất
    hiện đầy trong câu mô tả bệnh nhân thật. Thể thức phát ngôn phải do LLM đọc mà phân biệt, với
    đủ câu chứa entity (cửa sổ ±400 phủ 99,5%) và heading của mục (gắn riêng).

    Chỉ đề xuất XOÁ, không bao giờ đề xuất thêm: xoá nhãn sai lãi gấp ~13 lần thêm nhãn đúng, và
    nguồn "thiếu nhãn" lớn nhất là occurrence lặp — thêm chúng đúng là luật đã bị probe bác bỏ.
    """
    text = record["text"]
    rows = []
    for index, entity in enumerate(record["entities"]):
        start, end = entity["position"]
        if not (0 <= start < end <= len(text)) or text[start:end] != entity["text"]:
            continue
        rows.append({
            "file": record["id"],
            "entity_index": index,
            "entity": entity,
            "context": _context(text, start, end),
            "kind": "should_not_be_labeled",
            "mechanical": False,
            "rule": "CANONICAL §1.1 — không phải mọi thuật ngữ y khoa xuất hiện đều là entity",
            "current": {
                "assertions": list(entity.get("assertions") or []), "type": entity["type"],
            },
            "proposed": {"delete": True},
            "reason": "Soát lại quyết định gán theo thể thức phát ngôn.",
        })
    return rows
