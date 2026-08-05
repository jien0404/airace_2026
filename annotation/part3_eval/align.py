"""Align surface do LLM trả về về offset RAW một cách tất định.

LLM phải chỉ ra line_id + occurrence. Không fuzzy-match và không tự lan một
prediction ra mọi occurrence, vì cả hai hành vi từng gây lỗi có hệ thống.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from .prompts import ASSERTIONS, TYPES


CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
ICD_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$")
RXCUI_RE = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class Line:
    line_id: int
    text: str
    start: int
    end: int


def raw_lines(raw: str) -> list[Line]:
    out: list[Line] = []
    offset = 0
    for line_id, chunk in enumerate(raw.splitlines(keepends=True), 1):
        text = chunk.rstrip("\r\n")
        out.append(Line(line_id, text, offset, offset + len(text)))
        offset += len(chunk)
    if not out and raw == "":
        return []
    if offset < len(raw):
        out.append(Line(len(out) + 1, raw[offset:], offset, len(raw)))
    return out


def _nfc_maps(raw: str) -> tuple[str, list[int], list[int]]:
    """NFC string và ánh xạ mỗi ký tự NFC về biên RAW."""
    parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i = 0
    while i < len(raw):
        j = i + 1
        while j < len(raw) and unicodedata.combining(raw[j]):
            j += 1
        segment = unicodedata.normalize("NFC", raw[i:j])
        for index in range(len(segment)):
            starts.append(i if index == 0 else j)
            ends.append(j)
        parts.append(segment)
        i = j
    return "".join(parts), starts, ends


def _all_matches(haystack: str, needle: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    cursor = 0
    while needle and cursor <= len(haystack):
        start = haystack.find(needle, cursor)
        if start < 0:
            break
        end = start + len(needle)
        matches.append((start, end))
        cursor = start + 1
    return matches


def _word_bounded(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start].isalnum() and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        return False
    return True


def _find_in_raw(
    raw: str,
    quote: str,
    line: Line | None,
    occurrence: int,
) -> tuple[int, int] | None:
    """Exact/NFC match. Nếu thiếu line_id chỉ nhận unique match toàn văn."""
    scope = line.text if line else raw
    base = line.start if line else 0

    # Ưu tiên exact RAW.
    matches = [
        match for match in _all_matches(scope, quote)
        if _word_bounded(scope, *match)
    ]
    if matches:
        if line and occurrence <= len(matches):
            start, end = matches[occurrence - 1]
            return base + start, base + end
        if not line and len(matches) == 1:
            start, end = matches[0]
            return start, end

    # LLM thường trả NFC trong khi RAW có thể NFD.
    normalized_scope, starts, ends = _nfc_maps(scope)
    normalized_quote = unicodedata.normalize("NFC", quote)
    matches = [
        match for match in _all_matches(normalized_scope, normalized_quote)
        if _word_bounded(normalized_scope, *match)
    ]
    if line and occurrence <= len(matches):
        start, end = matches[occurrence - 1]
        return base + starts[start], base + ends[end - 1]
    if not line and len(matches) == 1:
        start, end = matches[0]
        return starts[start], ends[end - 1]
    return None


def _as_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
    return out


def sanitize_candidates(
    entity_type: str,
    candidates: Any,
    valid_icd: set[str],
    valid_rxnorm: set[str],
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for code in _clean_list(candidates):
        if entity_type == "CHẨN_ĐOÁN":
            normalized = code.upper()
            valid = bool(ICD_RE.fullmatch(normalized)) and normalized in valid_icd
        elif entity_type == "THUỐC":
            normalized = code
            valid = bool(RXCUI_RE.fullmatch(normalized)) and normalized in valid_rxnorm
        else:
            normalized = code
            valid = False
        target = kept if valid else dropped
        if normalized not in target:
            target.append(normalized)
    return kept, dropped


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def align_predictions(
    raw: str,
    predictions: Iterable[dict[str, Any]],
    valid_icd: set[str],
    valid_rxnorm: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Trả (competition labels, audit, aligned entities có metadata)."""
    lines = {line.line_id: line for line in raw_lines(raw)}
    proposed = list(predictions)
    candidates: list[dict[str, Any]] = []
    unaligned: list[dict[str, Any]] = []
    dropped_codes: list[dict[str, Any]] = []
    sanitizations: list[dict[str, Any]] = []

    for source_index, prediction in enumerate(proposed):
        if not isinstance(prediction, dict):
            unaligned.append({"index": source_index, "reason": "not_object"})
            continue
        entity_type = prediction.get("type")
        quote = prediction.get("text")
        if entity_type not in TYPES or not isinstance(quote, str) or not quote:
            unaligned.append({
                "index": source_index,
                "reason": "invalid_type_or_text",
                "prediction": prediction,
            })
            continue

        parsed_line_id = _optional_positive_int(prediction.get("line_id"))
        line_id = parsed_line_id or 0
        line = lines.get(parsed_line_id) if parsed_line_id is not None else None
        occurrence = _as_int(prediction.get("occurrence"), default=1)
        span = _find_in_raw(raw, quote, line, occurrence)
        adjusted_quote = quote
        if span is None and quote.strip() != quote:
            adjusted_quote = quote.strip()
            span = _find_in_raw(raw, adjusted_quote, line, occurrence)
            if span is not None:
                sanitizations.append({
                    "index": source_index,
                    "kind": "trimmed_quote_whitespace",
                    "before": quote,
                    "after": adjusted_quote,
                })
        if span is None:
            # Chỉ fallback toàn văn khi surface là unique; không đoán occurrence.
            span = _find_in_raw(raw, adjusted_quote, None, occurrence)
        if span is None:
            unaligned.append({
                "index": source_index,
                "reason": "surface_not_unique_or_not_found",
                "prediction": prediction,
            })
            continue

        assertions = [
            assertion for assertion in _clean_list(prediction.get("assertions"))
            if assertion in ASSERTIONS
        ]
        if entity_type in ("TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM") and assertions:
            sanitizations.append({
                "index": source_index,
                "kind": "assertions_removed_for_lab_type",
                "values": assertions,
            })
            assertions = []

        kept_codes, invalid_codes = sanitize_candidates(
            entity_type,
            prediction.get("candidates"),
            valid_icd,
            valid_rxnorm,
        )
        if invalid_codes:
            dropped_codes.append({
                "index": source_index,
                "text": raw[span[0]:span[1]],
                "type": entity_type,
                "codes": invalid_codes,
            })

        confidence = str(prediction.get("confidence", "medium")).lower()
        if confidence not in CONFIDENCE_RANK:
            confidence = "medium"
        candidates.append({
            "text": raw[span[0]:span[1]],
            "position": [span[0], span[1]],
            "type": entity_type,
            "assertions": assertions,
            "candidates": kept_codes,
            "_source_index": source_index,
            "_line_id": line_id,
            "_occurrence": occurrence,
            "_rule_ids": _clean_list(prediction.get("rule_ids")),
            "_confidence": confidence,
        })

    # Chọn span không overlap: confidence cao, span dài, rồi xuất hiện sớm.
    ranked = sorted(
        candidates,
        key=lambda item: (
            -CONFIDENCE_RANK[item["_confidence"]],
            -(item["position"][1] - item["position"][0]),
            item["position"][0],
            item["_source_index"],
        ),
    )
    kept: list[dict[str, Any]] = []
    overlap_dropped: list[dict[str, Any]] = []
    exact_keys: set[tuple[int, int, str]] = set()
    for entity in ranked:
        start, end = entity["position"]
        key = (start, end, entity["type"])
        if key in exact_keys:
            overlap_dropped.append({
                "reason": "duplicate",
                "entity": entity,
            })
            continue
        if any(_overlap((start, end), tuple(other["position"])) for other in kept):
            overlap_dropped.append({
                "reason": "overlap",
                "entity": entity,
            })
            continue
        exact_keys.add(key)
        kept.append(entity)
    kept.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))

    labels = [
        {
            "text": entity["text"],
            "position": entity["position"],
            "type": entity["type"],
            "assertions": entity["assertions"],
            "candidates": entity["candidates"],
        }
        for entity in kept
    ]
    audit = {
        "proposed": len(proposed),
        "aligned_before_overlap": len(candidates),
        "kept": len(labels),
        "alignment_rate": round(len(candidates) / max(1, len(proposed)), 4),
        "unaligned": unaligned,
        "overlap_dropped": overlap_dropped,
        "invalid_candidates_dropped": dropped_codes,
        "sanitizations": sanitizations,
    }
    return labels, audit, kept


def validate_labels(raw: str, labels: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    intervals: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, entity in enumerate(labels):
        position = entity.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(isinstance(value, int) for value in position)
        ):
            errors.append(f"[{index}] position không hợp lệ")
            continue
        start, end = position
        if not 0 <= start < end <= len(raw):
            errors.append(f"[{index}] offset ngoài RAW: {position}")
            continue
        if raw[start:end] != entity.get("text"):
            errors.append(f"[{index}] text không khớp RAW: {position}")
        if entity.get("type") not in TYPES:
            errors.append(f"[{index}] type không hợp lệ")
        assertions = entity.get("assertions")
        if not isinstance(assertions, list) or any(a not in ASSERTIONS for a in assertions):
            errors.append(f"[{index}] assertions không hợp lệ")
        if (
            entity.get("type") in ("TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM")
            and assertions
        ):
            errors.append(f"[{index}] lab type có assertion")
        candidates = entity.get("candidates")
        if not isinstance(candidates, list):
            errors.append(f"[{index}] candidates không phải list")
        if entity.get("type") not in ("CHẨN_ĐOÁN", "THUỐC") and candidates:
            errors.append(f"[{index}] non-code type có candidate")
        key = (start, end, entity.get("type"))
        if key in seen:
            errors.append(f"[{index}] duplicate {key}")
        seen.add(key)
        intervals.append((start, end, index))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if right[0] < left[1]:
            errors.append(f"overlap entity {left[2]} và {right[2]}")
    return errors
