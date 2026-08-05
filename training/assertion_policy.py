"""Assertion policy dùng chung cho inference mới và hậu xử lý prediction cũ.

Policy ``part3`` chỉ chứa những luật đã được probe nhãn-thuần xác nhận:

* không xuất ``isFamily``;
* không xuất ``isNegated`` trên ``CHẨN_ĐOÁN``;
* assertion chỉ áp dụng cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC.

Data synthetic dùng cùng convention: ``isFamily`` và ``isNegated`` trên ``CHẨN_ĐOÁN`` không
được dùng làm supervision trực tiếp; nếu cần xử lý các pattern hiếm đó thì làm ở hậu xử lý.

Không chặn ``isHistorical`` trên THUỐC ở inference: nghiệp vụ này chỉ yêu cầu thận trọng,
không phải luật cấm tuyệt đối.  ``legacy`` giữ nguyên output để làm đối chứng.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema_v2 import ASSERTIONS, ASSERTION_TYPES

POLICIES = ("legacy", "part3")
EDUCATION_CUE_RE = re.compile(
    r"(?:nguyên nhân|thường gặp|có thể gặp|biểu hiện|dấu hiệu|yếu tố nguy cơ|"
    r"được định nghĩa|là gì|phòng ngừa)", re.IGNORECASE,
)
PATIENT_GROUNDING_RE = re.compile(
    r"(?:bệnh nhân|người bệnh|người nhà|tôi|em|cháu|tiền sử|đã từng|"
    r"trước nhập viện|đợt trước|lần khám trước)", re.IGNORECASE,
)


def resolve_thresholds(
    global_threshold: float,
    overrides: Mapping[str, float | None] | None = None,
) -> dict[str, float]:
    """Trả threshold cho từng assertion, dùng global khi không có override."""
    values = {name: float(global_threshold) for name in ASSERTIONS}
    for name, value in (overrides or {}).items():
        if name not in values:
            raise ValueError(f"Assertion không hợp lệ: {name}")
        if value is not None:
            values[name] = float(value)
    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold {name} phải trong [0,1], nhận {value}")
    return values


def resolve_type_thresholds(
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Validate threshold overrides keyed by assertion then entity type.

    The returned mapping is deliberately sparse.  Callers fall back to the existing
    per-assertion/global threshold when a pair is absent, so old command lines and checkpoints
    keep exactly the same behaviour.
    """
    values: dict[str, dict[str, float]] = {}
    for assertion, by_type in (overrides or {}).items():
        if assertion not in ASSERTIONS:
            raise ValueError(f"Assertion không hợp lệ trong threshold map: {assertion}")
        if not isinstance(by_type, Mapping):
            raise ValueError(f"Threshold map của {assertion} phải là object theo entity type")
        current: dict[str, float] = {}
        for entity_type, raw_value in by_type.items():
            if entity_type not in ASSERTION_TYPES:
                raise ValueError(
                    f"Entity type không hỗ trợ assertion trong threshold map: {entity_type}"
                )
            value = float(raw_value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Threshold {assertion}/{entity_type} phải trong [0,1], nhận {value}"
                )
            current[entity_type] = value
        values[assertion] = current
    return values


def load_type_thresholds(path: Path) -> dict[str, dict[str, float]]:
    """Load ``{assertion: {entity_type: threshold}}`` from JSON."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Assertion threshold map phải là một JSON object")
    return resolve_type_thresholds(raw)


def filter_assertions(
    entity_type: str,
    assertions: Iterable[str],
    policy: str = "part3",
) -> list[str]:
    """Áp firewall mà không đổi thứ tự assertion còn lại."""
    if policy not in POLICIES:
        raise ValueError(f"Policy không hợp lệ: {policy}")
    values = list(assertions)
    if policy == "legacy":
        return values
    if entity_type not in ASSERTION_TYPES:
        return []
    output = [name for name in values if name != "isFamily"]
    if entity_type == "CHẨN_ĐOÁN":
        output = [name for name in output if name != "isNegated"]
    return output


def assertions_from_probabilities(
    entity_type: str,
    names: Sequence[str],
    probabilities: Sequence[float],
    thresholds: Mapping[str, float],
    policy: str = "part3",
    type_thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> list[str]:
    if len(names) != len(probabilities):
        raise ValueError("Số assertion và probability không khớp")
    selected = []
    for name, probability in zip(names, probabilities):
        threshold = (type_thresholds or {}).get(name, {}).get(
            entity_type, thresholds[name]
        )
        if float(probability) >= threshold:
            selected.append(name)
    return filter_assertions(entity_type, selected, policy)


def apply_policy_to_predictions(
    rows: list[dict],
    policy: str = "part3",
) -> tuple[list[dict], Counter]:
    """Sao chép prediction và chỉ sửa trường assertions."""
    output = copy.deepcopy(rows)
    stats: Counter = Counter()
    for entity in output:
        before = list(entity.get("assertions") or [])
        after = filter_assertions(entity.get("type"), before, policy)
        if before != after:
            stats["entities_changed"] += 1
            for name in before:
                if name not in after:
                    stats[f"removed:{entity.get('type')}:{name}"] += 1
            entity["assertions"] = after
    return output, stats


def apply_optional_historical_education_firewall(
    text: str, rows: list[dict], context_characters: int = 240,
) -> tuple[list[dict], Counter]:
    """Remove historical only in clearly generic educational context.

    This rule is intentionally *not* part of ``part3`` and must remain opt-in until a reviewed
    external audit proves >=30% historical-FP reduction with zero v67 regression.
    """
    output = copy.deepcopy(rows)
    stats: Counter = Counter()
    for entity in output:
        assertions = list(entity.get("assertions") or [])
        if "isHistorical" not in assertions:
            continue
        start, end = entity["position"]
        left = text.rfind("\n\n", max(0, start - context_characters), start)
        right = text.find("\n\n", end, min(len(text), end + context_characters))
        context = text[left + 2 if left >= 0 else max(0, start - context_characters):
                       right if right >= 0 else min(len(text), end + context_characters)]
        if EDUCATION_CUE_RE.search(context) and not PATIENT_GROUNDING_RE.search(context):
            entity["assertions"] = [name for name in assertions if name != "isHistorical"]
            stats["removed_historical_education"] += 1
    return output, stats


def postprocess_zip(input_zip: Path, output_zip: Path, policy: str) -> dict:
    if input_zip.resolve() == output_zip.resolve():
        raise ValueError("Không ghi đè ZIP nguồn; hãy chọn --out-zip khác")
    if output_zip.exists():
        raise FileExistsError(f"Output đã tồn tại: {output_zip}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter = Counter()
    with zipfile.ZipFile(input_zip) as source, zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        json_files = 0
        for info in source.infolist():
            raw = source.read(info.filename)
            if not info.filename.endswith(".json"):
                target.writestr(info, raw)
                continue
            rows = json.loads(raw)
            if not isinstance(rows, list):
                raise ValueError(f"{info.filename}: prediction phải là JSON list")
            filtered, stats = apply_policy_to_predictions(rows, policy)
            # Bảo đảm postprocess không thể vô tình sửa field khác assertion.
            for before, after in zip(rows, filtered):
                left = {key: value for key, value in before.items() if key != "assertions"}
                right = {key: value for key, value in after.items() if key != "assertions"}
                if left != right:
                    raise RuntimeError(f"{info.filename}: field ngoài assertions bị đổi")
            target.writestr(
                info.filename,
                json.dumps(filtered, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            totals.update(stats)
            json_files += 1
        totals["json_files"] = json_files
    return {
        "input": str(input_zip),
        "output": str(output_zip),
        "policy": policy,
        **dict(totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-zip", required=True)
    parser.add_argument("--out-zip", required=True)
    parser.add_argument("--policy", choices=POLICIES, default="part3")
    args = parser.parse_args()
    report = postprocess_zip(Path(args.input_zip), Path(args.out_zip), args.policy)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
