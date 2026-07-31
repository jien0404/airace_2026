# -*- coding: utf-8 -*-
"""Phủ quyết các phán xử SAI RÕ RÀNG của LLM, rồi sinh `decisions.json` tự động.

Dùng khi cần một bản nhãn đã sửa để ĐO THỬ, chưa qua duyệt tay. Mọi phủ quyết đều dựa trên bằng
chứng đã đo, không dựa trên cảm tính về từng ca.

Ba luật phủ quyết:

1. **Chỉ được xoá `TRIỆU_CHỨNG`.** Đo trực tiếp trên bảng xếp hạng: xoá `TÊN_XÉT_NGHIỆM` lỗ
   −0,0096, xoá `CHẨN_ĐOÁN`/`THUỐC` lỗ −0,0071, còn cắt tỉa `TRIỆU_CHỨNG` là thao tác duy nhất
   từng có lãi. Bằng chứng đó đo trên bài nộp chứ không phải trên dữ liệu train, nên đây là lựa
   chọn THẬN TRỌNG chứ không phải chân lý — nhưng khi chưa có gì tốt hơn thì đi theo hướng đã
   biết là an toàn.

2. **Không xoá entity mang assertion.** Nếu người gán đã bỏ công xác định phủ định/tiền sử/gia
   đình cho nó thì đó là một finding có chủ ý, không phải danh từ vơ vẩn.

3. **Không đổi type sang `TÊN_XÉT_NGHIỆM`/`KẾT_QUẢ_XÉT_NGHIỆM` nếu entity đang mang assertion**
   mà LLM lại không xoá assertion — mâu thuẫn nội tại, luật tuyệt đối bắt hai type này rỗng.

    python -m annotation.label_audit.postprocess --audit-dir annotation/data/label_audit
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_factory.schema import ASSERTION_TYPES

from .run import load_all_findings

DELETABLE_TYPES = {"TRIỆU_CHỨNG"}


def veto(finding: dict[str, Any]) -> str | None:
    """Trả về lý do phủ quyết, hoặc None nếu đề xuất được giữ."""
    current_type = finding["current"]["type"]
    assertions = finding["current"]["assertions"]
    if finding.get("proposed_delete"):
        if current_type not in DELETABLE_TYPES:
            return f"chỉ được xoá TRIỆU_CHỨNG; đây là {current_type}"
        if assertions:
            return f"entity mang assertion {assertions} — là finding có chủ ý"
        return None
    proposed_type = finding.get("proposed_type")
    if (
        proposed_type
        and proposed_type not in ASSERTION_TYPES
        and finding.get("proposed_assertions")
    ):
        return f"đổi sang {proposed_type} mà vẫn giữ assertion — trái luật tuyệt đối"
    return None


def build_decisions(audit_dir: Path) -> dict[str, Any]:
    findings = load_all_findings(audit_dir)
    decisions: dict[str, str] = {}
    vetoed: Counter = Counter()
    kept: Counter = Counter()
    for key, finding in findings.items():
        reason = veto(finding)
        if reason:
            decisions[key] = "reject"
            vetoed[reason.split(";")[0].split("—")[0].strip()] += 1
        else:
            decisions[key] = "accept"
            kept["xoá" if finding.get("proposed_delete") else "sửa"] += 1
    (audit_dir / "decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    report = {
        "stage": "label_audit_autodecided",
        "warning": "quyết định TỰ ĐỘNG, chưa qua duyệt tay — chỉ dùng để đo thử",
        "findings": len(findings),
        "accepted": dict(kept),
        "vetoed": dict(vetoed),
        "vetoed_total": sum(vetoed.values()),
    }
    (audit_dir / "autodecide_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit-dir", default="annotation/data/label_audit")
    args = parser.parse_args()
    print(json.dumps(build_decisions(Path(args.audit_dir)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
