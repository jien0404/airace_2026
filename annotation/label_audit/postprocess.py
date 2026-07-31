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

4. **Không cắt từ phủ định khi phần còn lại vô nghĩa.** Cắt `không` khỏi span rồi gán `isNegated`
   là đúng với đa số (`không sốt` → `sốt` + `isNegated`, khớp 2.093 ca gold), nhưng sai với hai
   lớp: `không thể X` (đơn vị là `không thể`, cắt ra còn `thể X` vỡ ngữ pháp) và các cụm mà gold
   Part 3 đã gán NGUYÊN CỤM với assertions rỗng vì sự bất lực chính là triệu chứng
   (`không vững khi đứng`, `không nhấc chân phải khỏi mặt giường`). Đo trên lượt chạy đầu:
   80/94 ca cắt đúng, 14 ca thuộc hai lớp này.

    python -m annotation.label_audit.postprocess --audit-dir annotation/data/label_audit
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from business_rules.artifacts import current_labels_zip
from dataset_factory.schema import ASSERTION_TYPES

from .run import load_all_findings

DELETABLE_TYPES = {"TRIỆU_CHỨNG"}

LEADING_NEGATOR = re.compile(r"^(không|chưa|ko|phủ nhận)\s+", re.IGNORECASE)
# `không thể` là MỘT đơn vị. Cắt riêng `không` ra khỏi `không thể vận động chi dưới` để lại
# `thể vận động chi dưới` — vô nghĩa về ngữ pháp. `nên`/`còn` cùng kiểu.
MODAL_REMNANT = re.compile(r"^(thể|nên|còn)\b", re.IGNORECASE)


def _part3_whole_negated_surfaces() -> frozenset[str]:
    """Cụm mở đầu bằng từ phủ định mà gold Part 3 gán NGUYÊN CỤM với assertions rỗng.

    Đây là lớp "sự bất lực CHÍNH LÀ triệu chứng": `không thể tự đứng dậy`, `không vững khi đứng`,
    `không nhấc chân phải khỏi mặt giường`. Gold đã phán trên đúng những surface này nên không
    được cắt chúng — cắt ra thì phần còn lại mô tả người khoẻ mạnh, và gán thêm `isNegated` là
    một assertion FP so với gold.
    """
    surfaces: set[str] = set()
    with zipfile.ZipFile(current_labels_zip()) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            for entity in json.loads(archive.read(name)):
                text = entity["text"].strip()
                if LEADING_NEGATOR.match(text) and not entity.get("assertions"):
                    surfaces.add(text.casefold())
    return frozenset(surfaces)


@lru_cache(maxsize=1)
def part3_whole_negated_surfaces() -> frozenset[str]:
    return _part3_whole_negated_surfaces()


def veto(finding: dict[str, Any]) -> str | None:
    """Trả về lý do phủ quyết, hoặc None nếu đề xuất được giữ."""
    current_type = finding["current"]["type"]
    assertions = finding["current"]["assertions"]
    proposed_text = finding.get("proposed_text")
    if proposed_text:
        original = (finding.get("entity") or {}).get("text") or finding.get(
            "context", {}
        ).get("surface", "")
        match = LEADING_NEGATOR.match(original.strip())
        if match and original.strip()[match.end():].strip() == proposed_text.strip():
            if MODAL_REMNANT.match(proposed_text.strip()):
                return (
                    f"cắt {original.strip()!r} để lại {proposed_text.strip()!r} — "
                    "`không thể` là một đơn vị, không tách được"
                )
            if original.strip().casefold() in part3_whole_negated_surfaces():
                return (
                    f"gold Part 3 gán {original.strip()!r} NGUYÊN CỤM với assertions rỗng — "
                    "sự bất lực chính là triệu chứng"
                )
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
