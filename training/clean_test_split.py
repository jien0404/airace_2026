# -*- coding: utf-8 -*-
"""Tách nhóm tài liệu Part 3 KHÔNG trùng văn bản với Part 1, rồi chấm lại trên nhóm đó.

Part 1 là bộ test vòng trước, nhãn do ta gán tay (đạt 54,92), và **40,5% shingle 12 từ của
Part 3 xuất hiện nguyên văn trong Part 1** — 73/100 tài liệu Part 3 trùng trên 10%. Mọi nhánh
train có Part 1 vì thế được đọc lại chính văn bản đề bài, và điểm test cục bộ không phân biệt
được "học được cách gán" với "nhớ lại đoạn đã thấy".

Đích thật là private test, nơi không có tài liệu nào trùng Part 1. Chấm trên nhóm sạch mới là
con số dự đoán được private test.

    python -m training.clean_test_split --pred result/v2_ep10/labels result/v2_realonly/labels \\
      --gold-zip business_rules/artifacts/current/v66ab_...zip

Cảnh báo: nhóm sạch chỉ vài chục tài liệu nên sai số lấy mẫu lớn. Dùng để phát hiện chênh lệch
lớn, không dùng để tinh chỉnh ngưỡng.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .score_local import _load_dir, _load_zip, score

REPO_ROOT = Path(__file__).resolve().parent.parent
SHINGLE_WORDS = 12


def shingles(text: str, size: int = SHINGLE_WORDS) -> set[str]:
    words = re.findall(r"\S+", text.lower())
    return {" ".join(words[index:index + size]) for index in range(max(0, len(words) - size + 1))}


def clean_files(
    target_notes: Path, donor_notes: Path, max_overlap: float,
) -> tuple[list[str], dict[str, float]]:
    donor: set[str] = set()
    for path in donor_notes.glob("*.txt"):
        donor |= shingles(path.read_text(encoding="utf-8"))
    ratios: dict[str, float] = {}
    for path in sorted(target_notes.glob("*.txt")):
        own = shingles(path.read_text(encoding="utf-8"))
        ratios[path.stem] = len(own & donor) / len(own) if own else 0.0
    keep = [stem for stem, ratio in ratios.items() if ratio <= max_overlap]
    return keep, ratios


def _subset(store: dict[str, list[dict[str, Any]]], keep: set[str]) -> dict[str, list]:
    return {name: rows for name, rows in store.items() if name in keep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", nargs="+", required=True, help="một hoặc nhiều thư mục dự đoán")
    parser.add_argument("--gold-zip", required=True)
    parser.add_argument("--target-notes", default="input_turn2")
    parser.add_argument("--donor-notes", default="annotation/data/best_54.92/notes")
    parser.add_argument("--max-overlap", type=float, default=0.10)
    parser.add_argument("--out", help="ghi danh sách file sạch ra đây")
    args = parser.parse_args()

    keep, ratios = clean_files(
        REPO_ROOT / args.target_notes, REPO_ROOT / args.donor_notes, args.max_overlap,
    )
    print(
        f"{len(keep)}/{len(ratios)} tài liệu trùng Part 1 <= {100 * args.max_overlap:.0f}% "
        f"(trùng trung bình toàn bộ: {100 * sum(ratios.values()) / len(ratios):.1f}%)"
    )
    if args.out:
        Path(args.out).write_text("\n".join(keep) + "\n", encoding="utf-8")

    gold = _load_zip(Path(args.gold_zip))
    keep_set = set(keep) & set(gold)
    gold_clean = _subset(gold, keep_set)
    print(f"gold trên nhóm sạch: {sum(len(v) for v in gold_clean.values())} entity\n")
    header = f"{'nhánh':22} {'F1 toàn bộ':>11} {'F1 nhóm sạch':>13} {'chênh':>7} {'P sạch':>8} {'R sạch':>8}"
    print(header)
    print("-" * len(header))
    for pred_dir in args.pred:
        pred = _load_dir(Path(pred_dir))
        full = score(gold, pred)["overlap"]
        clean = score(gold_clean, _subset(pred, keep_set))["overlap"]
        name = Path(pred_dir).parent.name or Path(pred_dir).name
        print(
            f"{name:22} {full['f1']:>11.2f} {clean['f1']:>13.2f} "
            f"{clean['f1'] - full['f1']:>+7.2f} {clean['precision']:>8.1f} {clean['recall']:>8.1f}"
        )


if __name__ == "__main__":
    main()
