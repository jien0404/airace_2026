# -*- coding: utf-8 -*-
"""Đo lại tập `NEGATABLE_CORES` của `build_ner_corpus` trên một corpus bất kỳ.

Quy ước gold: từ phủ định nằm NGOÀI span, entity mang `isNegated`. Nhưng có một lớp span mà
phủ định là một phần KHÔNG TÁCH ĐƯỢC của phát hiện (`không thể tự đứng dậy`, `chưa phát hiện
bất thường`) — cắt ra thì phần còn lại vô nghĩa hoặc đảo nghĩa.

Phân biệt bằng bằng chứng: chỉ cắt khi phần lõi TỰ NÓ đã là entity đứng riêng trong gold thật
(gt2 + part1 + part3) ít nhất `--min-support` lần. Script này in ra danh sách ứng viên kèm số
lần đỡ, để người đọc quyết định — nó KHÔNG tự sửa hằng số.

    python -m training.negation_span_audit --data datasets/ner_v4/track_a
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .build_ner_corpus import LEADING_NEGATOR, NEGATABLE_CORES

GOLD_SOURCES = ("gt2", "part1", "part3")
PRECEDING_NEGATOR = re.compile(r"(không|chưa|ko|phủ nhận)\s*$")


def audit(data_dir: Path, min_support: int) -> dict:
    standalone: Counter = Counter()
    negator_initial: list[tuple[str, str, str]] = []
    convention: Counter = Counter()
    for name in ("train", "validation", "test"):
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            source, text = record["source"], record["text"]
            for entity in record.get("entities") or []:
                surface = entity["text"].strip()
                if LEADING_NEGATOR.match(surface):
                    negator_initial.append((source, surface, entity["type"]))
                    if "isNegated" in (entity.get("assertions") or []):
                        convention["phủ định TRONG span"] += 1
                elif source in GOLD_SOURCES:
                    standalone[(surface.lower(), entity["type"])] += 1
                start = entity["position"][0]
                prefix = text[max(0, start - 14):start].lower()
                if PRECEDING_NEGATOR.search(prefix) and "isNegated" in (
                    entity.get("assertions") or []
                ):
                    convention["phủ định NGOÀI span"] += 1

    candidates: Counter = Counter()
    for source, surface, entity_type in negator_initial:
        core = LEADING_NEGATOR.sub("", surface).strip()
        support = standalone.get((core.lower(), entity_type), 0)
        if support >= min_support:
            candidates[(core.lower(), support)] += 1
    return {
        "quy_ước_gold": dict(convention),
        "span_bắt_đầu_bằng_phủ_định": len(negator_initial),
        "min_support": min_support,
        "ứng_viên_cắt": [
            {"lõi": core, "gold_đỡ": support, "số_span": count,
             "đang_bật": core in NEGATABLE_CORES}
            for (core, support), count in sorted(candidates.items(), key=lambda kv: -kv[1])
        ],
        "đã_bật_và_không_còn_span_nào_chờ_cắt": sorted(
            core for core in NEGATABLE_CORES
            if not any(core == candidate for candidate, _ in candidates)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="datasets/ner_v4/track_a")
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.data), args.min_support), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
