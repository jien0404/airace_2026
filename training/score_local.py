"""Chấm local theo đúng cơ chế chấm đã cô lập bằng thực nghiệm trên leaderboard.

Điểm khác biệt so với F1 exact-match thông thường:

- **Khớp theo CHỒNG LẤN, ranh giới không tính.** Một span dự đoán khớp entity gold khi hai
  khoảng giao nhau ít nhất một ký tự VÀ cùng `type`. Nới/cắt đầu-đuôi span đã được đo là không
  làm đổi thành phần J_assertion/J_candidates (chiếm 0,7 trọng số).
- Ranh giới chỉ ảnh hưởng WER (0,3 trọng số) nên báo cáo riêng dưới dạng `boundary_f1` để tham
  khảo, không trộn vào chỉ số chính.
- Ghép 1-1 tham lam theo độ chồng lấn giảm dần: một gold chỉ được tính một lần, span thừa là FP.

Vì vậy `overlap_f1` ở đây mới là chỉ số đi cùng chiều leaderboard; `boundary_f1` luôn thấp hơn
và KHÔNG nên dùng để chọn checkpoint.

Dùng:

    python -m training.score_local --pred runs/.../pred_part3 \\
        --notes input_turn2      # gold mặc định lấy artifact hiện hành trong MANIFEST
    python -m training.score_local --pred <dir> --gold-dir <dir labels>
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from business_rules.artifacts import current_labels_zip

from .schema_v2 import ASSERTIONS, TYPES


def _load_dir(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        item.stem: json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(path.glob("*.json"))
    }


def _load_zip(path: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".json"):
                out[Path(name).stem] = json.loads(archive.read(name))
    return out


def match(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Ghép tham lam theo số ký tự chồng lấn; cùng type mới được ghép."""
    pairs = []
    for g_index, g in enumerate(gold):
        gs, ge = g["position"]
        for p_index, p in enumerate(pred):
            if p["type"] != g["type"]:
                continue
            ps, pe = p["position"]
            overlap = min(ge, pe) - max(gs, ps)
            if overlap > 0:
                pairs.append((overlap, g_index, p_index))
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matched = []
    for _, g_index, p_index in pairs:
        if g_index in used_gold or p_index in used_pred:
            continue
        used_gold.add(g_index)
        used_pred.add(p_index)
        matched.append((g_index, p_index))
    return matched


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(100 * precision, 2),
        "recall": round(100 * recall, 2),
        "f1": round(100 * f1, 2),
    }


def score(
    gold_by_file: dict[str, list[dict[str, Any]]],
    pred_by_file: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    tp = n_pred = n_gold = 0
    exact_tp = 0
    per_type: dict[str, Counter] = defaultdict(Counter)
    assertion = {name: Counter() for name in ASSERTIONS}
    type_confusion: Counter = Counter()
    missing_files = sorted(set(gold_by_file) - set(pred_by_file))
    extra_words = 0

    for file_id, gold in gold_by_file.items():
        pred = pred_by_file.get(file_id, [])
        n_gold += len(gold)
        n_pred += len(pred)
        for entity in gold:
            per_type[entity["type"]]["gold"] += 1
        for entity in pred:
            per_type[entity["type"]]["pred"] += 1
        matched = match(gold, pred)
        tp += len(matched)
        for g_index, p_index in matched:
            g, p = gold[g_index], pred[p_index]
            per_type[g["type"]]["tp"] += 1
            if g["position"] == p["position"]:
                exact_tp += 1
            extra_words += max(0, len(p["text"].split()) - len(g["text"].split()))
            for name in ASSERTIONS:
                in_gold = name in (g.get("assertions") or [])
                in_pred = name in (p.get("assertions") or [])
                if in_gold and in_pred:
                    assertion[name]["tp"] += 1
                elif in_pred:
                    assertion[name]["fp"] += 1
                elif in_gold:
                    assertion[name]["fn"] += 1
        # Span chạm gold nhưng SAI type: không tính khớp, nhưng đáng theo dõi riêng.
        matched_pred = {p_index for _, p_index in matched}
        for p_index, p in enumerate(pred):
            if p_index in matched_pred:
                continue
            ps, pe = p["position"]
            for g in gold:
                gs, ge = g["position"]
                if min(ge, pe) - max(gs, ps) > 0 and g["type"] != p["type"]:
                    type_confusion[f"{g['type']}→{p['type']}"] += 1
                    break

    report = {
        "files_scored": len(gold_by_file),
        "files_missing_prediction": missing_files,
        "gold_entities": n_gold,
        "pred_entities": n_pred,
        "overlap": _prf(tp, n_pred, n_gold),
        "boundary_f1_reference_only": _prf(exact_tp, n_pred, n_gold)["f1"],
        "exact_boundary_rate_of_matched": round(100 * exact_tp / max(tp, 1), 2),
        "extra_words_on_matched": extra_words,
        "per_type": {
            typ: {
                **_prf(per_type[typ]["tp"], per_type[typ]["pred"], per_type[typ]["gold"]),
                "gold": per_type[typ]["gold"],
                "pred": per_type[typ]["pred"],
            }
            for typ in TYPES
        },
        "type_confusion_top": dict(type_confusion.most_common(6)),
        "assertion_on_matched": {
            name: _prf(
                counts["tp"], counts["tp"] + counts["fp"], counts["tp"] + counts["fn"]
            )
            for name, counts in assertion.items()
        },
        "assertion_micro_on_matched": _prf(
            sum(counts["tp"] for counts in assertion.values()),
            sum(counts["tp"] + counts["fp"] for counts in assertion.values()),
            sum(counts["tp"] + counts["fn"] for counts in assertion.values()),
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", required=True, help="thư mục *.json dự đoán")
    parser.add_argument("--gold-dir", help="thư mục labels gold")
    parser.add_argument("--gold-zip", help="zip labels gold (artifact bài nộp)")
    parser.add_argument("--out", help="ghi report ra file JSON")
    parser.add_argument(
        "--drop-surfaces",
        help="file .txt, mỗi dòng một surface; bỏ chúng khỏi CẢ gold lẫn pred trước khi chấm. "
             "Dùng để đo trên phần Part 3 KHÔNG bị synthetic nhiễm.",
    )
    args = parser.parse_args()
    if args.gold_dir and args.gold_zip:
        raise SystemExit("Chọn đúng một trong --gold-dir hoặc --gold-zip")
    if args.gold_dir:
        gold = _load_dir(Path(args.gold_dir))
    else:
        # Không truyền gì thì lấy artifact hiện hành trong MANIFEST, khỏi ghi tên file bằng tay.
        gold = _load_zip(Path(args.gold_zip) if args.gold_zip else current_labels_zip())
    pred = _load_dir(Path(args.pred))
    dropped = 0
    if args.drop_surfaces:
        blocked = {
            line.strip().lower()
            for line in Path(args.drop_surfaces).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        def sieve(store):
            nonlocal dropped
            for name, rows in store.items():
                keep = [r for r in rows if (r.get("text") or "").strip().lower() not in blocked]
                dropped += len(rows) - len(keep)
                store[name] = keep

        sieve(gold)
        sieve(pred)
    report = score(gold, pred)
    if args.drop_surfaces:
        report["dropped_by_surface_filter"] = dropped
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
