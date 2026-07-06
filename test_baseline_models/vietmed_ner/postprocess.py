#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phương án A: hậu xử lý output thô của VietMed-NER để bớt phân mảnh & false-positive.

Đầu vào : output/*.raw.json (entity thô 36 nhãn) + input/*.txt (text gốc).
Pipeline (mỗi stage LƯU RIÊNG để rà soát logic):
    stage1_stripped : cắt dấu câu / khoảng trắng ở 2 đầu span, bỏ span rỗng
    stage2_filtered : loại các từ tiêu đề mục / từ chung (BLACKLIST)
    stage3_merged   : nối span liền kề CÙNG DÒNG (same-type, hoặc gộp thuốc)
    final           : map 5 loại cuộc thi + dedup + format object (assertions/candidates rỗng)

KHÔNG cần torch/GPU. Chạy: python postprocess.py
"""

import argparse
import json
import os
import glob
import collections

from labels import LABEL_MAP, DRUG_GROUP, STRIP_CHARS, BLACKLIST

MERGE_MAX_GAP = 3  # số ký tự tối đa giữa 2 span để xét nối (chỉ khoảng trắng)


# ----------------------------------------------------------------------------
# Stage 1: strip dấu câu 2 đầu, tính lại position theo raw_text
# ----------------------------------------------------------------------------
def strip_entity(ent, raw_text):
    s, e = ent["position"]
    # cắt trái
    while s < e and raw_text[s] in STRIP_CHARS:
        s += 1
    # cắt phải
    while e > s and raw_text[e - 1] in STRIP_CHARS:
        e -= 1
    if s >= e:
        return None
    return {"text": raw_text[s:e], "position": [s, e], "vietmed_type": ent["vietmed_type"]}


def stage_strip(ents, raw_text):
    out = []
    for ent in ents:
        st = strip_entity(ent, raw_text)
        if st is not None:
            out.append(st)
    return out


# ----------------------------------------------------------------------------
# Stage 2: bỏ từ tiêu đề mục / từ chung
# ----------------------------------------------------------------------------
def stage_filter(ents):
    return [e for e in ents if e["text"].strip().lower() not in BLACKLIST]


# ----------------------------------------------------------------------------
# Stage 3: nối span liền kề cùng dòng
#   - cùng vietmed_type  -> nối, giữ type
#   - cùng NHÓM THUỐC (DRUGCHEMICAL/UNITCALIBRATOR) & có >=1 DRUGCHEMICAL -> nối, type=DRUGCHEMICAL
#   Điều kiện gap: chỉ khoảng trắng, KHÔNG xuống dòng, độ dài <= MERGE_MAX_GAP.
#   (Nhờ đó 'buồn nôn, hay nôn' KHÔNG bị nối vì gap chứa dấu phẩy.)
# ----------------------------------------------------------------------------
def _can_extend(run_types, new_type):
    types = set(run_types) | {new_type}
    if len(types) == 1:
        return True
    if types <= DRUG_GROUP and "DRUGCHEMICAL" in types:
        return True
    return False


def stage_merge(ents, raw_text):
    ents = sorted(ents, key=lambda e: e["position"][0])
    out = []
    i, n = 0, len(ents)
    while i < n:
        run = [ents[i]]
        run_types = {ents[i]["vietmed_type"]}
        j = i
        while j + 1 < n:
            a, b = ents[j], ents[j + 1]
            gap = raw_text[a["position"][1]:b["position"][0]]
            if (0 <= len(gap) <= MERGE_MAX_GAP and gap.strip() == "" and "\n" not in gap
                    and _can_extend(run_types, b["vietmed_type"])):
                run.append(b)
                run_types.add(b["vietmed_type"])
                j += 1
            else:
                break
        s = run[0]["position"][0]
        e = run[-1]["position"][1]
        if len(run_types) == 1:
            mtype = next(iter(run_types))
        else:  # nhóm thuốc trộn -> DRUGCHEMICAL
            mtype = "DRUGCHEMICAL"
        out.append({"text": raw_text[s:e], "position": [s, e],
                    "vietmed_type": mtype, "n_merged": len(run)})
        i = j + 1
    return out


# ----------------------------------------------------------------------------
# Final: map 5 loại + dedup (text chuẩn hóa + type) + format cuộc thi
# ----------------------------------------------------------------------------
def stage_final(ents):
    out = []
    seen = set()
    for e in ents:
        mapped = LABEL_MAP.get(e["vietmed_type"])
        if mapped is None:
            continue
        key = (e["text"].strip().lower(), mapped)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "text": e["text"],
            "position": e["position"],
            "type": mapped,
            "assertions": [],
            "candidates": [],
        })
    return out


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.join(here, "output"),
                    help="Thư mục chứa *.raw.json từ run_ner.py")
    ap.add_argument("--input_dir",
                    default=os.path.abspath(os.path.join(here, "..", "..", "input")))
    ap.add_argument("--out_dir", default=os.path.join(here, "postprocess_output"))
    args = ap.parse_args()

    stages = ["stage1_stripped", "stage2_filtered", "stage3_merged", "final"]
    for st in stages:
        os.makedirs(os.path.join(args.out_dir, st), exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "*.raw.json")),
                       key=lambda p: int(os.path.basename(p).split(".")[0])
                       if os.path.basename(p).split(".")[0].isdigit() else 1e9)

    counts = collections.OrderedDict((s, 0) for s in ["raw"] + stages)
    final_type = collections.Counter()

    for rf in raw_files:
        name = os.path.basename(rf).split(".")[0]
        raw = json.load(open(rf, encoding="utf-8"))
        raw_text = open(os.path.join(args.input_dir, f"{name}.txt"), encoding="utf-8").read()

        s1 = stage_strip(raw, raw_text)
        s2 = stage_filter(s1)
        s3 = stage_merge(s2, raw_text)
        fin = stage_final(s3)

        dump(s1, os.path.join(args.out_dir, "stage1_stripped", f"{name}.json"))
        dump(s2, os.path.join(args.out_dir, "stage2_filtered", f"{name}.json"))
        dump(s3, os.path.join(args.out_dir, "stage3_merged", f"{name}.json"))
        dump(fin, os.path.join(args.out_dir, "final", f"{name}.json"))

        counts["raw"] += len(raw)
        counts["stage1_stripped"] += len(s1)
        counts["stage2_filtered"] += len(s2)
        counts["stage3_merged"] += len(s3)
        counts["final"] += len(fin)
        for e in fin:
            final_type[e["type"]] += 1

    n = len(raw_files)
    print(f"Đã xử lý {n} file -> {args.out_dir}\n")
    print(f"{'STAGE':18s} {'TỔNG':>7s} {'TB/file':>8s}   giảm so với trước")
    prev = None
    for k, v in counts.items():
        delta = "" if prev is None else f"{v - prev:+d} ({100*(v-prev)/prev:+.1f}%)"
        print(f"{k:18s} {v:7d} {v/n:8.1f}   {delta}")
        prev = v
    print("\nPhân bố 5 loại (final):")
    tot = sum(final_type.values())
    for t, c in final_type.most_common():
        print(f"  {t:20s} {c:5d}  ({100*c/tot:4.1f}%)")


if __name__ == "__main__":
    main()
