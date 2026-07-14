# -*- coding: utf-8 -*-
"""Inference: input/*.txt -> output/*.json đúng format thi (text, position, type,
assertions, candidates). Dùng model multi-task đã train (type + assertion).
candidates để rỗng (bước normalization làm sau).

  python -m training.predict --model_dir runs/xlmr/best \
      --input_dir input --out_dir output
  # nộp: zip output.zip từ thư mục output/
"""

import argparse
import glob
import json
import os
import unicodedata

import torch

from .model import MultiTaskNER
from .build_ner_dataset import ASSERTIONS, ASSERT_TYPES
from .windowing import make_windows


@torch.no_grad()
def predict_note(text, model, tokenizer, meta, device, max_len=256, assert_thr=0.5,
                 max_words=110):
    """SECTION-AWARE (khớp train): cắt window theo mục, window nối tiếp mang header mục làm
    PREFIX ngữ cảnh (đưa vào model nhưng KHÔNG decode). Chỉ decode token CONTENT -> mỗi token
    note được gán đúng 1 lần, offset chuẩn."""
    text = unicodedata.normalize("NFC", text)
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    unk = tokenizer.unk_token_id or 0

    tokens = []          # (tok,s,e) content theo thứ tự = đúng chuỗi token note
    tok_labels = []      # nhãn BIO tương ứng
    tok_assert = []      # vec 3 xác suất
    for w in make_windows(text, max_words):
        content, prefix = w["content"], w["prefix"]
        if not content:
            continue
        c_sub = [tokenizer.encode(t[0], add_special_tokens=False) or [unk] for t in content]
        p_sub = [s for t in prefix
                 for s in (tokenizer.encode(t[0], add_special_tokens=False) or [unk])]
        i = 0
        while i < len(content):
            ids = [bos]
            if 0 < len(p_sub) <= max_len - 40:        # prefix ngữ cảnh nếu còn chỗ
                ids += p_sub
            first_pos, start = [], i
            while i < len(content) and len(ids) + len(c_sub[i]) <= max_len - 1:
                first_pos.append(len(ids)); ids += c_sub[i]; i += 1
            if i == start:                            # 1 token quá dài -> cắt
                first_pos.append(len(ids)); ids += c_sub[i][:max_len - len(ids) - 1]; i += 1
            ids.append(eos)
            inp = torch.tensor([ids], device=device)
            out = model(inp, torch.ones_like(inp))
            ner = out["ner_logits"][0].argmax(-1).cpu().tolist()
            asr = torch.sigmoid(out["assert_logits"][0]).cpu().tolist()
            for k, pos in enumerate(first_pos):
                tokens.append(content[start + k])
                tok_labels.append(id2label.get(ner[pos], "O"))
                tok_assert.append(asr[pos])

    # BIO decode -> span
    ents = []
    i, n = 0, len(tokens)
    while i < n:
        lab = tok_labels[i]
        if lab == "O":
            i += 1; continue
        typ = lab.split("-", 1)[1] if "-" in lab else lab
        j = i + 1
        while j < n and tok_labels[j] == "I-" + typ:
            j += 1
        s = tokens[i][1]; e = tokens[j - 1][2]
        # assertion: gộp các token trong span (any > threshold), chỉ loại có-assertion
        assertions = []
        if typ in ASSERT_TYPES:
            probs = [tok_assert[k] for k in range(i, j)]
            for ai, a in enumerate(ASSERTIONS):
                if max(p[ai] for p in probs) >= assert_thr:
                    assertions.append(a)
        ents.append({"text": text[s:e], "position": [s, e], "type": typ,
                     "assertions": assertions, "candidates": []})
        i = j
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="thư mục best (có mtl_meta.json)")
    ap.add_argument("--input_dir", default="input")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_words", type=int, default=110,
                    help="độ dài window (word) — PHẢI khớp lúc build dataset để nhất quán")
    ap.add_argument("--assert_thr", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model, meta = MultiTaskNER.load(args.model_dir, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                   if os.path.splitext(os.path.basename(p))[0].isdigit() else 1e9)
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        text = open(fp, encoding="utf-8").read()
        ents = predict_note(text, model, tokenizer, meta, device, args.max_len,
                            args.assert_thr, args.max_words)
        with open(os.path.join(args.out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(ents, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(ents)} entity", flush=True)
    print(f"[done] -> {args.out_dir} (zip lại thành output.zip để nộp)")


if __name__ == "__main__":
    main()
