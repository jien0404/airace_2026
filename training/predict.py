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
from .build_ner_dataset import TOK_RE, ASSERTIONS, ASSERT_TYPES


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOK_RE.finditer(text)]


def chunk_tokens(tokens, tokenizer, max_len):
    """Chia token thành cửa sổ sao cho tổng subword <= max_len-2. Trả list (chunk_tokens,
    subword_ids_per_token)."""
    unk = tokenizer.unk_token_id or 0
    chunks, cur, cur_sub, n_sub = [], [], [], 0
    for tok in tokens:
        sub = tokenizer.encode(tok[0], add_special_tokens=False) or [unk]
        if cur and n_sub + len(sub) > max_len - 2:
            chunks.append((cur, cur_sub)); cur, cur_sub, n_sub = [], [], 0
        cur.append(tok); cur_sub.append(sub); n_sub += len(sub)
    if cur:
        chunks.append((cur, cur_sub))
    return chunks


@torch.no_grad()
def predict_note(text, model, tokenizer, meta, device, max_len=256, assert_thr=0.5):
    text = unicodedata.normalize("NFC", text)
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    tokens = tokenize(text)
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    tok_labels = []      # nhãn BIO cho từng token
    tok_assert = []      # vec 3 xác suất cho từng token
    for chunk_toks, chunk_sub in chunk_tokens(tokens, tokenizer, max_len):
        ids = [bos]
        first_pos = []   # vị trí subword đầu mỗi token (trong input_ids)
        for sub in chunk_sub:
            first_pos.append(len(ids)); ids.extend(sub)
        ids.append(eos)
        inp = torch.tensor([ids], device=device)
        att = torch.ones_like(inp)
        out = model(inp, att)
        ner = out["ner_logits"][0].argmax(-1).cpu().tolist()
        asr = torch.sigmoid(out["assert_logits"][0]).cpu().tolist()
        for p in first_pos:
            tok_labels.append(id2label.get(ner[p], "O"))
            tok_assert.append(asr[p])

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
        ents = predict_note(text, model, tokenizer, meta, device, args.max_len, args.assert_thr)
        with open(os.path.join(args.out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(ents, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(ents)} entity", flush=True)
    print(f"[done] -> {args.out_dir} (zip lại thành output.zip để nộp)")


if __name__ == "__main__":
    main()
