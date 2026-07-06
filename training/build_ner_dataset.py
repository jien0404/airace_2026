# -*- coding: utf-8 -*-
"""Chuyển dữ liệu sinh (train.jsonl: text + entity char-span) -> dataset BIO token-level
model-agnostic cho token classification.

- Tokenize theo syllable/subword-độc-lập: regex tách chữ (gồm dấu tiếng Việt) và từng dấu
  câu, KÈM offset ký tự -> gán nhãn BIO từ span. Không cần VnCoreNLP (dùng bản syllable +
  XLM-R subword nên train ở mức syllable, khỏi word-segment).
- Chỉ dùng `type` cho nhãn NER (5 loại). `assertions` để dành model riêng.
- Chia: val/test tách Ở MỨC NOTE từ nguồn LLM (sát thật hơn); train = LLM còn lại + no-LLM.
- Đóng gói theo DÒNG rồi gộp thành cửa sổ <= max_words token (entity nằm gọn trong 1 dòng
  nên không bị cắt ngang).

    python -m training.build_ner_dataset \
        --llm data_gen/generated/train_v1 \
        --extra data_gen/generated/no-llm_train_v1 \
        --out training/dataset --val 0.1 --test 0.1
"""

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

TYPES = ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "THUỐC"]
# assertion CHỈ cho 3 loại này (DE_BAI); XN/kết quả luôn rỗng -> không giám sát.
ASSERT_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
ASSERTIONS = ["isNegated", "isFamily", "isHistorical"]
ASSERT_IDX = {a: i for i, a in enumerate(ASSERTIONS)}
TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _norm(s):
    return unicodedata.normalize("NFC", s)


def tokenize_line(line, base):
    """Trả list (token, start, end) offset toàn cục (base = offset đầu dòng)."""
    return [(m.group(), base + m.start(), base + m.end()) for m in TOK_RE.finditer(line)]


def bio_tags(tokens, entities):
    """Gán BIO + nhãn assertion cho list token (token,s,e).
    Trả (tags, assert_mask, assert_vecs):
      assert_mask[i] = 1 nếu token i thuộc entity loại có-assertion (giám sát assertion).
      assert_vecs[i] = [neg,fam,hist] (chỉ có nghĩa khi mask=1)."""
    n = len(tokens)
    tags = ["O"] * n
    amask = [0] * n
    avecs = [[0, 0, 0] for _ in range(n)]
    ents = sorted(entities, key=lambda x: (x["position"][0], -(x["position"][1] - x["position"][0])))
    for ent in ents:
        es, ee = ent["position"]
        typ = ent["type"]
        idxs = [i for i, (_, s, e) in enumerate(tokens)
                if s >= es and e <= ee and tags[i] == "O"]
        if not idxs:
            idxs = [i for i, (_, s, e) in enumerate(tokens)
                    if not (e <= es or s >= ee) and tags[i] == "O"]
        vec = [0, 0, 0]
        for a in ent.get("assertions", []):
            if a in ASSERT_IDX:
                vec[ASSERT_IDX[a]] = 1
        for k, i in enumerate(idxs):
            tags[i] = ("B-" if k == 0 else "I-") + typ
            if typ in ASSERT_TYPES:
                amask[i] = 1
                avecs[i] = list(vec)
    return tags, amask, avecs


def note_to_windows(text, entities, max_words=110):
    """text -> list cửa sổ {tokens, ner_tags}. Gộp theo dòng, entity gọn trong dòng."""
    text = _norm(text)
    # tách dòng giữ offset
    lines, base = [], 0
    for ln in text.split("\n"):
        lines.append((ln, base))
        base += len(ln) + 1  # + '\n'
    windows = []
    cur_tok, cur_tag, cur_am, cur_av = [], [], [], []
    for ln, b in lines:
        toks = tokenize_line(ln, b)
        if not toks:
            continue
        tags, amask, avecs = bio_tags(toks, entities)
        surf = [t[0] for t in toks]
        if cur_tok and len(cur_tok) + len(surf) > max_words:
            windows.append({"tokens": cur_tok, "ner_tags": cur_tag,
                            "assert_mask": cur_am, "assert_tags": cur_av})
            cur_tok, cur_tag, cur_am, cur_av = [], [], [], []
        cur_tok += surf
        cur_tag += tags
        cur_am += amask
        cur_av += avecs
    if cur_tok:
        windows.append({"tokens": cur_tok, "ner_tags": cur_tag,
                        "assert_mask": cur_am, "assert_tags": cur_av})
    return windows


def load_notes(d):
    d = Path(d)
    f = d / "train.jsonl"
    if not f.exists():
        raise FileNotFoundError(f)
    return [json.loads(l) for l in f.open(encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="thư mục data LLM (nguồn chính, để tách val/test)")
    ap.add_argument("--extra", default=None, help="thư mục data phụ (no-LLM), chỉ vào train")
    ap.add_argument("--out", default="training/dataset")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--max_words", type=int, default=110)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    llm = load_notes(args.llm)
    rng.shuffle(llm)
    n_test = int(len(llm) * args.test)
    n_val = int(len(llm) * args.val)
    test_notes = llm[:n_test]
    val_notes = llm[n_test:n_test + n_val]
    train_notes = llm[n_test + n_val:]
    if args.extra:
        train_notes = train_notes + load_notes(args.extra)
        rng.shuffle(train_notes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
    (out / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    (out / "assertions.txt").write_text("\n".join(ASSERTIONS), encoding="utf-8")

    def dump(notes, name):
        n_win = n_ent = 0
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for note in notes:
                for w in note_to_windows(note["text"], note["entities"], args.max_words):
                    if not any(t != "O" for t in w["ner_tags"]) and rng.random() < 0.5:
                        continue  # bớt cửa sổ toàn O (giữ 1 nửa cho ngữ cảnh âm)
                    f.write(json.dumps(w, ensure_ascii=False) + "\n")
                    n_win += 1
                    n_ent += sum(1 for t in w["ner_tags"] if t.startswith("B-"))
        print(f"  {name}: {len(notes)} note -> {n_win} cửa sổ, {n_ent} entity")

    print(f"labels={len(labels)} | split note: train={len(train_notes)} val={len(val_notes)} test={len(test_notes)}")
    dump(train_notes, "train")
    dump(val_notes, "validation")
    dump(test_notes, "test")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
