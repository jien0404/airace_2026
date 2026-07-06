#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thử nghiệm model NER: leduckhai/VietMed-NER (phobert-base-v2-VietMed-NER)
cho bước NER của bài toán Clinical Information Extraction (xem DE_BAI.md).

Model gốc dùng bộ 36 nhãn y tế (BIO). Ta:
  1. Chạy token-classification trên từng file input/*.txt.
  2. Xuất nhãn GỐC của VietMed (để đánh giá chất lượng span).
  3. Ánh xạ tạm sang 5 loại của cuộc thi để dễ nhìn (xem LABEL_MAP).

LƯU Ý QUAN TRỌNG:
  - VietMed gộp bệnh + triệu chứng vào 1 nhãn DISEASESYMTOM, trong khi đề tách
    TRIỆU_CHỨNG vs CHẨN_ĐOÁN. Đây đúng là chỗ "sai type = phạt kép". Bản map
    dưới chỉ để tham khảo, KHÔNG phải lời giải cuối.
  - Đề không chấm `position`, nhưng CHẤM `text` (WER) nên script cố lấy đúng
    span ký tự gốc.
  - Bước này chỉ làm NER: assertions/candidates để rỗng.

Script viết để chạy được cả với slow tokenizer (PhobertTokenizer) — không phụ
thuộc offset_mapping / fast tokenizer.
"""

import argparse
import json
import os
import sys
import glob
import time

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --------------------------------------------------------------------------
# Ánh xạ nhãn VietMed -> 5 loại của cuộc thi (chỉ tham khảo, dễ chỉnh).
# Đặt None để loại bỏ khỏi output theo format cuộc thi.
# --------------------------------------------------------------------------
LABEL_MAP = {
    "DRUGCHEMICAL":      "THUỐC",
    "DISEASESYMTOM":     "TRIỆU_CHỨNG",   # <-- NHẬP NHẰNG: có thể là CHẨN_ĐOÁN
    "DIAGNOSTICS":       "TÊN_XÉT_NGHIỆM",
    "UNITCALIBRATOR":    "KẾT_QUẢ_XÉT_NGHIỆM",
    # Các nhãn dưới đây không có trong 5 loại của đề -> bỏ:
    "ORGAN":             None,
    "TREATMENT":         None,
    "SURGERY":           None,
    "MEDDEVICETECHNIQUE": None,
    "PREVENTIVEMED":     None,
    "AGE":               None,
    "GENDER":            None,
    "DATETIME":          None,
    "LOCATION":          None,
    "FOODDRINK":         None,
    "ORGANIZATION":      None,
    "OCCUPATION":        None,
    "TRANSPORTATION":    None,
    "PERSONALCARE":      None,
}


def try_load_segmenter(vncorenlp_dir):
    """Nạp VnCoreNLP (RDRSegmenter) nếu có. Trả về callable(str)->str hoặc None.

    PhoBERT được train trên text ĐÃ tách từ (word segmentation), nên bật cái này
    thường tăng chất lượng cho phần tiếng Việt. Cần Java + py_vncorenlp.
    """
    try:
        import py_vncorenlp
    except ImportError:
        print("[seg] py_vncorenlp chưa cài -> chạy KHÔNG tách từ (raw whitespace).",
              file=sys.stderr)
        return None
    try:
        os.makedirs(vncorenlp_dir, exist_ok=True)
        # Tải model lần đầu nếu chưa có
        jar = os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")
        if not os.path.exists(jar):
            print(f"[seg] Đang tải VnCoreNLP về {vncorenlp_dir} ...", file=sys.stderr)
            py_vncorenlp.download_model(save_dir=vncorenlp_dir)
        seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)

        def _segment(text: str) -> str:
            # word_segment trả list câu đã nối bằng '_' cho từ ghép
            out = seg.word_segment(text)
            return " ".join(out) if isinstance(out, list) else str(out)

        print("[seg] VnCoreNLP sẵn sàng (word segmentation BẬT).", file=sys.stderr)
        return _segment
    except Exception as e:  # noqa
        print(f"[seg] Không khởi tạo được VnCoreNLP ({e}) -> chạy KHÔNG tách từ.",
              file=sys.stderr)
        return None


def build_words_with_spans(raw_text, segmenter):
    """Trả về list (surface_token, char_start, char_end) map về raw_text.

    Nếu có segmenter: token có thể chứa '_' (từ ghép). char span luôn tính trên
    raw_text gốc để `text` xuất ra khớp input.
    """
    if segmenter is not None:
        seg_text = segmenter(raw_text)
        tokens = seg_text.split()
    else:
        tokens = raw_text.split()

    spans = []
    cursor = 0
    for tok in tokens:
        surface = tok.replace("_", " ")  # bề mặt thật của từ ghép
        idx = raw_text.find(surface, cursor)
        if idx == -1:
            idx = raw_text.find(surface)  # fallback: tìm từ đầu
        if idx == -1:
            # segmentation làm biến dạng ký tự (hiếm) -> gán tạm tại cursor
            start, end = cursor, cursor + len(surface)
        else:
            start, end = idx, idx + len(surface)
            cursor = end
        spans.append((tok, start, end))
    return spans


def predict_word_labels(words_spans, tokenizer, model, id2label, device,
                        max_subtokens=250):
    """Gán 1 nhãn cho mỗi word (lấy nhãn của subtoken đầu tiên).

    Không dùng offset_mapping/word_ids để tương thích slow tokenizer.
    Tự đóng gói word vào các chunk <= max_subtokens rồi chạy model.
    """
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    # tiền tính subtoken ids cho từng word
    word_subtoks = []
    for tok, s, e in words_spans:
        ids = tokenizer.encode(tok.replace("_", " "), add_special_tokens=False)
        if len(ids) == 0:
            ids = [tokenizer.unk_token_id]
        word_subtoks.append(ids)

    labels = ["O"] * len(words_spans)

    # đóng gói chunk theo số subtoken
    i = 0
    n = len(words_spans)
    while i < n:
        chunk_word_idx = []
        chunk_ids = []
        # vị trí subtoken-đầu của mỗi word trong chunk (sau bos)
        first_pos = []
        while i < n and (len(chunk_ids) + len(word_subtoks[i])) <= max_subtokens:
            first_pos.append(len(chunk_ids))
            chunk_ids.extend(word_subtoks[i])
            chunk_word_idx.append(i)
            i += 1
        if not chunk_word_idx:
            # 1 word quá dài -> cắt bớt
            first_pos.append(0)
            chunk_ids = word_subtoks[i][:max_subtokens]
            chunk_word_idx.append(i)
            i += 1

        input_ids = [bos] + chunk_ids + [eos]
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits[0]
        preds = logits.argmax(-1).tolist()

        for w_local, w_global in enumerate(chunk_word_idx):
            # +1 vì có bos ở đầu
            pos = first_pos[w_local] + 1
            if pos < len(preds):
                labels[w_global] = id2label[preds[pos]]
    return labels


def bio_decode(words_spans, labels, raw_text):
    """Gộp word thành entity theo BIO. Trả list entity dict.

    entity = {text, position:[s,e], vietmed_type, score(None)}
    text lấy nguyên văn raw_text[s:e].
    """
    entities = []
    cur_type = None
    cur_start = None
    cur_end = None

    def close():
        nonlocal cur_type, cur_start, cur_end
        if cur_type is not None:
            text = raw_text[cur_start:cur_end]
            entities.append({
                "text": text,
                "position": [cur_start, cur_end],
                "vietmed_type": cur_type,
            })
        cur_type = cur_start = cur_end = None

    for (tok, s, e), lab in zip(words_spans, labels):
        if lab == "O" or lab == "0":
            close()
            continue
        prefix, _, etype = lab.partition("-")
        if etype == "":
            etype = prefix  # phòng nhãn lạ
        if prefix == "B":
            close()
            cur_type, cur_start, cur_end = etype, s, e
        elif prefix == "I":
            if cur_type == etype:
                cur_end = e
            else:
                # I- không có B- trước -> coi như bắt đầu mới (lenient)
                close()
                cur_type, cur_start, cur_end = etype, s, e
        else:
            close()
    close()
    return entities


def to_competition_format(entities):
    """Ánh xạ sang 5 loại + format object của đề (assertions/candidates rỗng)."""
    out = []
    for ent in entities:
        mapped = LABEL_MAP.get(ent["vietmed_type"], None)
        if mapped is None:
            continue
        out.append({
            "text": ent["text"],
            "position": ent["position"],
            "type": mapped,
            "assertions": [],
            "candidates": [],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=None,
                    help="Thư mục chứa *.txt (mặc định: ../../input)")
    ap.add_argument("--out_dir", default=None,
                    help="Thư mục xuất (mặc định: ./output)")
    ap.add_argument("--model", default="leduckhai/VietMed-NER")
    ap.add_argument("--subfolder", default="phobert-base-v2-VietMed-NER")
    ap.add_argument("--tokenizer", default="vinai/phobert-base-v2",
                    help="Tokenizer nguồn: subfolder của model KHÔNG có file "
                         "tokenizer, nên nạp từ base PhoBERT (mặc định).")
    ap.add_argument("--segment", action="store_true",
                    help="Bật word segmentation bằng VnCoreNLP (khuyến nghị nếu có Java)")
    ap.add_argument("--vncorenlp_dir", default="./vncorenlp")
    ap.add_argument("--device", default=None, help="cpu / cuda (mặc định: tự dò)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.abspath(os.path.join(here, "..", "..", "input"))
    out_dir = args.out_dir or os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] device={device}  model={args.model}/{args.subfolder}  "
          f"tokenizer={args.tokenizer}  segment={args.segment}", file=sys.stderr)

    # Subfolder của model chỉ có config.json + pytorch_model.bin (không có
    # vocab/bpe.codes) -> tokenizer nạp từ base PhoBERT.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.model, subfolder=args.subfolder)
    model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    segmenter = try_load_segmenter(args.vncorenlp_dir) if args.segment else None

    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                   if os.path.splitext(os.path.basename(p))[0].isdigit() else 1e9)
    print(f"[cfg] {len(files)} file trong {input_dir}", file=sys.stderr)

    t0 = time.time()
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, encoding="utf-8") as f:
            raw_text = f.read()

        words_spans = build_words_with_spans(raw_text, segmenter)
        labels = predict_word_labels(words_spans, tokenizer, model, id2label, device)
        entities = bio_decode(words_spans, labels, raw_text)
        comp = to_competition_format(entities)

        # 1) nhãn gốc VietMed (để soi chất lượng NER)
        with open(os.path.join(out_dir, f"{name}.raw.json"), "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        # 2) format cuộc thi (đã map 5 loại)
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(comp, f, ensure_ascii=False, indent=2)

        print(f"  {name}: {len(entities)} ent (raw) -> {len(comp)} ent (5-loại)",
              file=sys.stderr)

    dt = time.time() - t0
    print(f"[done] {len(files)} file trong {dt:.1f}s -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
