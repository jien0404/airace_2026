# -*- coding: utf-8 -*-
"""Windowing SECTION-AWARE dùng CHUNG cho build_ner_dataset (train) và predict (inference)
để train/infer NHẤT QUÁN.

Chiến lược (data có cấu trúc mục "1." "2." "3."):
- Cắt window ở MỖI header mục lớn -> window KHÔNG bao giờ vắt ngang 2 mục (cue historical/
  family/type sạch theo mục).
- Trong 1 mục, gói NGUYÊN DÒNG tới <= max_words (không xé entity vì entity nằm gọn trong dòng).
- Mục dài > max_words tách nhiều window; window ĐẦU chứa header trong 'content', các window
  NỐI TIẾP mang header vào 'prefix' = NGỮ CẢNH (đưa vào model nhưng KHÔNG gán nhãn / KHÔNG decode).

Mỗi window = {"content": [(tok,s,e)...], "prefix": [(tok,s,e)...]}.
- content: token cần gán nhãn (train) / cần decode ra entity (infer). Gộp content mọi window
  theo thứ tự = ĐÚNG chuỗi token của note (mỗi token content xuất hiện đúng 1 lần -> offset chuẩn).
- prefix: chỉ là bản sao header mục để làm ngữ cảnh; không sinh nhãn/entity.
"""

import re
import unicodedata

TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TOP_RE = re.compile(r"^\s*\d+\s*[.)]")     # dòng mở mục: "1." "2)" ...


def _norm(s):
    return unicodedata.normalize("NFC", s)


def _line_tokens(line, base):
    return [(m.group(), base + m.start(), base + m.end()) for m in TOK_RE.finditer(line)]


def make_windows(text, max_words=120):
    text = _norm(text)
    # tách dòng + offset toàn cục
    lines, base = [], 0
    for ln in text.split("\n"):
        lines.append((ln, base))
        base += len(ln) + 1

    # nhóm dòng theo mục (mỗi mục bắt đầu ở 1 header)
    sections, cur = [], []
    for ln, b in lines:
        if _TOP_RE.match(ln) and cur:
            sections.append(cur); cur = []
        cur.append((ln, b))
    if cur:
        sections.append(cur)

    windows = []
    for sec in sections:
        first_ln, first_b = sec[0]
        header_tokens = _line_tokens(first_ln, first_b) if _TOP_RE.match(first_ln) else []
        cur_toks, first_win = [], True
        for ln, b in sec:
            lt = _line_tokens(ln, b)
            if not lt:
                continue
            if cur_toks and len(cur_toks) + len(lt) > max_words:
                windows.append({"content": cur_toks,
                                "prefix": [] if first_win else header_tokens})
                cur_toks, first_win = [], False
            cur_toks += lt
        if cur_toks:
            windows.append({"content": cur_toks,
                            "prefix": [] if first_win else header_tokens})
    return windows


def full_tokens(text):
    """Chuỗi token toàn note (để đối chiếu tính partition)."""
    text = _norm(text)
    return [(m.group(), m.start(), m.end()) for m in TOK_RE.finditer(text)]


if __name__ == "__main__":
    # self-test: gộp content mọi window == full tokenization (partition đúng -> offset chuẩn)
    import glob
    bad = 0
    for nf in glob.glob("data_gen/generated/public_gold/notes/*.txt")[:50]:
        text = open(nf, encoding="utf-8").read()
        ft = full_tokens(text)
        merged = [t for w in make_windows(text) for t in w["content"]]
        if merged != ft:
            bad += 1
            print("MISMATCH", nf, len(ft), len(merged))
    print(f"partition OK trên {50-bad}/50 note (content gộp == full tokens)")
