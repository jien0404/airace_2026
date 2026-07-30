# -*- coding: utf-8 -*-
"""Windowing SECTION-AWARE dùng CHUNG cho build_ner_dataset (train) và predict (inference)
để train/infer NHẤT QUÁN.

Chiến lược:
- Cắt window ở MỖI header mục lớn -> window KHÔNG bao giờ vắt ngang 2 mục (cue historical/
  family/type sạch theo mục).
- Trong 1 mục, gói NGUYÊN DÒNG tới <= max_words (không xé entity vì entity nằm gọn trong dòng).
- Mục dài > max_words tách nhiều window; window ĐẦU chứa header trong 'content', các window
  NỐI TIẾP mang header vào 'prefix' = NGỮ CẢNH (đưa vào model nhưng KHÔNG gán nhãn / KHÔNG decode).
- ⭐ Dòng DÀI HƠN max_words được cắt tiếp theo RANH GIỚI CÂU (rồi dấu phẩy, rồi cắt cứng) để
  KHÔNG window nào vượt max_words.

  Vì sao cần: bản cũ không bao giờ xé một dòng, nên một dòng 543 token sinh ra một window 543
  token. Lúc train, `MtlDataset.__getitem__` cắt ở `max_len-2` subword và VỨT phần đuôi cùng
  toàn bộ nhãn của nó; lúc infer thì `predict.py` lại đóng gói lại thành nhiều chunk và decode
  hết. Bất đối xứng đó rơi đúng vào nhóm file văn xuôi dài (`input_turn2` có 48 dòng > 110 token,
  dài nhất 543) và vào ~12% window của các dataset gốc gt2.

Mỗi window = {"content": [(tok,s,e)...], "prefix": [(tok,s,e)...]}.
- content: token cần gán nhãn (train) / cần decode ra entity (infer). Gộp content mọi window
  theo thứ tự = ĐÚNG chuỗi token của note (mỗi token content xuất hiện đúng 1 lần -> offset chuẩn).
- prefix: chỉ là bản sao header mục để làm ngữ cảnh; không sinh nhãn/entity.
"""

import re
import unicodedata

TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TOP_RE = re.compile(r"^\s*\d+\s*[.)]")     # dòng mở mục: "1." "2)" ...
_SENT_END = {".", "!", "?", ";", "…"}
_SOFT_END = {",", ":", "-", "–", "—"}


def _norm(s):
    return unicodedata.normalize("NFC", s)


def _line_tokens(line, base):
    return [(m.group(), base + m.start(), base + m.end()) for m in TOK_RE.finditer(line)]


def _pack(segs, max_words):
    """Gói các đoạn liền kề thành đơn vị <= max_words (đoạn dài hơn max_words giữ nguyên)."""
    out, cur = [], []
    for s in segs:
        if cur and len(cur) + len(s) > max_words:
            out.append(cur)
            cur = []
        cur += s
    if cur:
        out.append(cur)
    return out


def _split_at(toks, marks):
    """Cắt chuỗi token sau mỗi token thuộc `marks`. Giữ dấu ở cuối đoạn trước."""
    segs, cur = [], []
    for t in toks:
        cur.append(t)
        if t[0] in marks:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def _units(line_toks, max_words):
    """Chia token của MỘT dòng thành các đơn vị, mỗi đơn vị <= max_words nếu có thể.

    Thứ tự ưu tiên chỗ cắt: ranh giới CÂU -> dấu phẩy/hai chấm -> cắt cứng.
    Cắt ở dấu câu nên rủi ro xé một entity là rất thấp (entity dài nhất trong gold < 30 token),
    và dù sao vẫn tốt hơn hẳn bản cũ vốn VỨT THẲNG phần đuôi lúc train.
    """
    if len(line_toks) <= max_words:
        return [line_toks]

    units = _pack(_split_at(line_toks, _SENT_END), max_words)

    # đơn vị nào vẫn quá dài -> cắt tiếp ở dấu phẩy/hai chấm
    step2 = []
    for u in units:
        step2 += _pack(_split_at(u, _SOFT_END), max_words) if len(u) > max_words else [u]

    # vẫn quá dài (một câu liền không dấu) -> cắt cứng
    out = []
    for u in step2:
        if len(u) <= max_words:
            out.append(u)
        else:
            out += [u[i:i + max_words] for i in range(0, len(u), max_words)]
    return out


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
            sections.append(cur)
            cur = []
        cur.append((ln, b))
    if cur:
        sections.append(cur)

    windows = []
    for sec in sections:
        first_ln, first_b = sec[0]
        header_tokens = _line_tokens(first_ln, first_b) if _TOP_RE.match(first_ln) else []
        if len(header_tokens) > max_words:          # header bất thường: đừng dùng làm prefix
            header_tokens = []
        cur_toks, first_win = [], True
        for ln, b in sec:
            lt = _line_tokens(ln, b)
            if not lt:
                continue
            for unit in _units(lt, max_words):
                if cur_toks and len(cur_toks) + len(unit) > max_words:
                    windows.append({"content": cur_toks,
                                    "prefix": [] if first_win else header_tokens})
                    cur_toks, first_win = [], False
                cur_toks += unit
        if cur_toks:
            windows.append({"content": cur_toks,
                            "prefix": [] if first_win else header_tokens})
    return windows


def full_tokens(text):
    """Chuỗi token toàn note (để đối chiếu tính partition)."""
    text = _norm(text)
    return [(m.group(), m.start(), m.end()) for m in TOK_RE.finditer(text)]


if __name__ == "__main__":
    # self-test: (1) gộp content mọi window == full tokenization (partition đúng -> offset chuẩn)
    #            (2) KHÔNG window nào vượt max_words
    import glob
    import sys

    max_words = int(sys.argv[1]) if len(sys.argv) > 1 else 110
    dirs = ["data_gen/generated/public_gold/notes/*.txt", "input_turn2/*.txt", "input/*.txt"]
    allok = True
    for pat in dirs:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"(bỏ qua, không có file: {pat})")
            continue
        bad_part, bad_size, mx, nwin = 0, 0, 0, 0
        for nf in files:
            text = open(nf, encoding="utf-8").read()
            ws = make_windows(text, max_words)
            nwin += len(ws)
            if [t for w in ws for t in w["content"]] != full_tokens(text):
                bad_part += 1
                print("  MISMATCH partition:", nf)
            for w in ws:
                mx = max(mx, len(w["content"]))
                if len(w["content"]) > max_words:
                    bad_size += 1
        ok = (bad_part == 0 and bad_size == 0)
        allok &= ok
        print(f"{'OK ' if ok else 'LỖI'} {pat:<45} {len(files):>4} file · {nwin:>5} window · "
              f"max {mx:>4} token · partition sai {bad_part} · quá khổ {bad_size}")
    print("\n=> " + ("TẤT CẢ ĐẠT" if allok else "CÓ LỖI"))
