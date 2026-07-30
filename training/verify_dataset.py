# -*- coding: utf-8 -*-
"""SÀNG LỌC SƠ + BÁO CÁO THỐNG KÊ cho một bộ nhãn / một nguồn dữ liệu.

⚠ Công cụ này KHÔNG tự loại bỏ gì và KHÔNG phán quyết. Nó chỉ:
  1. kiểm các điều kiện CƠ HỌC (thứ máy chắc chắn đúng/sai),
  2. liệt kê CHI TIẾT từng ca bị gắn cờ, kèm ngữ cảnh,
  3. in thống kê để người đọc quyết định.

Vì sao không tự lọc: lọc mạnh làm GIẢM ĐỘ ĐA DẠNG của dataset, và phần lớn các cờ ở đây là
báo động giả đã biết (span 'lệch ranh giới từ' thường chỉ là văn bản gốc thiếu dấu cách).
Quyết định giữ/bỏ là của người, không phải của script.

Nhóm cờ:
  [CƠ HỌC]   offset không cắt ra đúng `text` · span rỗng · offset ngoài phạm vi ·
             trùng (position,type) · type/assertion không hợp lệ
             -> gần như chắc chắn là lỗi, nhưng vẫn chỉ LIỆT KÊ
  [CẦN ĐỌC]  span lệch ranh giới từ · span vắt ngang xuống dòng · span chồng/lồng nhau ·
             assertion trên TÊN_XN/KQ_XN
             -> có thể đúng, có thể sai; phải đọc ngữ cảnh mới biết

    python -m training.verify_dataset --labels annotation/data/best_part3_f24/labels --notes input_turn2
    python -m training.verify_dataset --gen data_gen/generated/public_gold --out bao_cao.md
"""

import argparse
import collections
import glob
import json
import os
import re
import unicodedata

TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "THUỐC"}
ASSERT_OK = {"isNegated", "isFamily", "isHistorical"}
NO_ASSERT = {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}

MECH = "CƠ HỌC"
READ = "CẦN ĐỌC"


def token_bounds(text):
    starts, ends = set(), set()
    for m in TOK_RE.finditer(text):
        starts.add(m.start())
        ends.add(m.end())
    return starts, ends


def ctx(raw, a, b, pad=25):
    s = raw[max(0, a - pad):a].replace("\n", "⏎")
    m = raw[a:b].replace("\n", "⏎")
    e = raw[b:b + pad].replace("\n", "⏎")
    return f"…{s}⟦{m}⟧{e}…"


def scan(raw, ents, fid, flags):
    """Gắn cờ cho một bản ghi. Không sửa, không loại."""
    starts, ends = token_bounds(raw)
    seen = set()
    spans = []
    nfc = unicodedata.normalize("NFC", raw)

    for e in ents:
        t = e.get("text", "")
        ty = e.get("type")
        try:
            a, b = int(e["position"][0]), int(e["position"][1])
        except Exception:
            flags[(MECH, "position hỏng")].append((fid, "", str(e)[:80]))
            continue

        if b <= a:
            flags[(MECH, "span rỗng")].append((fid, f"[{a}:{b}]", repr(t)))
            continue
        if not (0 <= a < b <= len(raw)):
            flags[(MECH, "offset ngoài phạm vi")].append((fid, f"[{a}:{b}]", repr(t)))
            continue

        if raw[a:b] != t:
            tag = "offset lệch — khớp trên NFC ⇒ BUG NFC" if nfc[a:b] == t else "offset lệch"
            flags[(MECH, tag)].append((fid, f"[{a}:{b}]", f"text={t!r} nhưng raw[pos]={raw[a:b]!r}"))

        if ty not in TYPES:
            flags[(MECH, "type không hợp lệ")].append((fid, f"[{a}:{b}]", repr(ty)))
        for x in (e.get("assertions") or []):
            if x not in ASSERT_OK:
                flags[(MECH, "assertion không hợp lệ")].append((fid, f"[{a}:{b}]", repr(x)))

        k = (a, b, ty)
        if k in seen:
            flags[(MECH, "trùng (position, type)")].append((fid, f"[{a}:{b}]", f"{ty} {t!r}"))
        seen.add(k)

        if "\n" in raw[a:b]:
            flags[(READ, "span vắt ngang xuống dòng")].append((fid, f"[{a}:{b}]", ctx(raw, a, b)))
        if a not in starts or b not in ends:
            flags[(READ, "span lệch ranh giới từ")].append((fid, f"[{a}:{b}]", ctx(raw, a, b)))
        if (e.get("assertions") or []) and ty in NO_ASSERT:
            flags[(READ, "assertion trên TÊN_XN / KQ_XN")].append(
                (fid, f"[{a}:{b}]", f"{ty} {t!r} {e['assertions']}"))

        spans.append((a, b, ty, t))

    spans.sort()
    for i in range(1, len(spans)):
        pa, pb, pt, ptx = spans[i - 1]
        ca, cb, ct, ctx_ = spans[i]
        if ca < pb:
            tag = "span LỒNG nhau" if cb <= pb else "span CHỒNG lấn"
            flags[(READ, tag)].append((fid, f"[{pa}:{pb}] ∩ [{ca}:{cb}]",
                                       f"{pt} {ptx!r}  ×  {ct} {ctx_!r}"))
    return spans


# ---------- các cách nạp dữ liệu ----------

def load_gen(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "notes", "*.txt"))):
        fid = os.path.basename(p)[:-4]
        lp = os.path.join(d, "labels", fid + ".json")
        if not os.path.exists(lp):
            continue
        lab = json.load(open(lp, encoding="utf-8"))
        out[fid] = (open(p, encoding="utf-8").read(),
                    lab if isinstance(lab, list) else lab.get("entities", []))
    return out


def load_jsonl(d):
    out = {}
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p, encoding="utf-8")):
            r = json.loads(line)
            if "text" not in r:
                return None
            out[f"{name}:{r.get('id', i)}"] = (r["text"], r.get("entities", []))
    return out


def load_labels(labdir, notedir):
    out = {}
    for p in sorted(glob.glob(os.path.join(labdir, "*.json"))):
        fid = os.path.basename(p)[:-5]
        if fid.startswith("_"):
            continue
        np_ = os.path.join(notedir, fid + ".txt")
        if not os.path.exists(np_):
            continue
        lab = json.load(open(p, encoding="utf-8"))
        out[fid] = (open(np_, encoding="utf-8").read(),
                    lab if isinstance(lab, list) else lab.get("entities", []))
    return out


# ---------- thống kê ----------

def genre(raw):
    t = unicodedata.normalize("NFC", raw)
    q = bool(re.search(r"Câu hỏi từ người dùng|Câu trả lời của bác sĩ", t, re.I))
    s = bool(re.search(r"^\s*\d+\s*[.)]\s*(Tiền sử|Bệnh sử|Lịch sử|Đánh giá|Khám)", t, re.M))
    return "LAI" if (q and s) else "TƯ VẤN" if q else "BỆNH ÁN" if s else "KHÁC"


def stats_block(data):
    L = []
    n_ent = sum(len(v[1]) for v in data.values())
    by_type = collections.Counter()
    by_ass = collections.Counter()
    n_ass = 0
    wlen = collections.Counter()
    g = collections.defaultdict(lambda: [0, 0, 0])   # entity, ký tự, file
    for fid, (raw, ents) in data.items():
        k = genre(raw)
        g[k][0] += len(ents)
        g[k][1] += len(raw)
        g[k][2] += 1
        for e in ents:
            by_type[e.get("type")] += 1
            a = e.get("assertions") or []
            n_ass += bool(a)
            for x in a:
                by_ass[x] += 1
            wlen[min(len(TOK_RE.findall(e.get("text", ""))), 15)] += 1

    L.append(f"- **{len(data)} bản ghi · {n_ent} entity · "
             f"{n_ent / max(1, len(data)):.1f} entity/bản ghi**")
    L.append("")
    L.append("### Mật độ theo thể loại")
    L.append("")
    L.append("| thể loại | file | entity | ký tự | **entity/1000 ký tự** |")
    L.append("|---|--:|--:|--:|--:|")
    for k in ("BỆNH ÁN", "LAI", "TƯ VẤN", "KHÁC"):
        if k not in g:
            continue
        ent, ch, nf = g[k]
        L.append(f"| {k} | {nf} | {ent} | {ch} | **{ent / max(1, ch) * 1000:.2f}** |")
    tot_e = sum(v[0] for v in g.values())
    tot_c = sum(v[1] for v in g.values())
    L.append(f"| **TOÀN BỘ** | {len(data)} | {tot_e} | {tot_c} | **{tot_e / max(1, tot_c) * 1000:.2f}** |")
    L.append("")
    L.append("> Mốc đối chiếu đo trên V66ab: BỆNH ÁN 15.4 · LAI 10.2 · TƯ VẤN 6.6 · "
             "corpus 13.3 (`business_rules/CANONICAL.md` §6)")
    L.append("")
    L.append("### Phân bố type")
    L.append("")
    L.append("| type | n | % |")
    L.append("|---|--:|--:|")
    for k, v in by_type.most_common():
        L.append(f"| {k} | {v} | {100 * v / max(1, n_ent):.1f}% |")
    L.append("")
    L.append(f"### Assertion — {n_ass}/{n_ent} entity ({100 * n_ass / max(1, n_ent):.1f}%)")
    L.append("")
    L.append("| assertion | n |")
    L.append("|---|--:|")
    for k, v in by_ass.most_common():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("### Độ dài span (số token)")
    L.append("")
    L.append("| token | " + " | ".join(str(i) if i < 15 else "15+" for i in sorted(wlen)) + " |")
    L.append("|---" * (len(wlen) + 1) + "|")
    L.append("| n | " + " | ".join(str(wlen[i]) for i in sorted(wlen)) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", help="thư mục data_gen (notes/ + labels/)")
    ap.add_argument("--jsonl", help="thư mục dataset có train.jsonl kèm text")
    ap.add_argument("--labels")
    ap.add_argument("--notes")
    ap.add_argument("--out", help="ghi báo cáo markdown ra file")
    ap.add_argument("--full", action="store_true", help="liệt kê ĐẦY ĐỦ, không cắt bớt")
    ap.add_argument("--show", type=int, default=25, help="số ca in ra mỗi nhóm cờ (mặc định 25)")
    a = ap.parse_args()

    if a.gen:
        data, src = load_gen(a.gen), a.gen
    elif a.jsonl:
        data, src = load_jsonl(a.jsonl), a.jsonl
        if data is None:
            raise SystemExit("Dataset đã window-hoá (không có trường `text`) — "
                             "kiểm ở nguồn train.jsonl gốc trong data_gen/generated/.")
    elif a.labels and a.notes:
        data, src = load_labels(a.labels, a.notes), a.labels
    else:
        raise SystemExit("cần --gen | --jsonl | (--labels và --notes)")
    if not data:
        raise SystemExit(f"Không đọc được bản ghi nào từ {src}")

    flags = collections.defaultdict(list)
    for fid, (raw, ents) in data.items():
        scan(raw, ents, fid, flags)

    L = [f"# Báo cáo sàng lọc — `{src}`", "",
         "> Máy chỉ GẮN CỜ và liệt kê. Không loại bỏ gì. Quyết định giữ/bỏ là của người đọc.", ""]
    L += stats_block(data)
    L += ["", "---", "", "## Cờ đã gắn", ""]

    n_mech = sum(len(v) for (grp, _), v in flags.items() if grp == MECH)
    n_read = sum(len(v) for (grp, _), v in flags.items() if grp == READ)
    L.append(f"- **[{MECH}]** {n_mech} ca — máy gần như chắc chắn đúng/sai")
    L.append(f"- **[{READ}]** {n_read} ca — cần đọc ngữ cảnh mới quyết được")
    L.append("")

    if not flags:
        L.append("Không có cờ nào.")
    for grp in (MECH, READ):
        items = [(k, v) for (g_, k), v in flags.items() if g_ == grp]
        if not items:
            continue
        L.append(f"### [{grp}]")
        L.append("")
        for k, v in sorted(items, key=lambda x: -len(x[1])):
            L.append(f"#### {k} — {len(v)} ca")
            L.append("")
            L.append("| file | vị trí | chi tiết |")
            L.append("|---|---|---|")
            show = v if a.full else v[:a.show]
            for fid, pos, det in show:
                det = det.replace("|", "\\|")
                L.append(f"| {fid} | `{pos}` | {det} |")
            if len(v) > len(show):
                L.append(f"| … | | *còn {len(v) - len(show)} ca — chạy lại với `--full`* |")
            L.append("")

    L += ["", "---", "",
          "## Ghi chú để đọc cờ",
          "",
          "- **offset lệch — khớp trên NFC**: đúng bug NFC. Sửa bằng remap NFC→raw "
          "**KHÔNG ĐIỀU KIỆN** (remap chỉ những cái thấy lệch là sai — từng sót 41 entity).",
          "- **span lệch ranh giới từ**: có thể là span sai (`'ho'` nằm trong `chuyên khoa`), "
          "cũng có thể là **văn bản gốc thiếu dấu cách** (`thuốcVastarel`, `Bệnh dạithường`) — "
          "khi đó span vẫn ĐÚNG. Đã đo: trong một lô 26 ca, 7 là lỗi thật và 19 là báo động giả.",
          "- **span vắt ngang xuống dòng**: nhãn có thể đúng về nghiệp vụ, nhưng sơ đồ BIO sẽ "
          "decode nó thành HAI entity ⇒ ảnh hưởng dữ liệu train, không ảnh hưởng bài nộp.",
          "- **span chồng/lồng nhau**: BIO không biểu diễn được — khi build dataset, cái thua bị "
          "vứt âm thầm. Cần biết để không ngạc nhiên khi số entity giảm.",
          "- **assertion trên TÊN_XN / KQ_XN**: đề bài loại trừ thẳng; `gt2` của BTC lại có 322 ca "
          "như vậy. Trên bài nộp thì vô hại (đã đo), nhưng không nên dạy model.",
          ]

    out = "\n".join(L)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"Đã ghi báo cáo: {a.out}")
        print(f"  {len(data)} bản ghi · [{MECH}] {n_mech} cờ · [{READ}] {n_read} cờ")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
