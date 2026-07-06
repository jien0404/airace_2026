# -*- coding: utf-8 -*-
"""Train multi-task NER (type BIO) + assertion (3 nhãn nhị phân) — 1 model, 1 lần train.

Vòng train thủ công (không dùng Trainer) để kiểm soát 2 loss + 2 head. Align subword thủ
công nên chạy cả tokenizer slow (ViHealthBERT) lẫn fast (XLM-R).

  python -m training.train_mtl --data_dir training/dataset \
      --model demdecuong/vihealthbert-base-syllable --out runs/vihealthbert --fp16
  python -m training.train_mtl --data_dir training/dataset \
      --model xlm-roberta-base --out runs/xlmr --fp16
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report

from .model import MultiTaskNER


def load_labels(d):
    labels = [l for l in Path(d, "labels.txt").read_text(encoding="utf-8").split("\n") if l]
    asserts = [l for l in Path(d, "assertions.txt").read_text(encoding="utf-8").split("\n") if l]
    return labels, {l: i for i, l in enumerate(labels)}, {i: l for i, l in enumerate(labels)}, asserts


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8")]


class MtlDataset(Dataset):
    def __init__(self, rows, tokenizer, label2id, max_len=256):
        self.rows, self.tok, self.l2i, self.max_len = rows, tokenizer, label2id, max_len
        self.bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.unk = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else self.pad

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        toks, tags = r["tokens"], r["ner_tags"]
        am, av = r["assert_mask"], r["assert_tags"]
        ids, labs, amask, avecs = [], [], [], []
        for tok, tag, m, v in zip(toks, tags, am, av):
            sub = self.tok.encode(tok, add_special_tokens=False) or [self.unk]
            ids.extend(sub)
            labs.extend([self.l2i[tag]] + [-100] * (len(sub) - 1))
            amask.extend([m] + [0] * (len(sub) - 1))
            avecs.extend([v] + [[0, 0, 0]] * (len(sub) - 1))
            if len(ids) >= self.max_len - 2:
                break
        ids = ids[: self.max_len - 2]; labs = labs[: self.max_len - 2]
        amask = amask[: self.max_len - 2]; avecs = avecs[: self.max_len - 2]
        return {"input_ids": [self.bos] + ids + [self.eos],
                "labels": [-100] + labs + [-100],
                "assert_mask": [0] + amask + [0],
                "assert_tags": [[0, 0, 0]] + avecs + [[0, 0, 0]]}


def collate(batch, pad):
    m = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "assert_mask": [], "assert_labels": []}
    for b in batch:
        n = m - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad] * n)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * n)
        out["labels"].append(b["labels"] + [-100] * n)
        out["assert_mask"].append(b["assert_mask"] + [0] * n)
        out["assert_labels"].append(b["assert_tags"] + [[0, 0, 0]] * n)
    return {
        "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(out["labels"], dtype=torch.long),
        "assert_mask": torch.tensor(out["assert_mask"], dtype=torch.long),
        "assert_labels": torch.tensor(out["assert_labels"], dtype=torch.long),
    }


@torch.no_grad()
def evaluate(model, loader, id2label, device):
    model.eval()
    tt, pt = [], []
    a_tp = a_fp = a_fn = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"])
        ner = out["ner_logits"].argmax(-1).cpu().numpy()
        lab = batch["labels"].cpu().numpy()
        for pr, la in zip(ner, lab):
            a, b = [], []
            for pi, li in zip(pr, la):
                if li == -100:
                    continue
                a.append(id2label[int(li)]); b.append(id2label[int(pi)])
            tt.append(a); pt.append(b)
        # assertion micro-F1 trên token có mask
        ap = (torch.sigmoid(out["assert_logits"]) > 0.5).long()
        am = batch["assert_mask"].unsqueeze(-1)
        gt = batch["assert_labels"] * am
        pp = ap * am
        a_tp += int(((pp == 1) & (gt == 1)).sum())
        a_fp += int(((pp == 1) & (gt == 0)).sum())
        a_fn += int(((pp == 0) & (gt == 1)).sum())
    ner_f1 = f1_score(tt, pt)
    a_prec = a_tp / max(1, a_tp + a_fp)
    a_rec = a_tp / max(1, a_tp + a_fn)
    a_f1 = 2 * a_prec * a_rec / max(1e-9, a_prec + a_rec)
    return ner_f1, a_f1, (tt, pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--assert_weight", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels, l2i, i2l, asserts = load_labels(args.data_dir)

    tok_name = args.tokenizer or args.model
    try:
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=False)

    tr = MtlDataset(read_jsonl(Path(args.data_dir) / "train.jsonl"), tok, l2i, args.max_len)
    va = MtlDataset(read_jsonl(Path(args.data_dir) / "validation.jsonl"), tok, l2i, args.max_len)
    te = MtlDataset(read_jsonl(Path(args.data_dir) / "test.jsonl"), tok, l2i, args.max_len)
    cf = lambda b: collate(b, tr.pad)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=cf)
    vl = DataLoader(va, batch_size=args.eval_bs, shuffle=False, collate_fn=cf)
    tel = DataLoader(te, batch_size=args.eval_bs, shuffle=False, collate_fn=cf)

    model = MultiTaskNER(args.model, len(labels), len(asserts),
                         assert_weight=args.assert_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = int(len(tl) * args.epochs)
    sch = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    best_f1, step = -1.0, 0
    os.makedirs(args.out, exist_ok=True)
    n_epochs = int(np.ceil(args.epochs))
    for ep in range(n_epochs):
        model.train()
        for batch in tl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.fp16):
                out = model(**batch)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            step += 1
            if step % 50 == 0:
                print(f"ep{ep} step{step} loss {loss.item():.4f}", flush=True)
        ner_f1, a_f1, _ = evaluate(model, vl, i2l, device)
        print(f"[val] epoch {ep}: NER_F1={ner_f1:.4f}  ASSERT_F1={a_f1:.4f}", flush=True)
        if ner_f1 > best_f1:
            best_f1 = ner_f1
            model.save(os.path.join(args.out, "best"), tokenizer=tok,
                       id2label=i2l, assertions=asserts)
            print(f"  -> lưu best (NER_F1={ner_f1:.4f})", flush=True)

    # test bằng model tốt nhất
    best, _ = MultiTaskNER.load(os.path.join(args.out, "best"), device=device)
    ner_f1, a_f1, (tt, pt) = evaluate(best, tel, i2l, device)
    print("\n=== TEST ===")
    print(f"NER_F1={ner_f1:.4f}  ASSERT_F1={a_f1:.4f}")
    print(classification_report(tt, pt, digits=4))
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump({"ner_f1": ner_f1, "assert_f1": a_f1}, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {args.out}/best")


if __name__ == "__main__":
    main()
