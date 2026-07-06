# -*- coding: utf-8 -*-
"""Fine-tune token classification (NER 5 loại) — config-driven, chạy trên server GPU.

Hỗ trợ CẢ tokenizer slow (PhoBERT/ViHealthBERT) lẫn fast (XLM-R) nhờ align subword THỦ
CÔNG (encode từng token, nhãn gán vào subword ĐẦU, còn lại -100) — không phụ thuộc
offset_mapping/word_ids.

Ví dụ:
  python -m training.train_ner --data_dir training/dataset \
      --model demdecuong/vihealthbert-base-syllable --out runs/vihealthbert --fp16
  python -m training.train_ner --data_dir training/dataset \
      --model xlm-roberta-base --out runs/xlmr --fp16
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def load_labels(data_dir):
    labels = Path(data_dir, "labels.txt").read_text(encoding="utf-8").split("\n")
    labels = [l for l in labels if l]
    return labels, {l: i for i, l in enumerate(labels)}, {i: l for i, l in enumerate(labels)}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8")]


class NerDataset(Dataset):
    """Encode thủ công: mỗi token -> subwords; nhãn ở subword đầu, -100 phần còn lại."""

    def __init__(self, rows, tokenizer, label2id, max_len=256):
        self.rows = rows
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.unk = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else self.pad

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        toks = self.rows[i]["tokens"]
        tags = self.rows[i]["ner_tags"]
        ids, labs = [], []
        for tok, tag in zip(toks, tags):
            sub = self.tok.encode(tok, add_special_tokens=False)
            if not sub:
                sub = [self.unk]
            ids.extend(sub)
            labs.extend([self.label2id[tag]] + [-100] * (len(sub) - 1))
            if len(ids) >= self.max_len - 2:
                break
        ids = ids[: self.max_len - 2]
        labs = labs[: self.max_len - 2]
        input_ids = [self.bos] + ids + [self.eos]
        labels = [-100] + labs + [-100]
        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            n = m - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad] * n)
            out["attention_mask"].append(b["attention_mask"] + [0] * n)
            out["labels"].append(b["labels"] + [-100] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def build_metrics(id2label):
    def compute(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        true_tags, pred_tags = [], []
        for pr, la in zip(preds, labels):
            t, q = [], []
            for pi, li in zip(pr, la):
                if li == -100:
                    continue
                t.append(id2label[int(li)])
                q.append(id2label[int(pi)])
            true_tags.append(t)
            pred_tags.append(q)
        return {
            "precision": precision_score(true_tags, pred_tags),
            "recall": recall_score(true_tags, pred_tags),
            "f1": f1_score(true_tags, pred_tags),
        }
    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="mặc định = --model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels, label2id, id2label = load_labels(args.data_dir)
    tok_name = args.tokenizer or args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=False)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)

    train_ds = NerDataset(read_jsonl(Path(args.data_dir) / "train.jsonl"), tokenizer, label2id, args.max_len)
    val_ds = NerDataset(read_jsonl(Path(args.data_dir) / "validation.jsonl"), tokenizer, label2id, args.max_len)
    test_ds = NerDataset(read_jsonl(Path(args.data_dir) / "test.jsonl"), tokenizer, label2id, args.max_len)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=args.fp16,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(train_ds.pad),
        compute_metrics=build_metrics(id2label),
    )
    trainer.train()

    print("\n=== TEST ===")
    pred = trainer.predict(test_ds)
    print(pred.metrics)
    # báo cáo chi tiết theo loại
    logits, labs = pred.predictions, pred.label_ids
    preds = np.argmax(logits, axis=-1)
    tt, pt = [], []
    for pr, la in zip(preds, labs):
        a, b = [], []
        for pi, li in zip(pr, la):
            if li == -100:
                continue
            a.append(id2label[int(li)]); b.append(id2label[int(pi)])
        tt.append(a); pt.append(b)
    print(classification_report(tt, pt, digits=4))

    trainer.save_model(os.path.join(args.out, "best"))
    tokenizer.save_pretrained(os.path.join(args.out, "best"))
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(pred.metrics, f, ensure_ascii=False, indent=2)
    print(f"[done] model -> {args.out}/best")


if __name__ == "__main__":
    main()
