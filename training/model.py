# -*- coding: utf-8 -*-
"""Model multi-task: 1 encoder chung + 2 head.
  - head TYPE : token classification BIO (11 nhãn), CrossEntropy (ignore -100).
  - head ASSERT: 3 nhãn nhị phân độc lập (isNegated/isFamily/isHistorical) mức token,
    BCE, chỉ tính loss trên token entity thuộc loại có-assertion (assert_mask=1).

Lưu/nạp: dùng AutoModel (encoder HF) + 2 Linear; save state_dict + meta để predict dựng lại.
"""

import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MultiTaskNER(nn.Module):
    def __init__(self, base_model, num_ner, num_assert=3, dropout=0.1,
                 assert_weight=1.0, encoder=None):
        super().__init__()
        self.base_model = base_model
        self.num_ner = num_ner
        self.num_assert = num_assert
        self.assert_weight = assert_weight
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = AutoModel.from_pretrained(base_model)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(h, num_ner)
        self.assert_head = nn.Linear(h, num_assert)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids, attention_mask, labels=None,
                assert_labels=None, assert_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)          # [B,T,H]
        ner_logits = self.ner_head(seq)                    # [B,T,C]
        assert_logits = self.assert_head(seq)              # [B,T,3]

        loss = None
        if labels is not None:
            l_ner = self.ce(ner_logits.reshape(-1, self.num_ner), labels.reshape(-1))
            loss = l_ner
            if assert_labels is not None and assert_mask is not None:
                bce = self.bce(assert_logits, assert_labels.float())   # [B,T,3]
                m = assert_mask.unsqueeze(-1).float()                  # [B,T,1]
                denom = m.sum().clamp(min=1.0)
                l_ass = (bce * m).sum() / denom
                loss = l_ner + self.assert_weight * l_ass
        return {"loss": loss, "ner_logits": ner_logits, "assert_logits": assert_logits}

    # ---- lưu / nạp ----
    def save(self, out_dir, tokenizer=None, id2label=None, assertions=None):
        os.makedirs(out_dir, exist_ok=True)
        self.encoder.save_pretrained(out_dir)              # config + weights encoder
        torch.save({"ner_head": self.ner_head.state_dict(),
                    "assert_head": self.assert_head.state_dict()},
                   os.path.join(out_dir, "heads.pt"))
        meta = {"base_model": self.base_model, "num_ner": self.num_ner,
                "num_assert": self.num_assert,
                "id2label": id2label, "assertions": assertions}
        with open(os.path.join(out_dir, "mtl_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)

    @classmethod
    def load(cls, out_dir, device="cpu"):
        with open(os.path.join(out_dir, "mtl_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        encoder = AutoModel.from_pretrained(out_dir)
        model = cls(meta["base_model"], meta["num_ner"], meta["num_assert"], encoder=encoder)
        heads = torch.load(os.path.join(out_dir, "heads.pt"), map_location=device)
        model.ner_head.load_state_dict(heads["ner_head"])
        model.assert_head.load_state_dict(heads["assert_head"])
        model.to(device).eval()
        return model, meta
