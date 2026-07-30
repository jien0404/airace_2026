from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel


class HybridNER(nn.Module):
    """Shared encoder + BIO head + span type head + entity-level assertion head."""

    def __init__(
        self,
        base_model: str,
        num_bio: int,
        num_types: int = 5,
        num_assertions: int = 3,
        dropout: float = 0.1,
        span_weight: float = 0.5,
        assertion_weight: float = 1.0,
        encoder=None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_bio = num_bio
        self.num_types = num_types
        self.num_assertions = num_assertions
        self.span_weight = span_weight
        self.assertion_weight = assertion_weight
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.bio_head = nn.Linear(hidden, num_bio)
        self.span_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.span_type_head = nn.Linear(hidden, num_types + 1)
        self.assertion_head = nn.Linear(hidden, num_assertions)

    def span_repr(self, sequence, span_starts, span_ends):
        batch, _, hidden = sequence.shape
        safe_starts = span_starts.clamp(min=0)
        safe_ends = span_ends.clamp(min=0)
        gather_start = safe_starts.unsqueeze(-1).expand(-1, -1, hidden)
        gather_end = safe_ends.unsqueeze(-1).expand(-1, -1, hidden)
        start_repr = torch.gather(sequence, 1, gather_start)
        end_repr = torch.gather(sequence, 1, gather_end)
        return self.span_projection(torch.cat(
            [start_repr, end_repr, start_repr * end_repr], dim=-1
        ))

    def forward(
        self,
        input_ids,
        attention_mask,
        bio_labels=None,
        token_loss_mask=None,
        span_starts=None,
        span_ends=None,
        span_labels=None,
        assertion_labels=None,
        assertion_mask=None,
    ):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = self.dropout(encoded.last_hidden_state)
        bio_logits = self.bio_head(sequence)
        result = {"bio_logits": bio_logits, "loss": None}
        losses = []
        if bio_labels is not None:
            raw = nn.functional.cross_entropy(
                bio_logits.reshape(-1, self.num_bio),
                bio_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(bio_labels)
            mask = (bio_labels != -100).float()
            if token_loss_mask is not None:
                mask = mask * token_loss_mask.float()
            losses.append((raw * mask).sum() / mask.sum().clamp(min=1.0))
        if span_starts is not None and span_starts.shape[1] > 0:
            representation = self.span_repr(sequence, span_starts, span_ends)
            span_logits = self.span_type_head(representation)
            assertion_logits = self.assertion_head(representation)
            result.update(span_logits=span_logits, assertion_logits=assertion_logits)
            if span_labels is not None:
                span_loss = nn.functional.cross_entropy(
                    span_logits.reshape(-1, self.num_types + 1),
                    span_labels.reshape(-1),
                    ignore_index=-100,
                )
                losses.append(self.span_weight * span_loss)
            if assertion_labels is not None and assertion_mask is not None:
                raw = nn.functional.binary_cross_entropy_with_logits(
                    assertion_logits, assertion_labels.float(), reduction="none"
                )
                mask = assertion_mask.unsqueeze(-1).float()
                assertion_loss = (raw * mask).sum() / mask.sum().clamp(min=1.0)
                losses.append(self.assertion_weight * assertion_loss)
        if losses:
            result["loss"] = sum(losses)
        return result

    def save(self, out_dir, tokenizer, metadata):
        os.makedirs(out_dir, exist_ok=True)
        self.encoder.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        torch.save({
            "bio_head": self.bio_head.state_dict(),
            "span_projection": self.span_projection.state_dict(),
            "span_type_head": self.span_type_head.state_dict(),
            "assertion_head": self.assertion_head.state_dict(),
        }, os.path.join(out_dir, "hybrid_heads.pt"))
        meta = {
            **metadata,
            "base_model": self.base_model,
            "num_bio": self.num_bio,
            "num_types": self.num_types,
            "num_assertions": self.num_assertions,
            "span_weight": self.span_weight,
            "assertion_weight": self.assertion_weight,
        }
        with open(os.path.join(out_dir, "hybrid_meta.json"), "w", encoding="utf-8") as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, out_dir, device="cpu"):
        with open(os.path.join(out_dir, "hybrid_meta.json"), encoding="utf-8") as stream:
            meta = json.load(stream)
        encoder = AutoModel.from_pretrained(out_dir)
        model = cls(
            meta["base_model"],
            meta["num_bio"],
            meta["num_types"],
            meta["num_assertions"],
            span_weight=meta["span_weight"],
            assertion_weight=meta["assertion_weight"],
            encoder=encoder,
        )
        heads = torch.load(os.path.join(out_dir, "hybrid_heads.pt"), map_location=device)
        model.bio_head.load_state_dict(heads["bio_head"])
        model.span_projection.load_state_dict(heads["span_projection"])
        model.span_type_head.load_state_dict(heads["span_type_head"])
        model.assertion_head.load_state_dict(heads["assertion_head"])
        return model.to(device).eval(), meta
