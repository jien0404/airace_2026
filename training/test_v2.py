import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from .build_dataset_v2 import record_to_windows
from .audit_token_budget import audit_rows
from .model_v2 import HybridNER
from .train_assertion_head import _sampler_weights
from .train_v2 import targeted_sampler_weights
from .windowing_v2 import coverage_errors, sliding_windows, tokenize_raw


class DummyEncoder(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embedding = nn.Embedding(100, hidden)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyTokenizer:
    unk_token_id = 99
    pad_token_id = 0

    def encode(self, token, add_special_tokens=False):
        return [1, 2, 3] if token == "long" else [1]


class TrainingV2Test(unittest.TestCase):
    def test_targeted_sampler_mass_is_explicit(self):
        rows = [
            {"record_id": "hist:a"}, {"record_id": "hist:b"},
            {"record_id": "base:a"}, {"record_id": "base:b"}, {"record_id": "base:c"},
        ]
        weights = _sampler_weights(rows, "hist:", 0.10)
        self.assertAlmostEqual(sum(weights[:2]), 0.10)
        self.assertAlmostEqual(sum(weights[2:]), 0.90)

    def test_targeted_sampler_control_and_multiple_profiles(self):
        rows = [
            {"record_id": "patch:type.1", "sample_weight": 2.0},
            {"record_id": "patch:boundary.1", "sample_weight": 1.0},
            {"record_id": "patch:fp.1", "sample_weight": 1.0},
            {"record_id": "base:a", "sample_weight": 3.0},
            {"record_id": "base:b", "sample_weight": 1.0},
        ]
        weights, report = targeted_sampler_weights(
            rows,
            regexes=(r"patch:(?:type|boundary)",),
            targeted_mass=0.0,
        )
        self.assertEqual(weights[:2], [0.0, 0.0])
        self.assertAlmostEqual(sum(weights[2:]), 1.0)
        self.assertAlmostEqual(weights[3] / weights[4], 3.0)
        self.assertEqual(report["targeted_windows"], 2)

    def test_token_budget_audit_counts_truncation_and_clipped_span(self):
        rows = [{
            "id": "w1", "record_id": "r1", "tokens": ["a", "long", "b", "c"],
            "n_prefix": 1,
            "spans": [{"start": 2, "end": 4, "type_id": 1}],
        }]
        report = audit_rows(rows, DummyTokenizer(), max_len=6)
        self.assertEqual(report["truncated_windows"], 1)
        self.assertEqual(report["clipped_labeled_spans"], 1)

    def test_raw_offsets_and_overlap_coverage(self):
        text = "NFD a\u0300 và NFC à. " + " ".join(f"t{i}" for i in range(50))
        tokens = tokenize_raw(text)
        self.assertTrue(all(text[start:end] == token for token, start, end in tokens))
        windows = sliding_windows(text, max_words=15, overlap_words=5)
        self.assertGreater(len(windows), 1)
        self.assertGreater(len(windows[1]["prefix"]), 0)

    def test_no_header_still_gets_previous_raw_context(self):
        text = " ".join(f"t{i}" for i in range(50))
        windows = sliding_windows(
            text, max_words=12, overlap_words=3, include_header=False, context_words=5
        )
        self.assertGreater(len(windows), 1)
        self.assertEqual([token for token, _, _ in windows[1]["prefix"]], [
            f"t{i}" for i in range(4, 9)
        ])

    def test_record_to_windows(self):
        text = "Bệnh nhân không đau bụng hôm nay."
        start = text.index("đau bụng")
        record = {
            "id": "r1",
            "text": text,
            "entities": [{
                "text": "đau bụng",
                "position": [start, start + len("đau bụng")],
                "type": "TRIỆU_CHỨNG",
                "assertions": ["isNegated"],
                "candidates": [],
            }],
            "genre": "free_text",
            "record_kind": "segment",
        }
        windows, errors = record_to_windows(record, 10, 3, 1.0, 0.0, 1)
        self.assertEqual(errors, [])
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0]["spans"]), 1)
        span = windows[0]["spans"][0]
        self.assertEqual(span["assertions"], [1, 0, 0])
        self.assertTrue(any(tag.startswith("B-") for tag in windows[0]["ner_tags"]))

    def test_long_entity_covered_by_overlap(self):
        text = " ".join(f"t{i}" for i in range(40))
        tokens = tokenize_raw(text)
        start, end = tokens[12][1], tokens[16][2]
        entity = {"text": text[start:end], "position": [start, end]}
        self.assertEqual(
            coverage_errors(text, [entity], max_words=15, overlap_words=5),
            [],
        )

    def test_hybrid_model_shapes_and_loss(self):
        model = HybridNER("dummy", 11, encoder=DummyEncoder())
        result = model(
            torch.tensor([[1, 2, 3, 4]]),
            torch.ones((1, 4), dtype=torch.long),
            bio_labels=torch.tensor([[-100, 1, 2, -100]]),
            token_loss_mask=torch.tensor([[0, 1, 1, 0]]),
            span_starts=torch.tensor([[1, 2]]),
            span_ends=torch.tensor([[2, 2]]),
            span_labels=torch.tensor([[1, 0]]),
            assertion_labels=torch.tensor([[[1, 0, 0], [0, 0, 0]]]),
            assertion_mask=torch.tensor([[1, 0]]),
        )
        self.assertEqual(tuple(result["bio_logits"].shape), (1, 4, 11))
        self.assertEqual(tuple(result["span_logits"].shape), (1, 2, 6))
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_focused_assertion_update_preserves_ner_and_other_rows(self):
        model = HybridNER("dummy", 11, encoder=DummyEncoder())
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.assertion_head.weight.requires_grad = True
        model.assertion_head.bias.requires_grad = True
        encoder_before = model.encoder.embedding.weight.detach().clone()
        other_weight_before = model.assertion_head.weight[1:].detach().clone()
        other_bias_before = model.assertion_head.bias[1:].detach().clone()
        focus_before = model.assertion_head.weight[0].detach().clone()
        optimizer = torch.optim.AdamW(model.assertion_head.parameters(), lr=1e-2, weight_decay=0)
        result = model(
            torch.tensor([[1, 2, 3, 4]]),
            torch.ones((1, 4), dtype=torch.long),
            span_starts=torch.tensor([[1]]),
            span_ends=torch.tensor([[2]]),
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            result["assertion_logits"][..., 0], torch.ones((1, 1))
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(model.encoder.embedding.weight, encoder_before))
        self.assertTrue(torch.equal(model.assertion_head.weight[1:], other_weight_before))
        self.assertTrue(torch.equal(model.assertion_head.bias[1:], other_bias_before))
        self.assertFalse(torch.equal(model.assertion_head.weight[0], focus_before))


if __name__ == "__main__":
    unittest.main()
