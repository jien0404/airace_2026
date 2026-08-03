import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

from training.predict_v2 import encoder_capacity, _encode_words
from training.assertion_policy import (
    apply_policy_to_predictions,
    assertions_from_probabilities,
    postprocess_zip,
    resolve_thresholds,
)


class Cap(unittest.TestCase):
    def _model(self, limit, pad):
        cfg = types.SimpleNamespace(max_position_embeddings=limit, pad_token_id=pad)
        return types.SimpleNamespace(encoder=types.SimpleNamespace(config=cfg))

    def test_phobert(self):
        # 258 vị trí, padding_idx=1 -> chuỗi dài nhất an toàn là 256
        self.assertEqual(encoder_capacity(self._model(258, 1)), 256)

    def test_bert_style(self):
        self.assertEqual(encoder_capacity(self._model(512, 0)), 511)

    def test_encode_respects_max_len(self):
        tok = types.SimpleNamespace(encode=lambda w, add_special_tokens: [7, 8, 9])
        ids, first, last, limit = _encode_words(["a"] * 100, tok, 20, 0, 2, 3)
        self.assertLessEqual(len(ids), 20)
        self.assertEqual(limit, len(first))
        self.assertTrue(all(i < len(ids) for i in first + last))


class AssertionPolicyTest(unittest.TestCase):
    def test_part3_firewall(self):
        rows = [{
            "text": "bệnh",
            "position": [0, 4],
            "type": "CHẨN_ĐOÁN",
            "assertions": ["isNegated", "isFamily", "isHistorical"],
            "candidates": ["A00"],
        }]
        output, stats = apply_policy_to_predictions(rows)
        self.assertEqual(output[0]["assertions"], ["isHistorical"])
        self.assertEqual(rows[0]["assertions"], [
            "isNegated", "isFamily", "isHistorical"
        ])
        self.assertEqual(output[0]["candidates"], ["A00"])
        self.assertEqual(stats["entities_changed"], 1)

    def test_per_assertion_thresholds(self):
        thresholds = resolve_thresholds(0.60, {
            "isHistorical": 0.50,
            "isNegated": 0.75,
        })
        selected = assertions_from_probabilities(
            "TRIỆU_CHỨNG",
            ["isNegated", "isFamily", "isHistorical"],
            [0.70, 0.99, 0.55],
            thresholds,
            "part3",
        )
        self.assertEqual(selected, ["isHistorical"])

    def test_legacy_keeps_family_and_negated_diagnosis(self):
        thresholds = resolve_thresholds(0.60)
        selected = assertions_from_probabilities(
            "CHẨN_ĐOÁN",
            ["isNegated", "isFamily", "isHistorical"],
            [0.70, 0.70, 0.20],
            thresholds,
            "legacy",
        )
        self.assertEqual(selected, ["isNegated", "isFamily"])

    def test_zip_postprocess_changes_only_assertions(self):
        rows = [{
            "text": "đau",
            "position": [5, 8],
            "type": "TRIỆU_CHỨNG",
            "assertions": ["isFamily"],
            "candidates": [],
        }]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zip"
            target = Path(directory) / "target.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("1.json", json.dumps(rows))
            report = postprocess_zip(source, target, "part3")
            with zipfile.ZipFile(target) as archive:
                output = json.loads(archive.read("1.json"))
            self.assertEqual(output[0]["assertions"], [])
            self.assertEqual(output[0]["position"], [5, 8])
            self.assertEqual(report["removed:TRIỆU_CHỨNG:isFamily"], 1)


if __name__ == "__main__":
    unittest.main()
