from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.build_dataset_v2 import build_track
from training.build_final_n_datasets import sanitize, write_jsonl


class FinalNBuildTest(unittest.TestCase):
    def test_assertion_firewall(self):
        row = sanitize({
            "id": "x", "text": "hen", "entities": [{
                "text": "hen", "position": [0, 3], "type": "CHẨN_ĐOÁN",
                "assertions": ["isFamily", "isNegated", "isHistorical"],
            }],
        })
        self.assertEqual(row["entities"][0]["assertions"], ["isHistorical"])

    def test_source_mass_is_preserved_after_scaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            rows = []
            for index, group in enumerate(("a", "b")):
                rows.append({
                    "id": str(index), "text": "đau đầu", "entities": [{
                        "text": "đau đầu", "position": [0, 7],
                        "type": "TRIỆU_CHỨNG", "assertions": [],
                    }], "record_kind": "segment", "source": group,
                    "_sampling_group": group,
                })
            write_jsonl(source / "train.jsonl", rows)
            write_jsonl(source / "validation.jsonl", rows[:1])
            write_jsonl(source / "test.jsonl", rows[:1])
            out = root / "windows"
            build_track(
                source, out, 180, 45, 0, 1, 0.6, 1,
                source_masses={"a": 0.25, "b": 0.75},
            )
            import json
            with (out / "train.jsonl").open() as stream:
                windows = [json.loads(line) for line in stream]
            masses = {
                group: sum(row["sample_weight"] for row in windows if row["sampling_group"] == group)
                for group in ("a", "b")
            }
            self.assertAlmostEqual(masses["b"] / masses["a"], 3.0)


if __name__ == "__main__":
    unittest.main()
