from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from .align import align_predictions, validate_labels
from .run import (
    write_partial_submission,
    validate_submission,
    write_scores_template,
)


class AlignPredictionsTest(unittest.TestCase):
    def test_occurrence_nfd_and_candidate_validation(self):
        raw = "ho và ho\nXe\u0301t nghie\u0323\u0302m ALT: âm tính"
        predictions = [
            {
                "line_id": 1,
                "text": "ho",
                "occurrence": 1,
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
                "confidence": "high",
            },
            {
                "line_id": 1,
                "text": "ho",
                "occurrence": 2,
                "type": "TRIỆU_CHỨNG",
                "assertions": ["isNegated"],
                "candidates": [],
                "confidence": "high",
            },
            {
                "line_id": 2,
                "text": "Xét nghiệm ALT",
                "occurrence": 1,
                "type": "TÊN_XÉT_NGHIỆM",
                "assertions": ["isHistorical"],
                "candidates": ["I10"],
                "confidence": "high",
            },
            {
                "line_id": 2,
                "text": "âm tính",
                "occurrence": 1,
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "assertions": [],
                "candidates": [],
                "confidence": "high",
            },
        ]
        labels, audit, _ = align_predictions(
            raw,
            predictions,
            valid_icd={"I10"},
            valid_rxnorm={"123"},
        )
        self.assertEqual(4, len(labels))
        self.assertEqual([0, 2], labels[0]["position"])
        self.assertEqual([6, 8], labels[1]["position"])
        self.assertEqual([], labels[2]["assertions"])
        self.assertEqual([], labels[2]["candidates"])
        self.assertEqual([], validate_labels(raw, labels))
        self.assertEqual(1.0, audit["alignment_rate"])

    def test_overlap_keeps_higher_confidence(self):
        raw = "đau bụng ngày càng nặng"
        predictions = [
            {
                "line_id": 1,
                "text": "đau bụng ngày càng nặng",
                "occurrence": 1,
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
                "confidence": "low",
            },
            {
                "line_id": 1,
                "text": "đau bụng",
                "occurrence": 1,
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
                "confidence": "high",
            },
        ]
        labels, audit, _ = align_predictions(
            raw,
            predictions,
            valid_icd=set(),
            valid_rxnorm=set(),
        )
        self.assertEqual(["đau bụng"], [entity["text"] for entity in labels])
        self.assertEqual(1, len(audit["overlap_dropped"]))

    def test_invalid_codes_are_removed(self):
        raw = "tăng huyết áp và aspirin 81 mg"
        predictions = [
            {
                "line_id": 1,
                "text": "tăng huyết áp",
                "occurrence": 1,
                "type": "CHẨN_ĐOÁN",
                "assertions": [],
                "candidates": ["I10", "ZZZ"],
                "confidence": "high",
            },
            {
                "line_id": 1,
                "text": "aspirin 81 mg",
                "occurrence": 1,
                "type": "THUỐC",
                "assertions": [],
                "candidates": ["243670", "abc"],
                "confidence": "high",
            },
        ]
        labels, audit, _ = align_predictions(
            raw,
            predictions,
            valid_icd={"I10"},
            valid_rxnorm={"243670"},
        )
        self.assertEqual(["I10"], labels[0]["candidates"])
        self.assertEqual(["243670"], labels[1]["candidates"])
        self.assertEqual(2, len(audit["invalid_candidates_dropped"]))


class SubmissionTest(unittest.TestCase):
    def test_partial_zip_has_exactly_100_files(self):
        selected = {3}
        notes = {"3": "ho"}
        labels = {
            "3": [{
                "text": "ho",
                "position": [0, 2],
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
            }]
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.zip"
            write_partial_submission(path, selected, labels)
            self.assertEqual([], validate_submission(path, selected, notes))
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(100, len(archive.namelist()))
                self.assertEqual([], json.loads(archive.read("labels/1.json")))
                self.assertEqual(labels["3"], json.loads(archive.read("labels/3.json")))

    def test_scores_template_preserves_entered_scores_and_refreshes_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            write_scores_template(
                run_dir,
                {"reference_v66ab": {"zip": "submissions/old.zip"}},
            )
            path = run_dir / "scores_template.csv"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    "reference_v66ab,submissions/old.zip,,,,,\n",
                    "reference_v66ab,submissions/old.zip,5.0,50,60,40,done\n",
                ),
                encoding="utf-8",
            )
            write_scores_template(
                run_dir,
                {"reference_v66ab": {"zip": "submissions/new.zip"}},
            )
            refreshed = path.read_text(encoding="utf-8")
            self.assertIn(
                "reference_v66ab,submissions/new.zip,5.0,50,60,40,done",
                refreshed,
            )


if __name__ == "__main__":
    unittest.main()
