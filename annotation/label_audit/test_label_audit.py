"""Kiểm tra bộ rà soát nhãn: sàng đúng loại, và apply không đụng nhãn gốc."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .run import apply_decisions, finding_id
from .screen import screen_record


def _record(text, entities):
    return {"id": "gt2:1", "source": "gt2", "file_stem": "1", "text": text, "entities": entities}


def _entity(text, start, typ, assertions=()):
    return {
        "text": text, "position": [start, start + len(text)], "type": typ,
        "assertions": list(assertions), "candidates": [],
    }


class ScreenTest(unittest.TestCase):
    def test_assertion_on_test_result_is_mechanical(self):
        text = "Xét nghiệm: PLT: 190 g/L."
        rows = screen_record(_record(
            text, [_entity("PLT: 190 g/L", text.index("PLT"), "KẾT_QUẢ_XÉT_NGHIỆM", ["isNegated"])]
        ))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "assertion_on_forbidden_type")
        self.assertTrue(rows[0]["mechanical"])
        self.assertEqual(rows[0]["proposed"]["assertions"], [])

    def test_missing_negation_is_flagged(self):
        text = "Người bệnh không có đau ngực trong đợt này."
        rows = screen_record(_record(
            text, [_entity("đau ngực", text.index("đau ngực"), "TRIỆU_CHỨNG")]
        ))
        self.assertEqual([row["kind"] for row in rows], ["cue_without_assertion"])
        self.assertEqual(rows[0]["proposed"]["assertions"], ["isNegated"])

    def test_historical_without_any_cue_is_flagged(self):
        text = "Hiện tại người bệnh vàng da và mệt nhiều."
        rows = screen_record(_record(
            text, [_entity("vàng da", text.index("vàng da"), "TRIỆU_CHỨNG", ["isHistorical"])]
        ))
        self.assertEqual([row["kind"] for row in rows], ["assertion_without_cue"])
        self.assertEqual(rows[0]["proposed"]["assertions"], [])

    def test_supported_assertion_is_not_flagged(self):
        text = "Tiền sử: người bệnh từng mắc viêm phổi năm 2019."
        rows = screen_record(_record(
            text, [_entity("viêm phổi", text.index("viêm phổi"), "CHẨN_ĐOÁN", ["isHistorical"])]
        ))
        self.assertEqual(rows, [])

    def test_clean_label_is_not_flagged(self):
        text = "Người bệnh đau ngực từ sáng nay."
        rows = screen_record(_record(
            text, [_entity("đau ngực", text.index("đau ngực"), "TRIỆU_CHỨNG")]
        ))
        self.assertEqual(rows, [])


class ApplyTest(unittest.TestCase):
    def test_apply_writes_copy_and_leaves_original_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "src"
            (source / "notes").mkdir(parents=True)
            (source / "labels").mkdir(parents=True)
            text = "Người bệnh không có đau ngực."
            (source / "notes" / "1.txt").write_text(text, encoding="utf-8")
            entities = [_entity("đau ngực", text.index("đau ngực"), "TRIỆU_CHỨNG")]
            label_path = source / "labels" / "1.json"
            label_path.write_text(json.dumps(entities, ensure_ascii=False), encoding="utf-8")

            audit = root / "audit"
            audit.mkdir()
            finding = {
                "file": "gt2:1", "entity_index": 0, "source": "gt2", "kind": "cue_without_assertion",
                "mechanical": False, "rule": "r", "reason": "r",
                "current": {"assertions": [], "type": "TRIỆU_CHỨNG"},
                "proposed_assertions": ["isNegated"], "entity": entities[0],
                "context": {"before": "", "surface": "đau ngực", "after": "", "window_start": 0},
            }
            (audit / "findings.jsonl").write_text(
                json.dumps(finding, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (audit / "decisions.json").write_text(
                json.dumps({finding_id(finding): "accept"}), encoding="utf-8"
            )

            from . import run as run_module
            original_sources = dict(run_module.SOURCES)
            original_root = run_module.REPO_ROOT
            try:
                run_module.SOURCES.clear()
                run_module.SOURCES["gt2"] = "src"
                run_module.REPO_ROOT = root
                manifest = apply_decisions(audit, root / "fixed")
            finally:
                run_module.SOURCES.clear()
                run_module.SOURCES.update(original_sources)
                run_module.REPO_ROOT = original_root

            self.assertEqual(manifest["accepted_fixes"], {"gt2": 1})
            fixed = json.loads((root / "fixed" / "gt2" / "labels" / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(fixed[0]["assertions"], ["isNegated"])
            self.assertEqual(fixed[0]["position"], entities[0]["position"])
            # nhãn gốc không đổi
            self.assertEqual(
                json.loads(label_path.read_text(encoding="utf-8"))[0]["assertions"], []
            )


if __name__ == "__main__":
    unittest.main()


class SpanAndTypeScreenTest(unittest.TestCase):
    def test_span_swallowing_negation_is_flagged_with_trimmed_span(self):
        text = "Người bệnh: Không đau đầu, không sốt."
        start = text.index("Không đau đầu")
        rows = screen_record(_record(
            text, [_entity("Không đau đầu", start, "TRIỆU_CHỨNG", ["isNegated"])]
        ))
        self.assertEqual([row["kind"] for row in rows], ["span_swallows_negation"])
        proposed = rows[0]["proposed"]
        self.assertEqual(proposed["text"], "đau đầu")
        self.assertEqual(text[proposed["position"][0]:proposed["position"][1]], "đau đầu")

    def test_negation_in_span_without_assertion_gets_both_fixes(self):
        text = "Khám: không chóng mặt trong đợt này."
        start = text.index("không chóng mặt")
        rows = screen_record(_record(
            text, [_entity("không chóng mặt", start, "TRIỆU_CHỨNG")]
        ))
        self.assertEqual([row["kind"] for row in rows], ["negation_inside_span"])
        self.assertEqual(rows[0]["proposed"]["assertions"], ["isNegated"])
        self.assertEqual(rows[0]["proposed"]["text"], "chóng mặt")

    def test_type_minority_uses_corpus_majority(self):
        from annotation.label_audit.screen import screen_records
        majority = []
        # cần đủ 9/10 để vượt ngưỡng đa số 85%
        for i in range(9):
            body = f"Người bệnh khó thở nhiều ngày {i}."
            majority.append({
                **_record(body, [
                    _entity("khó thở", body.index("khó thở"), "TRIỆU_CHỨNG"),
                ]),
                "id": f"gt2:{i}",
            })
        odd_text = "Chẩn đoán: khó thở chưa rõ nguyên nhân."
        odd = {**_record(odd_text, [
            _entity("khó thở", odd_text.index("khó thở"), "CHẨN_ĐOÁN"),
        ]), "id": "gt2:99"}
        rows = [r for r in screen_records(majority + [odd])
                if r["kind"] == "type_minority_vs_corpus"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["proposed"]["type"], "TRIỆU_CHỨNG")

    def test_llm_span_must_be_substring_of_current(self):
        from annotation.label_audit.run import _resolve_change
        finding = {
            "entity": {"text": "đau đầu", "position": [10, 17], "type": "TRIỆU_CHỨNG"},
            "current": {"assertions": [], "type": "TRIỆU_CHỨNG"},
        }
        widened = _resolve_change(
            finding, {"verdict": "fix", "assertions": ["isNegated"], "span": "không đau đầu"}
        )
        self.assertNotIn("proposed_position", widened, "LLM nới rộng span mà vẫn được nhận")
        self.assertEqual(widened["proposed_assertions"], ["isNegated"])

    def test_type_change_to_non_assertion_type_clears_assertions(self):
        from annotation.label_audit.run import _resolve_change
        finding = {
            "entity": {"text": "PLT: 190", "position": [0, 8], "type": "TRIỆU_CHỨNG"},
            "current": {"assertions": ["isNegated"], "type": "TRIỆU_CHỨNG"},
        }
        change = _resolve_change(finding, {
            "verdict": "fix", "assertions": ["isNegated"], "type": "KẾT_QUẢ_XÉT_NGHIỆM",
        })
        self.assertEqual(change["proposed_type"], "KẾT_QUẢ_XÉT_NGHIỆM")
        self.assertEqual(change["proposed_assertions"], [])
