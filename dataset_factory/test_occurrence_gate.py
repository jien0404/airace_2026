"""Kiểm tra occurrence gate: marker, occurrence bị bỏ, và cue assertion hai chiều."""

from __future__ import annotations

import unittest

from .occurrence_report import analyse_record, summarise
from .pilot import (
    _cues_before,
    _cues_support,
    _generated_errors,
    _parse_marked_text,
    build_entity_contract,
    build_generation_brief,
    _coverage_case,
)
from .sources import EntityCatalog


def _plan_item(seed_id, text, *, should_label=True, typ="TRIỆU_CHỨNG", assertions=(),
               role="hard_case", occurrence_role="primary", gate_reason=None):
    item = {
        "seed_id": seed_id,
        "text": text,
        "semantic_type": "TRIỆU_CHỨNG",
        "occurrence_index": 0,
        "should_label": should_label,
        "type": typ if should_label else None,
        "assertions": list(assertions) if should_label else [],
        "role": role,
        "provenance": [],
        "occurrence_role": occurrence_role,
    }
    if gate_reason:
        item["gate_reason"] = gate_reason
    return item


def _request(plan):
    return {
        "request_id": "A:gate:v1",
        "track": "A",
        "variant_index": 0,
        "case": {"case_id": "test.case"},
        "generation_brief": {
            "header_mode": "none",
            "document_format": "free_text",
            "length_chars": [20, 400],
            "design": "occurrence_gate_v6",
            "assertion_policy": {"allow_historical_drug": False},
            "entity_budget": {
                "labeled": [0, 9], "intentional_o": [0, 9],
                "total_medical_mentions_max": 12, "primary_hard_pattern": 1,
            },
        },
        "entity_contract": {
            "schema_version": 2,
            "policy": "occurrence_gate_v1",
            "entity_plan": plan,
        },
    }


def _mention(seed_id, text, *, should_label=True, typ="TRIỆU_CHỨNG", assertions=(),
             evidence=None, rationale="lý do"):
    return {
        "seed_id": seed_id,
        "text": text,
        "should_label": should_label,
        "type": typ if should_label else None,
        "assertions": list(assertions) if should_label else [],
        "context_evidence": evidence or text,
        "rationale": rationale,
    }


class MarkerParsingTest(unittest.TestCase):
    def test_offsets_land_on_raw_after_markers_removed(self):
        raw, marks, errors = _parse_marked_text(
            'Người bệnh <E id="core:d1">đau ngực</E> từ sáng.\n'
            'Dặn dò: nhận biết sớm dấu hiệu đau ngực.'
        )
        self.assertEqual(errors, [])
        self.assertNotIn("<E", raw)
        self.assertEqual(len(marks), 1)
        start, end = marks[0]["position"]
        self.assertEqual(raw[start:end], "đau ngực")
        # occurrence thứ hai vẫn nằm trong raw nhưng không có marker
        self.assertEqual(raw.count("đau ngực"), 2)

    def test_marker_wrapping_whitespace_is_rejected(self):
        _, _, errors = _parse_marked_text('a <E id="x"> đau ngực </E> b')
        self.assertTrue(any("khoảng trắng" in error for error in errors))


class OccurrenceGateValidationTest(unittest.TestCase):
    def test_omitted_occurrence_must_exist_unmarked(self):
        plan = [
            _plan_item("core:d1", "đau ngực"),
            _plan_item(
                "repeat:01", "đau ngực", should_label=False,
                role="occurrence_gate_negative", occurrence_role="repeat_omitted",
                gate_reason="câu dặn dò",
            ),
        ]
        good = {
            "source_case_id": "test.case",
            "document_format": "free_text",
            "section_path": [],
            "marked_text": (
                'Người bệnh <E id="core:d1">đau ngực</E> từ sáng. '
                'Dặn dò: khi thấy đau ngực tái phát thì tái khám.'
            ),
            "mentions": [
                _mention("core:d1", "đau ngực", evidence="Người bệnh đau ngực từ sáng"),
                _mention(
                    "repeat:01", "đau ngực", should_label=False,
                    evidence="khi thấy đau ngực tái phát",
                ),
            ],
        }
        errors, draft = _generated_errors(_request(plan), good)
        self.assertEqual(errors, [])
        labeled = [row for row in draft["mentions"] if row["should_label"]]
        self.assertEqual(len(labeled), 1)
        start, end = labeled[0]["position"]
        self.assertEqual(draft["text"][start:end], "đau ngực")
        omitted = [row for row in draft["mentions"] if not row["should_label"]][0]
        self.assertGreater(omitted["position"][0], labeled[0]["position"][0])

        missing = dict(good)
        missing["marked_text"] = 'Người bệnh <E id="core:d1">đau ngực</E> từ sáng.'
        errors, _ = _generated_errors(_request(plan), missing)
        self.assertTrue(any("thiếu trong văn bản" in error for error in errors))

    def test_extra_unplanned_occurrence_is_rejected(self):
        plan = [_plan_item("core:d1", "đau ngực")]
        data = {
            "source_case_id": "test.case",
            "document_format": "free_text",
            "section_path": [],
            "marked_text": (
                'Người bệnh <E id="core:d1">đau ngực</E> từ sáng, '
                'chiều nay vẫn còn đau ngực nhiều.'
            ),
            "mentions": [_mention("core:d1", "đau ngực", evidence="Người bệnh đau ngực từ sáng")],
        }
        errors, _ = _generated_errors(_request(plan), data)
        self.assertTrue(any("xuất hiện 2, yêu cầu đúng 1" in error for error in errors))

    def test_assertion_label_follows_text_in_both_directions(self):
        plan = [_plan_item("core:d1", "đau ngực", assertions=["isHistorical"])]
        no_cue = {
            "source_case_id": "test.case",
            "document_format": "free_text",
            "section_path": [],
            "marked_text": 'Hiện tại người bệnh <E id="core:d1">đau ngực</E> nhiều.',
            "mentions": [
                _mention(
                    "core:d1", "đau ngực", assertions=["isHistorical"],
                    evidence="người bệnh đau ngực nhiều",
                )
            ],
        }
        # Không có cue quá khứ trong câu ⇒ nhãn đúng là rỗng. Sửa nhãn theo văn bản
        # thay vì loại cả draft, và ghi lại việc đã sửa.
        errors, draft = _generated_errors(_request(plan), no_cue)
        self.assertEqual(errors, [])
        self.assertEqual(draft["mentions"][0]["assertions"], [])
        self.assertTrue(any(
            flag.startswith("assertion_repair:") for flag in draft["quality_flags"]
        ))

        plan_plain = [_plan_item("core:d1", "đau ngực")]
        cue_without_assertion = {
            "source_case_id": "test.case",
            "document_format": "free_text",
            "section_path": [],
            "marked_text": 'Trước đây người bệnh từng <E id="core:d1">đau ngực</E> khi gắng sức.',
            "mentions": [
                _mention("core:d1", "đau ngực", evidence="từng đau ngực khi gắng sức")
            ],
        }
        # Ngược lại: câu nói rõ quá khứ ⇒ isHistorical mới là nhãn đúng dù plan để rỗng.
        errors, draft = _generated_errors(_request(plan_plain), cue_without_assertion)
        self.assertEqual(errors, [])
        self.assertEqual(draft["mentions"][0]["assertions"], ["isHistorical"])
        self.assertTrue(any(
            flag.startswith("assertion_repair:") for flag in draft["quality_flags"]
        ))

    def test_cue_detection_respects_word_boundary(self):
        self.assertEqual(_cues_before("Siêu âm bàng quang bình thường", 8), set())
        self.assertIn("isHistorical", _cues_before("tiền sử hen phế quản", 9))
        # Cue bằng chứng rộng tay hơn cue mâu thuẫn: nhận cả người thân và mốc năm.
        text = "Quan hệ gia đình: Mẹ bệnh nhân đang điều trị tăng huyết áp"
        start = text.index("tăng huyết áp")
        self.assertIn("isFamily", _cues_support(text, [start, start + 13]))
        history = "Tiền sử bản thân: Năm 2020, bệnh nhân từng mắc viêm phổi"
        start = history.index("viêm phổi")
        self.assertIn("isHistorical", _cues_support(history, [start, start + 9]))
        # Phủ định đứng SAU concept vẫn phải được nhận.
        posed = "Ghi nhận hiện tại sốt không có"
        start = posed.index("sốt")
        self.assertIn("isNegated", _cues_support(posed, [start, start + 3]))
        # "từng đợt" là lượng từ, không phải cue tiền sử.
        self.assertEqual(_cues_before("điều trị từng đợt với amoxicillin", 22), set())


class GatePlanTest(unittest.TestCase):
    def test_plan_contains_both_repeat_kinds_and_assertions(self):
        catalog = EntityCatalog("A", 4321, max_tier="A")
        roles: dict[str, int] = {}
        assertions = 0
        labeled = 0
        for index in range(60):
            case = _coverage_case(index)
            brief = build_generation_brief(case, index, "occurrence_gate_v6")
            plan = build_entity_contract(case, brief, catalog, 4321)["entity_plan"]
            for item in plan:
                roles[item["occurrence_role"]] = roles.get(item["occurrence_role"], 0) + 1
                if item["should_label"]:
                    labeled += 1
                    assertions += bool(item["assertions"])
        self.assertIn("repeat_omitted", roles)
        self.assertIn("repeat_labeled", roles)
        self.assertGreater(assertions / labeled, 0.10)
        self.assertLess(assertions / labeled, 0.32)


class OccurrenceReportTest(unittest.TestCase):
    def test_report_separates_mixed_from_take_all_and_covered(self):
        text = (
            "Người bệnh đau ngực từ sáng, chiều vẫn đau ngực.\n"
            "Dặn dò: nếu đau ngực tái phát thì tái khám.\n"
            "Kết luận: viêm phổi thuỳ dưới phải."
        )
        entities = [
            {"text": "đau ngực", "position": [11, 19], "type": "TRIỆU_CHỨNG", "assertions": []},
            {"text": "đau ngực", "position": [
                text.index("đau ngực", 20), text.index("đau ngực", 20) + 8,
            ], "type": "TRIỆU_CHỨNG", "assertions": []},
            {"text": "viêm phổi thuỳ dưới phải", "position": [
                text.index("viêm phổi"), text.index("viêm phổi") + len("viêm phổi thuỳ dưới phải"),
            ], "type": "CHẨN_ĐOÁN", "assertions": []},
        ]
        rows = analyse_record({"id": "t1", "text": text, "entities": entities})
        by_surface = {row["surface"]: row for row in rows}
        self.assertEqual(by_surface["đau ngực"]["selected_count"], 2)
        self.assertEqual(by_surface["đau ngực"]["omitted_count"], 1)
        summary = summarise(rows, 1)
        self.assertEqual(summary["mixed_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()


class BatchV3RulesTest(unittest.TestCase):
    """Các luật thêm cho mẻ v3: bucket có profile riêng, chặn phủ định kép, cue đa hình thức."""

    def test_sparse_bucket_carries_its_own_type_profile(self):
        from .pilot import DENSITY_BUCKETS
        name, span, weight, weights, assertion_rate = DENSITY_BUCKETS[0]
        self.assertEqual(name, "sparse")
        # Lớp tài liệu thưa của Part 3 nặng CHẨN_ĐOÁN và ít assertion.
        self.assertGreater(weights["CHẨN_ĐOÁN"], weights["TRIỆU_CHỨNG"])
        self.assertLess(assertion_rate, 0.15)
        self.assertGreaterEqual(weight, 0.30)

    def test_density_target_returns_bucket_profile(self):
        import random
        from .pilot import _density_target
        seen_sparse = False
        for seed in range(60):
            count, weights, rate = _density_target(random.Random(seed), False, 1000.0)
            self.assertIsInstance(count, int)
            if weights and weights["CHẨN_ĐOÁN"] > weights["TRIỆU_CHỨNG"]:
                seen_sparse = True
                self.assertLess(rate, 0.15)
        self.assertTrue(seen_sparse, "không bốc trúng bucket thưa lần nào")

    def test_self_negated_surface_never_gets_isnegated(self):
        import random
        from .pilot import _assign_supplement_assertions
        plan = [
            {
                "text": "không có máu", "type": "TRIỆU_CHỨNG", "role": "supplemental_random",
                "should_label": True, "assertions": [],
            }
            for _ in range(200)
        ]
        _assign_supplement_assertions(plan, random.Random(7), None, 1.0)
        self.assertEqual(
            [item for item in plan if "isNegated" in item["assertions"]], [],
            "surface tự phủ định vẫn bị gán isNegated → sinh phủ định kép",
        )

    def test_assertion_gets_a_cue_style(self):
        import random
        from .pilot import _assign_supplement_assertions, ASSERTION_CUE_STYLES
        plan = [
            {
                "text": "đau ngực", "type": "TRIỆU_CHỨNG", "role": "supplemental_random",
                "should_label": True, "assertions": [],
            }
            for _ in range(200)
        ]
        _assign_supplement_assertions(plan, random.Random(3), None, 1.0)
        styles = {
            item.get("assertion_cue_style") for item in plan if item["assertions"]
        }
        self.assertTrue(styles)
        self.assertEqual(styles - {name for name, _ in ASSERTION_CUE_STYLES}, set())
        self.assertGreater(len(styles), 1, "cue chỉ có một hình thức")
