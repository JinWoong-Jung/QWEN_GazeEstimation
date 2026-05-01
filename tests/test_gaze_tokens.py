from __future__ import annotations

import unittest

from model.utils.gaze_tokens import (
    ANSWER_END,
    ANSWER_START,
    COORD_BINS,
    GAZE_OBJ_UNKNOWN,
    OBJECT_PREFIX,
    POINT_PREFIX,
    build_gaze_special_tokens,
    build_structured_target_text,
    dequantize_coord,
    format_loc_token,
    format_obj_token,
    is_valid_structured_output,
    parse_structured_output_text,
    quantize_coord,
)


class TestQuantize(unittest.TestCase):
    def test_boundary_zero(self):
        self.assertEqual(quantize_coord(0.0), 0)

    def test_boundary_one(self):
        self.assertEqual(quantize_coord(1.0), 999)

    def test_midpoint(self):
        b = quantize_coord(0.5)
        # 0.5*999 = 499.5 → rounds to 500 → 500/999 ≈ 0.5005, within 2 decimal places
        self.assertAlmostEqual(dequantize_coord(b), 0.5, places=2)

    def test_roundtrip_arbitrary(self):
        for v in [0.0, 0.123, 0.5, 0.999, 1.0]:
            b = quantize_coord(v)
            rec = dequantize_coord(b)
            self.assertAlmostEqual(rec, max(0.0, min(1.0, round(v * 999) / 999)), places=6)

    def test_clamp_below_zero(self):
        self.assertEqual(quantize_coord(-0.5), 0)

    def test_clamp_above_one(self):
        self.assertEqual(quantize_coord(1.5), 999)

    def test_custom_bin_count(self):
        self.assertEqual(quantize_coord(1.0, bins=128), 127)
        self.assertEqual(quantize_coord(1.5, bins=128), 127)
        self.assertEqual(dequantize_coord(127, bins=128), 1.0)


class TestFormatTokens(unittest.TestCase):
    def test_loc_token_format(self):
        self.assertEqual(format_loc_token(0), "<loc_000>")
        self.assertEqual(format_loc_token(42), "<loc_042>")
        self.assertEqual(format_loc_token(999), "<loc_999>")

    def test_obj_token_width_3(self):
        self.assertEqual(format_obj_token(0, 3), "<obj_000>")
        self.assertEqual(format_obj_token(157, 3), "<obj_157>")

    def test_obj_token_width_4(self):
        self.assertEqual(format_obj_token(1024, 4), "<obj_1024>")

    def test_obj_token_width_auto(self):
        from model.utils.gaze_tokens import _obj_token_width
        self.assertEqual(_obj_token_width(158), 3)   # max ID 157 → 3 digits
        self.assertEqual(_obj_token_width(1000), 3)  # max ID 999 → 3 digits
        self.assertEqual(_obj_token_width(1001), 4)  # max ID 1000 → 4 digits
        self.assertEqual(_obj_token_width(10001), 5) # max ID 10000 → 5 digits


class TestBuildSpecialTokens(unittest.TestCase):
    def test_token_count(self):
        num_classes = 158
        tokens = build_gaze_special_tokens(num_classes)
        # 1000 loc + 158 obj + 1 unknown obj
        self.assertEqual(len(tokens), 1000 + 158 + 1)

    def test_loc_tokens_range(self):
        tokens = build_gaze_special_tokens(10)
        self.assertIn("<loc_000>", tokens)
        self.assertIn("<loc_999>", tokens)

    def test_custom_loc_tokens_range(self):
        tokens = build_gaze_special_tokens(10, coord_bins=128)
        self.assertEqual(len([tok for tok in tokens if tok.startswith("<loc_")]), 128)
        self.assertIn("<loc_000>", tokens)
        self.assertIn("<loc_127>", tokens)
        self.assertNotIn("<loc_128>", tokens)
        self.assertNotIn("<loc_999>", tokens)

    def test_obj_tokens_range(self):
        tokens = build_gaze_special_tokens(5)
        for i in range(5):
            self.assertIn(f"<obj_00{i}>", tokens)
        self.assertIn(GAZE_OBJ_UNKNOWN, tokens)


class TestBuildStructuredTargetText(unittest.TestCase):
    def test_output_is_string(self):
        t = build_structured_target_text(0.5, 0.5, 10, 100)
        self.assertIsInstance(t, str)

    def test_starts_ends_with_format_tokens(self):
        t = build_structured_target_text(0.3, 0.7, 5, 100)
        self.assertIn(POINT_PREFIX, t)
        self.assertIn(OBJECT_PREFIX, t)
        # ANSWER_START is now empty; ANSWER_END (<|im_end|>) is the chat-template
        # EOS added by the template, not part of the target text itself.
        self.assertFalse(t.startswith("<|im_start|>"))
        self.assertFalse(t.endswith("<|im_end|>"))

    def test_contains_obj_token(self):
        t = build_structured_target_text(0.0, 0.0, 42, 100)
        self.assertIn("<obj_042>", t)

    def test_contains_two_loc_tokens_and_one_obj_token(self):
        t = build_structured_target_text(0.5, 0.5, 0, 100)
        import re
        tokens = re.findall(r"<[^>]+>", t)
        self.assertEqual(sum(1 for tok in tokens if tok.startswith("<loc_")), 2)
        self.assertEqual(sum(1 for tok in tokens if tok.startswith("<obj_")), 1)

    def test_unknown_object_token(self):
        t = build_structured_target_text(0.5, 0.5, None, 100, obj_token=GAZE_OBJ_UNKNOWN)
        self.assertIn(GAZE_OBJ_UNKNOWN, t)

    def test_custom_coord_bins(self):
        t = build_structured_target_text(1.0, 0.0, 7, 100, coord_bins=128)
        self.assertIn("<loc_127><loc_000>", t)


class TestParseStructuredOutputText(unittest.TestCase):
    def _make(self, x: float, y: float, obj: int, nc: int = 100) -> str:
        return build_structured_target_text(x, y, obj, nc)

    def test_roundtrip(self):
        t = self._make(0.5, 0.3, 7)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertFalse(p["has_extra_text"])
        self.assertIsNotNone(p["point_bins"])
        self.assertIsNotNone(p["point_xy"])
        self.assertEqual(p["object_id"], 7)

    def test_boundary_zero(self):
        t = self._make(0.0, 0.0, 0)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["point_bins"], (0, 0))

    def test_boundary_one(self):
        t = self._make(1.0, 1.0, 99)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["point_bins"], (999, 999))

    def test_custom_coord_bins_roundtrip(self):
        t = build_structured_target_text(1.0, 0.0, 7, 100, coord_bins=128)
        p = parse_structured_output_text(t, 100, coord_bins=128)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["point_bins"], (127, 0))
        self.assertEqual(p["point_xy"], (1.0, 0.0))

    def test_custom_coord_bins_out_of_range(self):
        t = (
            f"{ANSWER_START}"
            f"{POINT_PREFIX} "
            f"{format_loc_token(128)}"
            f"{format_loc_token(0)}"
            f"\n"
            f"{OBJECT_PREFIX} "
            f"{format_obj_token(7, 3)}"
            f"{ANSWER_END}"
        )
        p = parse_structured_output_text(t, 100, coord_bins=128)
        self.assertFalse(p["valid_format"])

    def test_empty_string_invalid(self):
        p = parse_structured_output_text("", 100)
        self.assertFalse(p["valid_format"])
        self.assertFalse(p["has_extra_text"])

    def test_extra_text_detected(self):
        t = self._make(0.5, 0.5, 0) + " extra"
        p = parse_structured_output_text(t, 100)
        self.assertFalse(p["valid_format"])
        self.assertTrue(p["has_extra_text"])

    def test_prefix_extra_text(self):
        t = "prefix " + self._make(0.5, 0.5, 0)
        p = parse_structured_output_text(t, 100)
        self.assertFalse(p["valid_format"])
        self.assertTrue(p["has_extra_text"])

    def test_broken_format(self):
        p = parse_structured_output_text("<gaze_point_start><loc_500><gaze_point_end>", 100)
        self.assertFalse(p["valid_format"])

    def test_duplicate_structure(self):
        t = self._make(0.5, 0.5, 0)
        p = parse_structured_output_text(t + t, 100)
        self.assertFalse(p["valid_format"])
        self.assertTrue(p["has_extra_text"])

    def test_object_id_out_of_range(self):
        # manually construct text with obj_id = num_classes (out of range)
        from model.utils.gaze_tokens import format_loc_token, format_obj_token
        t = (
            f"{POINT_PREFIX} "
            f"{format_loc_token(0)}"
            f"{format_loc_token(0)}"
            f"\n"
            f"{OBJECT_PREFIX} "
            f"{format_obj_token(100, 3)}"
        )
        p = parse_structured_output_text(t, 100)
        self.assertFalse(p["valid_format"])

    def test_unknown_object_token_parses_as_valid_format(self):
        t = build_structured_target_text(0.5, 0.5, None, 100, obj_token=GAZE_OBJ_UNKNOWN)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertTrue(p["object_unknown"])
        self.assertIsNone(p["object_id"])

    def test_is_valid_helper(self):
        t = self._make(0.5, 0.5, 0)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(is_valid_structured_output(p))

    def test_point_xy_approx_correct(self):
        t = self._make(0.423, 0.612, 5)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        px, py = p["point_xy"]
        self.assertAlmostEqual(px, 0.423, delta=0.002)
        self.assertAlmostEqual(py, 0.612, delta=0.002)


if __name__ == "__main__":
    unittest.main()
