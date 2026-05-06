from __future__ import annotations

import unittest

from model.utils.gaze_tokens import (
    COORD_BINS,
    GAZE_OBJ_UNKNOWN,
    build_gaze_special_tokens,
    build_structured_target_text,
    dequantize_coord,
    format_loc_token,
    format_obj_token,
    is_valid_structured_output,
    parse_structured_output_text,
    quantize_coord,
)
from model.utils.special_tokens import (
    OBJECT_END_MARKER,
    OBJECT_START_MARKER,
    POINT_END_MARKER,
    POINT_START_MARKER,
    REASONING_END_MARKER,
    REASONING_START_MARKER,
    GAZE_SCHEMA_MARKERS,
)


class TestQuantize(unittest.TestCase):
    def test_boundary_zero(self):
        self.assertEqual(quantize_coord(0.0), 0)

    def test_boundary_one(self):
        self.assertEqual(quantize_coord(1.0), 999)

    def test_roundtrip_arbitrary(self):
        for v in [0.0, 0.123, 0.5, 0.999, 1.0]:
            b = quantize_coord(v)
            rec = dequantize_coord(b)
            self.assertAlmostEqual(rec, max(0.0, min(1.0, round(v * 999) / 999)), places=6)

    def test_custom_bin_count(self):
        self.assertEqual(quantize_coord(1.0, bins=128), 127)
        self.assertEqual(dequantize_coord(127, bins=128), 1.0)


class TestFormatTokens(unittest.TestCase):
    def test_loc_token_format(self):
        self.assertEqual(format_loc_token(0), "<loc_000>")
        self.assertEqual(format_loc_token(42), "<loc_042>")
        self.assertEqual(format_loc_token(999), "<loc_999>")

    def test_obj_token_format(self):
        self.assertEqual(format_obj_token(0, 3), "<obj_000>")
        self.assertEqual(format_obj_token(157, 3), "<obj_157>")


class TestBuildSpecialTokens(unittest.TestCase):
    def test_token_count(self):
        num_classes = 158
        tokens = build_gaze_special_tokens(num_classes)
        self.assertEqual(len(tokens), 6 + COORD_BINS + num_classes + 1)

    def test_schema_markers_present(self):
        tokens = build_gaze_special_tokens(10)
        for marker in GAZE_SCHEMA_MARKERS:
            self.assertIn(marker, tokens)

    def test_custom_loc_tokens_range(self):
        tokens = build_gaze_special_tokens(10, coord_bins=128)
        self.assertEqual(len([tok for tok in tokens if tok.startswith("<loc_")]), 128)
        self.assertIn("<loc_000>", tokens)
        self.assertIn("<loc_127>", tokens)
        self.assertNotIn("<loc_128>", tokens)

    def test_unknown_object_token_present(self):
        self.assertIn(GAZE_OBJ_UNKNOWN, build_gaze_special_tokens(5))


class TestBuildStructuredTargetText(unittest.TestCase):
    def test_point_object_exact_format(self):
        t = build_structured_target_text(0.5, 0.5, 10, 100, target_order="point_object")
        bx = quantize_coord(0.5)
        by = quantize_coord(0.5)
        expected = (
            f"{POINT_START_MARKER}{format_loc_token(bx)}{format_loc_token(by)}{POINT_END_MARKER}"
            f"{OBJECT_START_MARKER}<obj_010>{OBJECT_END_MARKER}"
        )
        self.assertEqual(t, expected)

    def test_object_point_exact_format(self):
        t = build_structured_target_text(0.5, 0.5, 10, 100, target_order="object_point")
        bx = quantize_coord(0.5)
        by = quantize_coord(0.5)
        expected = (
            f"{OBJECT_START_MARKER}<obj_010>{OBJECT_END_MARKER}"
            f"{POINT_START_MARKER}{format_loc_token(bx)}{format_loc_token(by)}{POINT_END_MARKER}"
        )
        self.assertEqual(t, expected)

    def test_reasoning_point_object_format(self):
        t = build_structured_target_text(
            0.5,
            0.5,
            10,
            100,
            target_order="reasoning_point_object",
            reasoning_text="Looking at the TV.",
        )
        self.assertTrue(t.startswith(REASONING_START_MARKER))
        self.assertIn(REASONING_END_MARKER, t)
        self.assertLess(t.index(POINT_START_MARKER), t.index(OBJECT_START_MARKER))

    def test_reasoning_only_format(self):
        t = build_structured_target_text(
            0.5,
            0.5,
            10,
            100,
            target_order="reasoning_only",
            reasoning_text="some reason",
        )
        self.assertEqual(t, f"{REASONING_START_MARKER}some reason.{REASONING_END_MARKER}")

    def test_unknown_object_token(self):
        t = build_structured_target_text(0.5, 0.5, None, 100, obj_token=GAZE_OBJ_UNKNOWN)
        self.assertIn(GAZE_OBJ_UNKNOWN, t)

    def test_unsupported_order_raises(self):
        with self.assertRaises(ValueError):
            build_structured_target_text(0.5, 0.5, 10, 100, target_order="unsupported_order")


class TestParseStructuredOutputText(unittest.TestCase):
    def _make(self, x: float, y: float, obj: int, nc: int = 100) -> str:
        return build_structured_target_text(x, y, obj, nc, target_order="point_object")

    def test_roundtrip_point_object(self):
        t = self._make(0.5, 0.3, 7)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)
        self.assertIsNotNone(p["point_xy"])

    def test_roundtrip_object_point(self):
        t = build_structured_target_text(0.5, 0.3, 7, 100, target_order="object_point")
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    def test_roundtrip_reasoning_object_point(self):
        t = build_structured_target_text(
            0.5,
            0.3,
            7,
            100,
            target_order="reasoning_object_point",
            reasoning_text="Looking at TV.",
        )
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    def test_custom_coord_bins_roundtrip(self):
        t = build_structured_target_text(1.0, 0.0, 7, 100, coord_bins=128)
        p = parse_structured_output_text(t, 100, coord_bins=128)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["point_bins"], (127, 0))

    def test_extra_text_detected(self):
        p = parse_structured_output_text(self._make(0.5, 0.5, 0) + " extra", 100)
        self.assertFalse(p["valid_format"])
        self.assertTrue(p["has_extra_text"])

    def test_unknown_marker_schema_invalid(self):
        p = parse_structured_output_text("<point><loc_500><loc_500></point><object><obj_001></object>", 100)
        self.assertFalse(p["valid_format"])

    def test_text_legacy_schema_invalid(self):
        p = parse_structured_output_text("<point><loc_500><loc_500></point><object><obj_001></object>", 100)
        self.assertFalse(p["valid_format"])

    def test_unknown_object_token_valid(self):
        t = build_structured_target_text(0.5, 0.5, None, 100, obj_token=GAZE_OBJ_UNKNOWN)
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertTrue(p["object_unknown"])
        self.assertIsNone(p["object_id"])

    def test_is_valid_helper(self):
        self.assertTrue(is_valid_structured_output(parse_structured_output_text(self._make(0.5, 0.5, 0), 100)))


if __name__ == "__main__":
    unittest.main()
