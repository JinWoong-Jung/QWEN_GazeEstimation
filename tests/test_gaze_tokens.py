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


class TestTargetOrders(unittest.TestCase):
    """Verify build_structured_target_text and parser round-trip for all four orders."""

    def _parse(self, text: str) -> dict:
        return parse_structured_output_text(text, 100)

    # --- object_point (new default) ---
    def test_object_point_order(self):
        t = build_structured_target_text(0.5, 0.3, 7, 100, target_order="object_point")
        self.assertTrue(t.index(OBJECT_PREFIX) < t.index(POINT_PREFIX))
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    def test_object_point_roundtrip_coords(self):
        t = build_structured_target_text(0.25, 0.75, 42, 100, target_order="object_point")
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        px, py = p["point_xy"]
        self.assertAlmostEqual(px, 0.25, delta=0.002)
        self.assertAlmostEqual(py, 0.75, delta=0.002)

    # --- reasoning_object_point ---
    def test_reasoning_object_point_order(self):
        from model.utils.gaze_tokens import REASONING_START, REASONING_END
        t = build_structured_target_text(
            0.5, 0.5, 10, 100,
            target_order="reasoning_object_point",
            reasoning_text="The person looks at the screen.",
        )
        think_pos = t.index(REASONING_START)
        obj_pos = t.index(OBJECT_PREFIX)
        pt_pos = t.index(POINT_PREFIX)
        self.assertLess(think_pos, obj_pos)
        self.assertLess(obj_pos, pt_pos)
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 10)

    def test_reasoning_object_point_no_reasoning_falls_back_to_object_point(self):
        t = build_structured_target_text(0.5, 0.5, 5, 100, target_order="reasoning_object_point")
        # No reasoning text → falls back to object_point (no <think> block)
        self.assertNotIn("<think>", t)
        self.assertTrue(t.index(OBJECT_PREFIX) < t.index(POINT_PREFIX))
        p = self._parse(t)
        self.assertTrue(p["valid_format"])

    def test_reasoning_object_point_forced_empty(self):
        from model.utils.gaze_tokens import REASONING_START, REASONING_END
        t = build_structured_target_text(
            0.5, 0.5, 3, 100,
            target_order="reasoning_object_point",
            force_reasoning_format=True,
        )
        self.assertIn(REASONING_START, t)
        self.assertIn("Reasoning:", t)
        self.assertIn(REASONING_END, t)
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 3)

    # --- point_object (legacy) ---
    def test_point_object_order(self):
        t = build_structured_target_text(0.5, 0.3, 7, 100, target_order="point_object")
        self.assertTrue(t.index(POINT_PREFIX) < t.index(OBJECT_PREFIX))
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    # --- point_object_reasoning (legacy post-hoc) ---
    def test_point_object_reasoning_order(self):
        from model.utils.gaze_tokens import REASONING_START
        t = build_structured_target_text(
            0.5, 0.5, 2, 100,
            target_order="point_object_reasoning",
            reasoning_text="Some reason.",
        )
        self.assertLess(t.index(POINT_PREFIX), t.index(OBJECT_PREFIX))
        self.assertLess(t.index(OBJECT_PREFIX), t.index(REASONING_START))
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 2)

    # --- parser: masks disjoint, all orders accepted ---
    def test_parser_accepts_all_orders(self):
        orders = ["object_point", "reasoning_object_point", "point_object", "point_object_reasoning"]
        for order in orders:
            t = build_structured_target_text(
                0.4, 0.6, 15, 100,
                target_order=order,
                reasoning_text="reason" if "reasoning" in order else None,
            )
            p = self._parse(t)
            self.assertTrue(p["valid_format"], f"Failed for target_order={order!r}")
            self.assertEqual(p["object_id"], 15, f"Wrong object_id for target_order={order!r}")

    def test_parser_rejects_missing_point(self):
        p = parse_structured_output_text("Object: <obj_010>", 100)
        self.assertFalse(p["valid_format"])

    def test_parser_rejects_missing_object(self):
        p = parse_structured_output_text("Point: <loc_500><loc_500>", 100)
        self.assertFalse(p["valid_format"])


class TestSafeAugmentation(unittest.TestCase):
    """Verify apply_safe_augmentation applies no spatial transforms."""

    def test_safe_aug_preserves_coords(self):
        from model.utils.data_utils import apply_safe_augmentation
        import random
        from PIL import Image
        # Deterministic seed: safe aug uses random for color jitter probability.
        random.seed(0)
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        gaze_x, gaze_y = 0.3, 0.7
        bbox = (100.0, 100.0, 200.0, 200.0)
        out_img, out_x, out_y, out_bbox = apply_safe_augmentation(img, gaze_x, gaze_y, bbox)
        # Coords must be unchanged (no crop/flip)
        self.assertAlmostEqual(out_x, gaze_x, places=6)
        self.assertAlmostEqual(out_y, gaze_y, places=6)
        # Bbox should be clamped/unchanged in safe aug
        self.assertEqual(out_img.size, (512, 512))


class TestLoadReasoningRecord(unittest.TestCase):
    """Verify load_reasoning_record parses both Object: and Reasoning: lines."""

    def test_parses_both_lines(self):
        import tempfile
        from pathlib import Path
        from model.utils.data_utils import load_reasoning_record
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Object: the television screen\nReasoning: The person looks at the TV.\n")
            path = Path(f.name)
        try:
            rec = load_reasoning_record(path)
            self.assertEqual(rec["object_text"], "the television screen")
            self.assertEqual(rec["reasoning_text"], "The person looks at the TV.")
        finally:
            path.unlink(missing_ok=True)

    def test_missing_object_line(self):
        import tempfile
        from pathlib import Path
        from model.utils.data_utils import load_reasoning_record
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Reasoning: Looking left.\n")
            path = Path(f.name)
        try:
            rec = load_reasoning_record(path)
            self.assertIsNone(rec["object_text"])
            self.assertEqual(rec["reasoning_text"], "Looking left.")
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found_returns_nones(self):
        from pathlib import Path
        from model.utils.data_utils import load_reasoning_record
        rec = load_reasoning_record(Path("/nonexistent/xyz_12345.txt"))
        self.assertIsNone(rec["object_text"])
        self.assertIsNone(rec["reasoning_text"])


if __name__ == "__main__":
    unittest.main()
