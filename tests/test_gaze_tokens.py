from __future__ import annotations

import unittest

from model.utils.gaze_tokens import (
    ANSWER_END,
    ANSWER_START,
    COORD_BINS,
    GAZE_OBJ_UNKNOWN,
    GAZE_OBJECT_MARKER,
    GAZE_POINT_MARKER,
    GAZE_REASONING_MARKER,
    GAZE_SCHEMA_MARKERS,
    OBJECT_PREFIX,
    POINT_PREFIX,
    REASONING_START,
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
        self.assertEqual(_obj_token_width(158), 3)
        self.assertEqual(_obj_token_width(1000), 3)
        self.assertEqual(_obj_token_width(1001), 4)
        self.assertEqual(_obj_token_width(10001), 5)


class TestBuildSpecialTokens(unittest.TestCase):
    def test_token_count(self):
        num_classes = 158
        tokens = build_gaze_special_tokens(num_classes)
        # 3 schema markers + 1000 loc + 158 obj + 1 unknown
        self.assertEqual(len(tokens), 3 + 1000 + 158 + 1)

    def test_schema_markers_present(self):
        tokens = build_gaze_special_tokens(10)
        for marker in GAZE_SCHEMA_MARKERS:
            self.assertIn(marker, tokens, f"schema marker {marker!r} missing")

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

    def test_contains_gaze_point_marker(self):
        t = build_structured_target_text(0.3, 0.7, 5, 100)
        self.assertIn(GAZE_POINT_MARKER, t)

    def test_contains_gaze_object_marker(self):
        t = build_structured_target_text(0.3, 0.7, 5, 100)
        self.assertIn(GAZE_OBJECT_MARKER, t)

    def test_no_trailing_im_end(self):
        t = build_structured_target_text(0.3, 0.7, 5, 100)
        self.assertFalse(t.startswith("<|im_start|>"))
        self.assertFalse(t.endswith("<|im_end|>"))

    def test_no_newlines_in_direct(self):
        t = build_structured_target_text(0.3, 0.7, 5, 100)
        self.assertNotIn("\n", t)

    def test_contains_obj_token(self):
        t = build_structured_target_text(0.0, 0.0, 42, 100)
        self.assertIn("<obj_042>", t)

    def test_contains_two_loc_tokens_and_one_obj_token(self):
        import re
        t = build_structured_target_text(0.5, 0.5, 0, 100)
        tokens = re.findall(r"<[^>]+>", t)
        self.assertEqual(sum(1 for tok in tokens if tok.startswith("<loc_")), 2)
        self.assertEqual(sum(1 for tok in tokens if tok.startswith("<obj_")), 1)

    def test_unknown_object_token(self):
        t = build_structured_target_text(0.5, 0.5, None, 100, obj_token=GAZE_OBJ_UNKNOWN)
        self.assertIn(GAZE_OBJ_UNKNOWN, t)

    def test_custom_coord_bins(self):
        t = build_structured_target_text(1.0, 0.0, 7, 100, coord_bins=128)
        self.assertIn("<loc_127><loc_000>", t)

    def test_direct_point_object_exact_format(self):
        """New schema: point_object produces <|gaze_point|><loc_x><loc_y><|gaze_object|><obj_k>."""
        t = build_structured_target_text(0.5, 0.5, 10, 100, target_order="point_object")
        bx = quantize_coord(0.5)
        by = quantize_coord(0.5)
        expected = f"{GAZE_POINT_MARKER}{format_loc_token(bx)}{format_loc_token(by)}{GAZE_OBJECT_MARKER}<obj_010>"
        self.assertEqual(t, expected)

    def test_direct_object_point_exact_format(self):
        """New schema: object_point produces <|gaze_object|><obj_k><|gaze_point|><loc_x><loc_y>."""
        t = build_structured_target_text(0.5, 0.5, 10, 100, target_order="object_point")
        bx = quantize_coord(0.5)
        by = quantize_coord(0.5)
        expected = f"{GAZE_OBJECT_MARKER}<obj_010>{GAZE_POINT_MARKER}{format_loc_token(bx)}{format_loc_token(by)}"
        self.assertEqual(t, expected)

    def test_reasoning_point_object_exact_format(self):
        """New schema: reasoning_point_object produces <|gaze_reasoning|>...<|gaze_point|>...<|gaze_object|>..."""
        t = build_structured_target_text(
            0.5, 0.5, 10, 100,
            target_order="reasoning_point_object",
            reasoning_text="Looking at the TV.",
        )
        self.assertTrue(t.startswith(GAZE_REASONING_MARKER))
        self.assertIn(GAZE_POINT_MARKER, t)
        self.assertIn(GAZE_OBJECT_MARKER, t)
        rsn_end = t.index(GAZE_POINT_MARKER)
        self.assertIn("Looking at the TV.", t[:rsn_end])
        self.assertLess(t.index(GAZE_POINT_MARKER), t.index(GAZE_OBJECT_MARKER))
        self.assertNotIn("\n", t)

    def test_reasoning_object_point_exact_format(self):
        """New schema: reasoning_object_point produces <|gaze_reasoning|>...<|gaze_object|>...<|gaze_point|>..."""
        t = build_structured_target_text(
            0.5, 0.5, 10, 100,
            target_order="reasoning_object_point",
            reasoning_text="Looking at the TV.",
        )
        self.assertTrue(t.startswith(GAZE_REASONING_MARKER))
        self.assertIn(GAZE_OBJECT_MARKER, t)
        self.assertIn(GAZE_POINT_MARKER, t)
        self.assertLess(t.index(GAZE_OBJECT_MARKER), t.index(GAZE_POINT_MARKER))
        self.assertNotIn("\n", t)

    def test_no_space_between_markers_and_content(self):
        """Markers must be immediately adjacent to their content (no space/newline)."""
        t = build_structured_target_text(0.5, 0.5, 10, 100, target_order="point_object")
        # <|gaze_point|> is followed immediately by <loc_...>
        pt_idx = t.index(GAZE_POINT_MARKER)
        after_pt = t[pt_idx + len(GAZE_POINT_MARKER)]
        self.assertEqual(after_pt, "<", "No space/newline between marker and loc token")


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
        # Manually construct legacy text with loc_128 out of range for 128-bin vocab
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

    def test_new_direct_object_point_valid(self):
        """ST object_point schema is parsed as valid_format=True."""
        t = f"{GAZE_OBJECT_MARKER}<obj_007>{GAZE_POINT_MARKER}<loc_499><loc_299>"
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    def test_new_direct_point_object_valid(self):
        """ST point_object schema is parsed as valid_format=True."""
        t = f"{GAZE_POINT_MARKER}<loc_499><loc_299>{GAZE_OBJECT_MARKER}<obj_007>"
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    def test_new_reasoning_object_point_valid(self):
        """ST reasoning_object_point schema is parsed as valid_format=True."""
        t = f"{GAZE_REASONING_MARKER}Looking at TV.{GAZE_OBJECT_MARKER}<obj_007>{GAZE_POINT_MARKER}<loc_499><loc_299>"
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    def test_new_reasoning_point_object_valid(self):
        """ST reasoning_point_object schema is parsed as valid_format=True."""
        t = f"{GAZE_REASONING_MARKER}Looking at TV.{GAZE_POINT_MARKER}<loc_499><loc_299>{GAZE_OBJECT_MARKER}<obj_007>"
        p = parse_structured_output_text(t, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)


class TestTargetOrders(unittest.TestCase):
    """Verify build_structured_target_text and parser round-trip for all orders."""

    def _parse(self, text: str) -> dict:
        return parse_structured_output_text(text, 100)

    # --- object_point (default) ---
    def test_object_point_order(self):
        t = build_structured_target_text(0.5, 0.3, 7, 100, target_order="object_point")
        # New schema: <|gaze_object|> before <|gaze_point|>
        self.assertLess(t.index(GAZE_OBJECT_MARKER), t.index(GAZE_POINT_MARKER))
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
        t = build_structured_target_text(
            0.5, 0.5, 10, 100,
            target_order="reasoning_object_point",
            reasoning_text="The person looks at the screen.",
        )
        # New schema: no legacy <think> tags, uses <|gaze_reasoning|> marker
        self.assertNotIn("<think>", t)
        self.assertNotIn("Reasoning:", t)
        rsn_pos = t.index(GAZE_REASONING_MARKER)
        obj_pos = t.index(GAZE_OBJECT_MARKER)
        pt_pos  = t.index(GAZE_POINT_MARKER)
        self.assertLess(rsn_pos, obj_pos)
        self.assertLess(obj_pos, pt_pos)
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 10)

    def test_reasoning_object_point_no_reasoning_falls_back_to_object_point(self):
        t = build_structured_target_text(0.5, 0.5, 5, 100, target_order="reasoning_object_point")
        # No reasoning text → falls back to direct object_point (no reasoning marker)
        self.assertNotIn("<think>", t)
        self.assertNotIn(GAZE_REASONING_MARKER, t)
        self.assertLess(t.index(GAZE_OBJECT_MARKER), t.index(GAZE_POINT_MARKER))
        p = self._parse(t)
        self.assertTrue(p["valid_format"])

    def test_reasoning_object_point_forced_empty(self):
        t = build_structured_target_text(
            0.5, 0.5, 3, 100,
            target_order="reasoning_object_point",
            force_reasoning_format=True,
        )
        # force=True with empty body → reasoning marker present, no content between markers
        self.assertNotIn("<think>", t)
        self.assertIn(GAZE_REASONING_MARKER, t)
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 3)

    # --- point_object ---
    def test_point_object_order(self):
        t = build_structured_target_text(0.5, 0.3, 7, 100, target_order="point_object")
        # New schema: <|gaze_point|> before <|gaze_object|>
        self.assertLess(t.index(GAZE_POINT_MARKER), t.index(GAZE_OBJECT_MARKER))
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 7)

    # --- point_object_reasoning (legacy post-hoc, keep old text format) ---
    def test_point_object_reasoning_order(self):
        t = build_structured_target_text(
            0.5, 0.5, 2, 100,
            target_order="point_object_reasoning",
            reasoning_text="Some reason.",
        )
        # Legacy format still uses "Point:" / "Object:" text prefix + <think> block
        self.assertLess(t.index(POINT_PREFIX), t.index(OBJECT_PREFIX))
        self.assertLess(t.index(OBJECT_PREFIX), t.index(REASONING_START))
        p = self._parse(t)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 2)

    # --- parser: all orders accepted ---
    def test_parser_accepts_all_orders(self):
        orders = [
            "object_point", "reasoning_object_point",
            "point_object", "point_object_reasoning", "reasoning_point_object",
        ]
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

    def test_parser_rejects_new_schema_missing_point(self):
        p = parse_structured_output_text(f"{GAZE_OBJECT_MARKER}<obj_010>", 100)
        self.assertFalse(p["valid_format"])

    def test_parser_rejects_new_schema_missing_object(self):
        p = parse_structured_output_text(f"{GAZE_POINT_MARKER}<loc_500><loc_500>", 100)
        self.assertFalse(p["valid_format"])


class TestSafeAugmentation(unittest.TestCase):
    """Verify apply_safe_augmentation applies no spatial transforms."""

    def test_safe_aug_preserves_coords(self):
        from model.utils.data_utils import apply_safe_augmentation
        import random
        from PIL import Image
        random.seed(0)
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        gaze_x, gaze_y = 0.3, 0.7
        bbox = (100.0, 100.0, 200.0, 200.0)
        out_img, out_x, out_y, out_bbox = apply_safe_augmentation(img, gaze_x, gaze_y, bbox)
        self.assertAlmostEqual(out_x, gaze_x, places=6)
        self.assertAlmostEqual(out_y, gaze_y, places=6)
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
