from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model.utils.data_utils import load_reasoning_text
from model.utils.gaze_tokens import (
    build_structured_target_text,
    build_structured_target_text_with_reasoning,
    parse_structured_output_text,
)
from model.utils.special_tokens import (
    OBJECT_START_MARKER,
    POINT_START_MARKER,
    REASONING_END_MARKER,
    REASONING_START_MARKER,
)


class TestBuildWithReasoning(unittest.TestCase):
    def test_no_reasoning_matches_base(self):
        base = build_structured_target_text(0.5, 0.5, 10, 100)
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100)
        self.assertEqual(result, base)

    def test_none_reasoning_matches_base(self):
        base = build_structured_target_text(0.5, 0.5, 10, 100)
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, None)
        self.assertEqual(result, base)

    def test_empty_reasoning_matches_base(self):
        base = build_structured_target_text(0.5, 0.5, 10, 100)
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "")
        self.assertEqual(result, base)

    def test_force_reasoning_format_with_empty_reasoning(self):
        result = build_structured_target_text_with_reasoning(
            0.5, 0.5, 10, 100, "",
            force_reasoning_format=True,
        )
        self.assertNotIn("Reasoning:", result)
        self.assertIn(REASONING_START_MARKER, result)
        self.assertIn(REASONING_END_MARKER, result)
        self.assertIn(POINT_START_MARKER, result)
        self.assertIn(OBJECT_START_MARKER, result)

    def test_with_reasoning_uses_gaze_reasoning_marker(self):
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "Looking at the TV.")
        self.assertNotIn("Reasoning:", result)
        self.assertIn(REASONING_START_MARKER, result)
        self.assertIn("Looking at the TV.", result)

    def test_reasoning_marker_before_point_and_object(self):
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "test reason")
        rsn_pos = result.index(REASONING_START_MARKER)
        pt_pos  = result.index(POINT_START_MARKER)
        obj_pos = result.index(OBJECT_START_MARKER)
        self.assertLess(rsn_pos, pt_pos)
        self.assertLess(rsn_pos, obj_pos)

    def test_reasoning_content_embedded_in_marker(self):
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "abc")
        self.assertIn(REASONING_START_MARKER + "abc", result)

    def test_base_content_preserved(self):
        # When using reasoning, the tail of the result matches the direct object_point format
        base = build_structured_target_text(0.3, 0.7, 5, 100)
        result = build_structured_target_text_with_reasoning(0.3, 0.7, 5, 100, "reason")
        self.assertTrue(result.endswith(base))

    def test_reasoning_stripped(self):
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "  padded  ")
        self.assertIn(REASONING_START_MARKER + "padded", result)

    def test_no_newlines_in_new_schema(self):
        result = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "some text")
        self.assertNotIn("\n", result)


class TestStrictReWithReasoning(unittest.TestCase):
    def test_parse_with_reasoning(self):
        text = build_structured_target_text_with_reasoning(0.5, 0.3, 10, 100, "Looking at the screen.")
        p = parse_structured_output_text(text, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 10)

    def test_parse_with_empty_reasoning_scaffold(self):
        text = build_structured_target_text_with_reasoning(
            0.5, 0.3, 10, 100, "",
            force_reasoning_format=True,
        )
        p = parse_structured_output_text(text, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 10)

    def test_parse_without_reasoning(self):
        text = build_structured_target_text(0.5, 0.3, 10, 100)
        p = parse_structured_output_text(text, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 10)

    def test_parse_multiline_reasoning_normalized(self):
        # Multiline input gets normalized (collapsed) by normalize_reasoning_text
        text = build_structured_target_text_with_reasoning(0.1, 0.9, 5, 100, "The person looks\nto the right.")
        p = parse_structured_output_text(text, 100)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 5)

    def test_parse_extracts_coords_with_reasoning(self):
        text = build_structured_target_text_with_reasoning(0.0, 1.0, 0, 100, "edge case test")
        p = parse_structured_output_text(text, 100)
        self.assertTrue(p["valid_format"])
        self.assertIsNotNone(p["point_xy"])

    def test_invalid_missing_point(self):
        text = "<|reasoning_start|>blah<|reasoning_end|><|object_start|><obj_010><|object_end|>"
        p = parse_structured_output_text(text, 100)
        self.assertFalse(p["valid_format"])

    def test_invalid_missing_object(self):
        text = "<|point_start|><loc_064><loc_064><|point_end|>"
        p = parse_structured_output_text(text, 100)
        self.assertFalse(p["valid_format"])

    def test_reasoning_with_custom_coord_bins(self):
        text = build_structured_target_text_with_reasoning(1.0, 0.0, 3, 10, "custom bins", coord_bins=128)
        p = parse_structured_output_text(text, 10, coord_bins=128)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["point_bins"], (127, 0))

    def test_new_schema_reasoning_markers_stripped_from_reasoning(self):
        """Schema markers injected into reasoning text must be removed by normalize."""
        bad_text = f"Look {POINT_START_MARKER} at screen"
        text = build_structured_target_text_with_reasoning(0.5, 0.5, 7, 100, bad_text)
        rsn_start = text.index(REASONING_START_MARKER) + len(REASONING_START_MARKER)
        next_marker = text.index(REASONING_END_MARKER)
        reasoning_body = text[rsn_start:next_marker]
        self.assertNotIn(POINT_START_MARKER, reasoning_body)
        p = parse_structured_output_text(text, 100)
        self.assertTrue(p["valid_format"])


class TestLoadReasoningText(unittest.TestCase):
    def test_normal_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Object: a television set\nReasoning: The person is looking at the TV.\n")
            path = Path(f.name)
        try:
            result = load_reasoning_text(path)
            self.assertEqual(result, "The person is looking at the TV.")
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found(self):
        result = load_reasoning_text(Path("/nonexistent/path_xyz_12345.txt"))
        self.assertIsNone(result)

    def test_no_reasoning_key_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Object: television\n")
            path = Path(f.name)
        try:
            result = load_reasoning_text(path)
            self.assertIsNone(result)
        finally:
            path.unlink(missing_ok=True)

    def test_reasoning_stripped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Reasoning:   some text with spaces   \n")
            path = Path(f.name)
        try:
            result = load_reasoning_text(path)
            self.assertEqual(result, "some text with spaces")
        finally:
            path.unlink(missing_ok=True)

    def test_only_reasoning_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Reasoning: some reasoning\n")
            path = Path(f.name)
        try:
            result = load_reasoning_text(path)
            self.assertEqual(result, "some reasoning")
        finally:
            path.unlink(missing_ok=True)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("")
            path = Path(f.name)
        try:
            result = load_reasoning_text(path)
            self.assertIsNone(result)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
