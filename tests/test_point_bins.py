from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.datasets import format_target_text
from model.utils.eval_utils import parse_point
from model.utils.point_tokens import point_bin_token, point_bin_value, render_point_text_human


class TestPointBins(unittest.TestCase):
    def test_format_target_text_bin_mode(self) -> None:
        text, valid = format_target_text(
            label_text="television",
            label_id=0,
            id2label={0: "television"},
            vocab2id={"television": 0},
            vocab2id_lower={"television": 0},
            num_classes=1,
            answer_template="Point: {point_x} {point_y}\nObject: {label_text}",
            fallback_target_text="unknown",
            point_x=0.25,
            point_y=0.75,
            point_decimals=4,
            point_mode="bin",
            point_bin_count=1000,
        )
        self.assertIn("Point: <pt1000_", text)
        self.assertIn("\nObject: television", text)
        self.assertEqual(valid, 1.0)

    def test_parse_point_bin_tokens(self) -> None:
        x_tok = point_bin_token(250, 1000)
        y_tok = point_bin_token(750, 1000)
        parsed = parse_point(f"Point: {x_tok} {y_tok}\nObject: chair")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertAlmostEqual(parsed[0], point_bin_value(250, 1000), places=6)
        self.assertAlmostEqual(parsed[1], point_bin_value(750, 1000), places=6)

    def test_parse_point_keeps_continuous_backward_compat(self) -> None:
        parsed = parse_point("Point: 0.2500 0.7500\nObject: chair")
        self.assertEqual(parsed, (0.25, 0.75))

    def test_render_point_text_human(self) -> None:
        x_tok = point_bin_token(250, 1000)
        y_tok = point_bin_token(750, 1000)
        rendered = render_point_text_human(f"Point: {x_tok} {y_tok}\nObject: chair")
        self.assertEqual(rendered, "Point: 0.2503 0.7508\nObject: chair")


if __name__ == "__main__":
    unittest.main()
