from __future__ import annotations

import math
import unittest

from model.utils.gaze_tokens import (
    build_structured_target_text,
    parse_structured_output_text,
)
from model.utils.eval_utils import l2_stats, valid_label_ids
import torch


class TestParseInEvalContext(unittest.TestCase):
    """Integration tests matching what run_test_metrics would process."""

    NUM_CLASSES = 100

    def _target(self, x: float, y: float, obj: int) -> str:
        return build_structured_target_text(x, y, obj, self.NUM_CLASSES)

    def test_correct_prediction_all_metrics(self):
        target = self._target(0.5, 0.5, 42)
        p = parse_structured_output_text(target, self.NUM_CLASSES)
        self.assertTrue(p["valid_format"])
        self.assertFalse(p["has_extra_text"])
        self.assertEqual(p["object_id"], 42)
        px, py = p["point_xy"]
        self.assertAlmostEqual(px, 0.5, delta=0.002)
        self.assertAlmostEqual(py, 0.5, delta=0.002)

    def test_wrong_object(self):
        pred = self._target(0.5, 0.5, 10)
        p = parse_structured_output_text(pred, self.NUM_CLASSES)
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["object_id"], 10)
        # object_acc would be 0 for gt_obj=42
        self.assertNotEqual(p["object_id"], 42)

    def test_format_invalid_no_count(self):
        p = parse_structured_output_text("Point: 0.5 0.5\nObject: television", self.NUM_CLASSES)
        self.assertFalse(p["valid_format"])

    def test_extra_text_flag(self):
        target = self._target(0.3, 0.7, 5)
        pred = target + "\nsome extra text"
        p = parse_structured_output_text(pred, self.NUM_CLASSES)
        self.assertFalse(p["valid_format"])
        self.assertTrue(p["has_extra_text"])


class TestMetricCalculations(unittest.TestCase):
    """Verify that L2 and binary metrics compute correctly given parsed output."""

    def test_point_l2_correct_prediction(self):
        pred_xy = (0.5, 0.5)
        gt_points = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        stats = l2_stats(pred_xy, gt_points)
        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats[0], 0.0, places=5)

    def test_point_l2_off_prediction(self):
        pred_xy = (0.0, 0.0)
        gt_points = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        stats = l2_stats(pred_xy, gt_points)
        self.assertIsNotNone(stats)
        expected = math.sqrt(2.0)
        self.assertAlmostEqual(stats[0], expected, places=5)

    def test_point_l2_multi_gt_avg(self):
        pred_xy = (0.5, 0.5)
        gt_points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        stats = l2_stats(pred_xy, gt_points)
        self.assertIsNotNone(stats)
        expected_avg = math.sqrt(0.5)
        self.assertAlmostEqual(stats[0], expected_avg, places=5)
        expected_min = math.sqrt(0.5)
        self.assertAlmostEqual(stats[1], expected_min, places=5)

    def test_l2_stats_empty_returns_none(self):
        pred_xy = (0.5, 0.5)
        gt_points = torch.zeros((0, 2))
        stats = l2_stats(pred_xy, gt_points)
        self.assertIsNone(stats)

    def test_valid_label_ids_filters_padding_and_deduplicates(self):
        label_ids = torch.tensor([7, -1, 3, 7, -1], dtype=torch.long)
        self.assertEqual(valid_label_ids(label_ids), [7, 3])

    def test_multi_acc_membership_semantics(self):
        parsed = {"object_id": 9}
        gt_obj_ids = valid_label_ids(torch.tensor([4, 9, -1], dtype=torch.long))
        self.assertIn(int(parsed["object_id"]), gt_obj_ids)


class TestFormatValidMetric(unittest.TestCase):
    NUM_CLASSES = 50

    def test_valid_count(self):
        target = build_structured_target_text(0.3, 0.7, 10, self.NUM_CLASSES)
        preds = [target, "bad output", target + " extra", target]
        valid = sum(1 for p in preds if parse_structured_output_text(p, self.NUM_CLASSES)["valid_format"])
        self.assertEqual(valid, 2)  # only 1st and 4th are valid

    def test_extra_text_rate(self):
        target = build_structured_target_text(0.5, 0.5, 0, self.NUM_CLASSES)
        preds = [
            target,               # valid
            target + " trailing", # extra text
            "something else",     # has_extra_text = True
            "",                   # has_extra_text = False (empty)
        ]
        extra = sum(
            1 for p in preds
            if parse_structured_output_text(p, self.NUM_CLASSES)["has_extra_text"]
        )
        self.assertEqual(extra, 2)  # 2nd and 3rd have extra text


class TestGazeObjEndStoppingCriteria(unittest.TestCase):
    def test_stops_when_obj_end_present(self):
        from model.utils.eval_utils import _GazeObjEndStoppingCriteria
        criteria = _GazeObjEndStoppingCriteria(prompt_len=5, obj_end_id=999)
        # Simulate: prompt tokens [0..4] + generated tokens [5..]
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 100, 200, 999]], dtype=torch.long)
        result = criteria(input_ids, None)
        self.assertTrue(result)

    def test_does_not_stop_without_obj_end(self):
        from model.utils.eval_utils import _GazeObjEndStoppingCriteria
        criteria = _GazeObjEndStoppingCriteria(prompt_len=5, obj_end_id=999)
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 100, 200, 300]], dtype=torch.long)
        result = criteria(input_ids, None)
        self.assertFalse(result)

    def test_requires_all_beams(self):
        from model.utils.eval_utils import _GazeObjEndStoppingCriteria
        criteria = _GazeObjEndStoppingCriteria(prompt_len=3, obj_end_id=999)
        # Two beams: one has obj_end, the other does not
        input_ids = torch.tensor([
            [0, 1, 2, 100, 999],
            [0, 1, 2, 100, 200],
        ], dtype=torch.long)
        result = criteria(input_ids, None)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
