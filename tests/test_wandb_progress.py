from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.utils.wandb_utils import train_progress_bucket


class TestTrainProgressBucket(unittest.TestCase):
    def test_zero_updates_returns_zero(self) -> None:
        self.assertEqual(train_progress_bucket(0, 100), 0)

    def test_logs_first_percent_at_expected_boundary(self) -> None:
        self.assertEqual(train_progress_bucket(1, 200), 0)
        self.assertEqual(train_progress_bucket(2, 200), 1)

    def test_caps_at_hundred_percent(self) -> None:
        self.assertEqual(train_progress_bucket(100, 100), 100)
        self.assertEqual(train_progress_bucket(150, 100), 100)

    def test_small_epoch_still_progresses_monotonically(self) -> None:
        self.assertEqual(train_progress_bucket(1, 8), 12)
        self.assertEqual(train_progress_bucket(4, 8), 50)
        self.assertEqual(train_progress_bucket(8, 8), 100)


if __name__ == "__main__":
    unittest.main()
