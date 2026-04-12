from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.utils.config_parser import build_parser, output_dir_from_run_name


class TestRunNameConfig(unittest.TestCase):
    def test_parser_uses_run_name_default(self) -> None:
        parser = build_parser(
            defaults={
                "run_name": "exp_alpha",
                "output_dir": "checkpoints",
            }
        )
        args = parser.parse_args([])
        self.assertEqual(args.run_name, "exp_alpha")
        self.assertEqual(args.wandb_run_name, "exp_alpha")

    def test_output_dir_replaces_existing_leaf_with_run_name(self) -> None:
        self.assertEqual(
            output_dir_from_run_name("checkpoints/qwen_ge", "exp_beta"),
            "checkpoints/exp_beta",
        )

    def test_output_dir_appends_run_name_when_given_parent_dir(self) -> None:
        self.assertEqual(
            output_dir_from_run_name("checkpoints", "exp_gamma"),
            "checkpoints/exp_gamma",
        )


if __name__ == "__main__":
    unittest.main()
