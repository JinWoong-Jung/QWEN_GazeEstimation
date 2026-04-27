from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model.trainer as trainer


class TestResolveModelSource(unittest.TestCase):
    def test_existing_model_path_is_used_directly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            local_model = root / "model" / "Qwen3-VL-2B-Instruct"
            local_model.mkdir(parents=True)
            with patch.object(trainer, "ROOT", root):
                resolved = trainer.resolve_model_source("model/Qwen3-VL-2B-Instruct")
            self.assertEqual(resolved, str(local_model))

    def test_missing_model_path_downloads_to_requested_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = root / "model" / "Qwen3-VL-2B-Instruct"
            with patch.object(trainer, "ROOT", root), patch.object(
                trainer,
                "_download_model_to_local_dir",
                return_value=str(expected),
            ) as download:
                resolved = trainer.resolve_model_source("model/Qwen3-VL-2B-Instruct")
            self.assertEqual(resolved, str(expected))
            download.assert_called_once_with("Qwen/Qwen3-VL-2B-Instruct", expected)

    def test_repo_id_downloads_to_repo_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = root / "model" / "Qwen3-VL-2B-Instruct"
            with patch.object(trainer, "ROOT", root), patch.object(
                trainer,
                "_download_model_to_local_dir",
                return_value=str(expected),
            ) as download:
                resolved = trainer.resolve_model_source("Qwen/Qwen3-VL-2B-Instruct")
            self.assertEqual(resolved, str(expected))
            download.assert_called_once_with("Qwen/Qwen3-VL-2B-Instruct", expected)


if __name__ == "__main__":
    unittest.main()
