"""Tests for sample_mode dataset construction.

Verifies:
1. direct_only: all samples use point_object target order, no reasoning markers.
2. reasoning_only (via GazeDataset): capable records → full reasoning_point_object schema.
3. direct&reasoning: total len == N, cap handling keeps total at N, correct schemas.
4. direct+reasoning: n_direct == N, total > N, only reasoning views resampled per epoch.
5. val/test force_reasoning_format=True: target contains reasoning markers + point + object.
6. Epoch resample behaviour for both MultiViewGazeDataset modes.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from model.datasets import GazeDataset, MultiViewGazeDataset
from model.utils.special_tokens import (
    OBJECT_END_MARKER,
    OBJECT_START_MARKER,
    POINT_END_MARKER,
    POINT_START_MARKER,
    REASONING_END_MARKER,
    REASONING_START_MARKER,
)


# ---------------------------------------------------------------------------
# Minimal Record stub (mirrors model/utils/data_utils.py:Record)
# ---------------------------------------------------------------------------
@dataclass
class _Record:
    sample_id: int
    image_rel: str
    image_path: Path
    gaze_x: float
    gaze_y: float
    bbox_px: tuple
    label_id: int
    label_text: str = ""


def _make_image(tmp: Path, rel: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(str(p))
    return p


def _make_records(tmp: Path, n: int) -> list[_Record]:
    recs = []
    for i in range(n):
        rel = f"scene/img_{i:04d}.jpg"
        path = _make_image(tmp, rel)
        recs.append(
            _Record(
                sample_id=i,
                image_rel=rel,
                image_path=path,
                gaze_x=0.5,
                gaze_y=0.5,
                bbox_px=(4.0, 4.0, 12.0, 12.0),
                label_id=0,
                label_text="computer",
            )
        )
    return recs


def _make_reasoning_index(tmp: Path, records: list[_Record], n_capable: int) -> dict[str, Path]:
    """Create reasoning text files for the first n_capable records."""
    rdir = tmp / "reasoning"
    rdir.mkdir(exist_ok=True)
    index: dict[str, Path] = {}
    for rec in records[:n_capable]:
        folder = Path(rec.image_rel).parent.name
        stem = Path(rec.image_rel).stem
        key = f"{folder}/{stem}_{rec.sample_id}"
        p = rdir / f"{folder}" / f"{stem}_{rec.sample_id}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("Reasoning: The person is looking at the computer screen.", encoding="utf-8")
        index[key] = p
    return index


_VOCAB2ID = {"computer": 0, "phone": 1, "tv": 2}
_ID2LABEL = {0: "computer", 1: "phone", 2: "tv"}
_VOCAB2ID_LOWER = {k.lower(): v for k, v in _VOCAB2ID.items()}
_NUM_CLASSES = len(_VOCAB2ID)
_COORD_BINS = 16
_PROMPT_TEMPLATE = ""
_PROMPT_DIRECT = "Predict gaze. Return: <|point_start|><loc_XXX><loc_YYY><|point_end|><|object_start|><obj_KKK><|object_end|>"
_PROMPT_REASONING = (
    "Reason then predict. Return: "
    "<|reasoning_start|><your reasoning here><|reasoning_end|>"
    "<|point_start|><loc_XXX><loc_YYY><|point_end|><|object_start|><obj_KKK><|object_end|>"
)


def _has_reasoning(text: str) -> bool:
    return REASONING_START_MARKER in text


def _has_point(text: str) -> bool:
    return POINT_START_MARKER in text


def _has_object(text: str) -> bool:
    return OBJECT_START_MARKER in text


def _is_direct_schema(text: str) -> bool:
    return _has_point(text) and _has_object(text) and not _has_reasoning(text)


def _is_full_schema(text: str) -> bool:
    return _has_reasoning(text) and _has_point(text) and _has_object(text)


class TestDirectOnlyGazeDataset(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_dir.name)
        self.records = _make_records(self.tmp, 10)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_all_samples_are_direct_schema(self):
        ds = GazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text=_PROMPT_DIRECT,
            apply_augmentation=False,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            target_order="point_object",
        )
        self.assertEqual(len(ds), 10)
        for i in range(len(ds)):
            target = ds[i]["target_text"]
            self.assertTrue(_is_direct_schema(target), f"sample {i} target: {target!r}")

    def test_point_and_object_valid_are_one(self):
        ds = GazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text=_PROMPT_DIRECT,
            apply_augmentation=False,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            target_order="point_object",
        )
        for i in range(len(ds)):
            item = ds[i]
            self.assertAlmostEqual(float(item["target_point_valid"].item()), 1.0)
            self.assertAlmostEqual(float(item["target_object_valid"].item()), 1.0)


class TestReasoningOnlyGazeDataset(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_dir.name)
        self.records = _make_records(self.tmp, 8)
        self.reasoning_index = _make_reasoning_index(self.tmp, self.records, n_capable=8)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_all_samples_have_full_schema(self):
        ds = GazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text=_PROMPT_REASONING,
            apply_augmentation=False,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            reasoning_index=self.reasoning_index,
            target_order="reasoning_point_object",
            force_reasoning_format=True,
        )
        self.assertEqual(len(ds), 8)
        for i in range(len(ds)):
            target = ds[i]["target_text"]
            self.assertTrue(_is_full_schema(target), f"sample {i} target: {target!r}")

    def test_prompt_text_is_reasoning_prompt(self):
        ds = GazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text=_PROMPT_REASONING,
            apply_augmentation=False,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            reasoning_index=self.reasoning_index,
            target_order="reasoning_point_object",
            force_reasoning_format=True,
        )
        for i in range(len(ds)):
            prompt = ds[i]["text_input"]
            self.assertIn("reasoning", prompt.lower())

    def test_reasoning_text_is_loaded_lazily(self):
        with patch("model.datasets.load_reasoning_text", return_value="The person is looking at the screen.") as mocked:
            ds = GazeDataset(
                records=self.records,
                prompt_template=_PROMPT_TEMPLATE,
                prompt_text=_PROMPT_REASONING,
                apply_augmentation=False,
                id2label=_ID2LABEL,
                vocab2id=_VOCAB2ID,
                vocab2id_lower=_VOCAB2ID_LOWER,
                num_classes=_NUM_CLASSES,
                coord_bins=_COORD_BINS,
                reasoning_index=self.reasoning_index,
                target_order="reasoning_point_object",
                force_reasoning_format=True,
            )
            self.assertEqual(mocked.call_count, 0)

            item = ds[0]
            self.assertTrue(item["has_reasoning"])
            self.assertEqual(mocked.call_count, 1)

            _ = ds[0]
            self.assertEqual(mocked.call_count, 1)


class TestMultiViewDirectAndReasoning(unittest.TestCase):
    """direct&reasoning: total == N, schemas correct."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_dir.name)
        self.N = 20
        self.records = _make_records(self.tmp, self.N)
        # All 20 records have reasoning files
        self.reasoning_index = _make_reasoning_index(self.tmp, self.records, n_capable=self.N)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _make_ds(self, ratio: float = 0.3, n_capable: int | None = None) -> MultiViewGazeDataset:
        index = (
            _make_reasoning_index(self.tmp, self.records, n_capable=n_capable)
            if n_capable is not None
            else self.reasoning_index
        )
        return MultiViewGazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text_direct=_PROMPT_DIRECT,
            prompt_text_reasoning=_PROMPT_REASONING,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            reasoning_index=index,
            reasoning_ratio=ratio,
            sample_mode="direct&reasoning",
            seed=0,
        )

    def test_total_length_equals_N(self):
        ds = self._make_ds(ratio=0.3)
        self.assertEqual(len(ds), self.N)

    def test_total_length_equals_N_when_cap_applied(self):
        # Only 4 capable but ratio would need 6 → cap → direct adjusts upward
        ds = self._make_ds(ratio=0.3, n_capable=4)
        self.assertEqual(len(ds), self.N)

    def test_view_counts_sum_to_N(self):
        ds = self._make_ds(ratio=0.4)
        n_d, n_r = ds.get_view_counts()
        self.assertEqual(n_d + n_r, self.N)

    def test_reasoning_views_have_full_schema(self):
        ds = self._make_ds(ratio=0.3)
        for i in range(len(ds)):
            item = ds[i]
            if item.get("view_type") == "reasoning":
                self.assertTrue(
                    _is_full_schema(item["target_text"]),
                    f"reasoning view {i} target: {item['target_text']!r}",
                )

    def test_direct_views_have_direct_schema(self):
        ds = self._make_ds(ratio=0.3)
        for i in range(len(ds)):
            item = ds[i]
            if item.get("view_type") == "direct":
                self.assertTrue(
                    _is_direct_schema(item["target_text"]),
                    f"direct view {i} target: {item['target_text']!r}",
                )

    def test_reasoning_views_point_object_valid_are_one(self):
        ds = self._make_ds(ratio=0.3)
        for i in range(len(ds)):
            item = ds[i]
            if item.get("view_type") == "reasoning":
                self.assertAlmostEqual(float(item["target_point_valid"].item()), 1.0,
                                       msg=f"reasoning view {i} point_valid should be 1.0")
                self.assertAlmostEqual(float(item["target_object_valid"].item()), 1.0,
                                       msg=f"reasoning view {i} object_valid should be 1.0")

    def test_resample_keeps_total_N(self):
        ds = self._make_ds(ratio=0.3)
        for _ in range(3):
            ds.resample_epoch_views()
            n_d, n_r = ds.get_view_counts()
            self.assertEqual(n_d + n_r, self.N, "total must stay N after resample")

    def test_reasoning_text_is_loaded_lazily(self):
        with patch("model.datasets.load_reasoning_text", return_value="The person is looking at the screen.") as mocked:
            ds = self._make_ds(ratio=0.3)
            self.assertEqual(mocked.call_count, 0)

            reasoning_idx = next(i for i in range(len(ds)) if ds._views[i][1] == "reasoning")
            item = ds[reasoning_idx]
            self.assertEqual(item["view_type"], "reasoning")
            self.assertTrue(item["has_reasoning"])
            self.assertEqual(mocked.call_count, 1)

            _ = ds[reasoning_idx]
            self.assertEqual(mocked.call_count, 1)


class TestMultiViewDirectPlusReasoning(unittest.TestCase):
    """direct+reasoning: n_direct == N, total > N, only reasoning resampled."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_dir.name)
        self.N = 20
        self.records = _make_records(self.tmp, self.N)
        self.reasoning_index = _make_reasoning_index(self.tmp, self.records, n_capable=self.N)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _make_ds(self, ratio: float = 0.3) -> MultiViewGazeDataset:
        return MultiViewGazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text_direct=_PROMPT_DIRECT,
            prompt_text_reasoning=_PROMPT_REASONING,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            reasoning_index=self.reasoning_index,
            reasoning_ratio=ratio,
            sample_mode="direct+reasoning",
            seed=0,
        )

    def test_n_direct_equals_N(self):
        ds = self._make_ds(ratio=0.3)
        n_d, _ = ds.get_view_counts()
        self.assertEqual(n_d, self.N)

    def test_total_greater_than_N(self):
        ds = self._make_ds(ratio=0.3)
        self.assertGreater(len(ds), self.N)

    def test_total_equals_N_plus_reasoning(self):
        ratio = 0.25
        ds = self._make_ds(ratio=ratio)
        n_d, n_r = ds.get_view_counts()
        self.assertEqual(n_d, self.N)
        expected_r = min(round(self.N * ratio), self.N)
        self.assertEqual(n_r, expected_r)

    def test_resample_keeps_n_direct_fixed(self):
        ds = self._make_ds(ratio=0.3)
        for _ in range(3):
            n_d_before, _ = ds.get_view_counts()
            ds.resample_epoch_views()
            n_d_after, _ = ds.get_view_counts()
            self.assertEqual(n_d_before, n_d_after, "direct views must not change after resample")

    def test_reasoning_views_have_full_schema(self):
        ds = self._make_ds(ratio=0.4)
        for i in range(len(ds)):
            item = ds[i]
            if item.get("view_type") == "reasoning":
                self.assertTrue(
                    _is_full_schema(item["target_text"]),
                    f"reasoning view {i} target: {item['target_text']!r}",
                )

    def test_direct_views_have_direct_schema(self):
        ds = self._make_ds(ratio=0.4)
        for i in range(len(ds)):
            item = ds[i]
            if item.get("view_type") == "direct":
                self.assertTrue(
                    _is_direct_schema(item["target_text"]),
                    f"direct view {i} target: {item['target_text']!r}",
                )


class TestValTestForceReasoningFormat(unittest.TestCase):
    """GazeDataset with force_reasoning_format=True (val/test reasoning mode)."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_dir.name)
        self.records = _make_records(self.tmp, 5)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_target_has_reasoning_markers_even_without_gt(self):
        # No reasoning_index provided (val/test has no GT reasoning)
        ds = GazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text=_PROMPT_REASONING,
            apply_augmentation=False,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            target_order="reasoning_point_object",
            force_reasoning_format=True,
        )
        for i in range(len(ds)):
            target = ds[i]["target_text"]
            self.assertIn(REASONING_START_MARKER, target,
                          f"sample {i} should have reasoning start: {target!r}")
            self.assertIn(REASONING_END_MARKER, target,
                          f"sample {i} should have reasoning end: {target!r}")
            self.assertIn(POINT_START_MARKER, target,
                          f"sample {i} should have point: {target!r}")
            self.assertIn(OBJECT_START_MARKER, target,
                          f"sample {i} should have object: {target!r}")

    def test_target_reasoning_comes_before_point(self):
        ds = GazeDataset(
            records=self.records,
            prompt_template=_PROMPT_TEMPLATE,
            prompt_text=_PROMPT_REASONING,
            apply_augmentation=False,
            id2label=_ID2LABEL,
            vocab2id=_VOCAB2ID,
            vocab2id_lower=_VOCAB2ID_LOWER,
            num_classes=_NUM_CLASSES,
            coord_bins=_COORD_BINS,
            target_order="reasoning_point_object",
            force_reasoning_format=True,
        )
        for i in range(len(ds)):
            target = ds[i]["target_text"]
            pos_rsn = target.find(REASONING_START_MARKER)
            pos_pt = target.find(POINT_START_MARKER)
            self.assertLess(pos_rsn, pos_pt,
                            f"sample {i}: reasoning must come before point in {target!r}")


class TestMultiViewInvalidMode(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_dir.name)
        self.records = _make_records(self.tmp, 4)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_invalid_sample_mode_raises(self):
        with self.assertRaises(ValueError):
            MultiViewGazeDataset(
                records=self.records,
                prompt_template=_PROMPT_TEMPLATE,
                prompt_text_direct=_PROMPT_DIRECT,
                prompt_text_reasoning=_PROMPT_REASONING,
                id2label=_ID2LABEL,
                vocab2id=_VOCAB2ID,
                vocab2id_lower=_VOCAB2ID_LOWER,
                num_classes=_NUM_CLASSES,
                coord_bins=_COORD_BINS,
                sample_mode="direct_only",  # invalid for MultiViewGazeDataset
            )


if __name__ == "__main__":
    unittest.main()
