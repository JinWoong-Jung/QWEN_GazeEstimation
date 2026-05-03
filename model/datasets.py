from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .utils.data_utils import (
    Record,
    TestGroup,
    apply_train_augmentation,
    apply_safe_augmentation,
    alias_label_text,
    build_prompt,
    sanitize_bbox_pixels,
    load_reasoning_text,
)
from .utils.gaze_tokens import (
    GAZE_OBJ_UNKNOWN,
    build_structured_target_text,
    quantize_coord,
)


class _ImageLRUCache:
    """LRU cache for decoded PIL images, keyed by file path string."""

    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._maxsize = max(1, int(maxsize))

    def get(self, path: str) -> Image.Image | None:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path].copy()
        return None

    def put(self, path: str, img: Image.Image) -> None:
        if path in self._cache:
            self._cache.move_to_end(path)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[path] = img.copy()


def _resolve_obj_id(
    label_text: str,
    label_id: int,
    id2label: dict[int, str],
    vocab2id: dict[str, int],
    vocab2id_lower: dict[str, int],
    num_classes: int,
) -> int:
    """Return a valid object id in [0, num_classes), or -1 if not resolvable."""
    obj_id = int(label_id)
    if obj_id < 0:
        raw = alias_label_text(label_text)
        if raw in vocab2id:
            obj_id = int(vocab2id[raw])
        else:
            obj_id = int(vocab2id_lower.get(raw.lower(), -1))

    if int(num_classes) > 0:
        return obj_id if 0 <= obj_id < int(num_classes) else -1
    return obj_id if obj_id >= 0 else -1


def format_structured_target_text(
    label_text: str,
    label_id: int,
    id2label: dict[int, str] | None,
    vocab2id: dict[str, int] | None,
    vocab2id_lower: dict[str, int] | None,
    num_classes: int,
    point_x: float,
    point_y: float,
    coord_bins: int = 1000,
    reasoning_text: str | None = None,
    force_reasoning_format: bool = False,
    target_order: str = "object_point",
) -> tuple[str, int, float, float]:
    """Build structured target text.

    Returns (target_text, resolved_obj_id, target_text_valid, target_object_valid).
    """
    raw = alias_label_text(label_text)
    if not raw and id2label is not None and int(label_id) >= 0:
        raw = alias_label_text(str(id2label.get(int(label_id), "")).strip())

    obj_id = _resolve_obj_id(
        label_text=raw,
        label_id=int(label_id),
        id2label=id2label or {},
        vocab2id=vocab2id or {},
        vocab2id_lower=vocab2id_lower or {},
        num_classes=int(num_classes),
    )

    obj_tok = None if obj_id >= 0 else GAZE_OBJ_UNKNOWN
    resolved_obj_id = obj_id if obj_id >= 0 else None

    text = build_structured_target_text(
        point_x=float(point_x),
        point_y=float(point_y),
        obj_id=resolved_obj_id,
        num_classes=int(num_classes),
        obj_token=obj_tok,
        coord_bins=int(coord_bins),
        target_order=str(target_order or "object_point"),
        reasoning_text=reasoning_text,
        force_reasoning_format=bool(force_reasoning_format),
    )

    return text, obj_id, 1.0, 1.0 if obj_id >= 0 else 0.0


def draw_head_bbox_prompt(
    scene: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Image.Image:
    w, h = scene.size
    rx1 = max(0, min(int(x1), max(w - 1, 0)))
    ry1 = max(0, min(int(y1), max(h - 1, 0)))
    rx2 = max(0, min(int(x2) - 1, max(w - 1, 0)))
    ry2 = max(0, min(int(y2) - 1, max(h - 1, 0)))
    if (rx2 <= rx1) or (ry2 <= ry1):
        return scene

    side = max(1, min(w, h))
    line_w = max(2, int(round(float(side) * 0.006)))
    out = scene.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([rx1, ry1, rx2, ry2], outline=(255, 0, 0), width=line_w)
    return out


class GazeDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        prompt_template: str,
        prompt_text: str = "",
        apply_augmentation: bool = False,
        id2label: dict[int, str] | None = None,
        vocab2id: dict[str, int] | None = None,
        vocab2id_lower: dict[str, int] | None = None,
        num_classes: int = 0,
        visual_prompting: bool = False,
        image_cache_size: int = 0,
        filter_invalid_object_samples: bool = True,
        coord_bins: int = 1000,
        reasoning_index: dict[str, Any] | None = None,
        force_reasoning_format: bool = False,
        target_order: str = "reasoning_object_point",
        disable_spatial_aug_for_reasoning: bool = True,
        # deprecated args kept for backward compat (ignored)
        answer_template: str = "",
        fallback_target_text: str = "",
        point_decimals: int = 4,
    ) -> None:
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.apply_augmentation = bool(apply_augmentation)
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.coord_bins = int(coord_bins)
        self.visual_prompting = bool(visual_prompting)
        self.reasoning_index: dict[str, Any] | None = reasoning_index
        self.force_reasoning_format = bool(force_reasoning_format)
        self.target_order = str(target_order or "reasoning_object_point")
        self.disable_spatial_aug_for_reasoning = bool(disable_spatial_aug_for_reasoning)
        self._image_cache: _ImageLRUCache | None = (
            _ImageLRUCache(int(image_cache_size)) if int(image_cache_size) > 0 else None
        )

        if bool(filter_invalid_object_samples):
            before = len(records)
            self.records = [
                r for r in records
                if _resolve_obj_id(
                    label_text=str(r.label_text or "").strip(),
                    label_id=int(r.label_id),
                    id2label=self.id2label,
                    vocab2id=self.vocab2id,
                    vocab2id_lower=self.vocab2id_lower,
                    num_classes=self.num_classes,
                ) >= 0
            ]
            dropped = before - len(self.records)
            if dropped > 0:
                print(
                    f"[INFO] GazeDataset: filtered {dropped}/{before} records "
                    f"with invalid object labels (filter_invalid_object_samples=True)"
                )
        else:
            self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        path_str = str(rec.image_path)
        scene: Image.Image | None = None
        if self._image_cache is not None:
            scene = self._image_cache.get(path_str)
        if scene is None:
            with Image.open(rec.image_path) as img:
                scene = img.convert("RGB")
            if self._image_cache is not None:
                self._image_cache.put(path_str, scene)

        gaze_x = float(rec.gaze_x)
        gaze_y = float(rec.gaze_y)
        bbox_px = rec.bbox_px

        # Probe reasoning BEFORE augmentation so aug policy can depend on it.
        reasoning_text: str | None = None
        has_reasoning_file = False
        if self.reasoning_index is not None:
            from pathlib import Path as _Path
            folder = _Path(rec.image_rel).parent.name
            stem = _Path(rec.image_rel).stem
            key = f"{folder}/{stem}_{int(rec.sample_id)}"
            rpath = self.reasoning_index.get(key)
            if rpath is not None:
                reasoning_text = load_reasoning_text(rpath)
                has_reasoning_file = reasoning_text is not None

        if self.apply_augmentation:
            # Spatial aug (hflip/crop) invalidates original-image reasoning text.
            # Use safe (color-jitter-only) aug for reasoning samples.
            if has_reasoning_file and self.disable_spatial_aug_for_reasoning:
                scene, gaze_x, gaze_y, bbox_px = apply_safe_augmentation(
                    scene=scene,
                    gaze_x=gaze_x,
                    gaze_y=gaze_y,
                    bbox_px=bbox_px,
                )
            else:
                scene, gaze_x, gaze_y, bbox_px = apply_train_augmentation(
                    scene=scene,
                    gaze_x=gaze_x,
                    gaze_y=gaze_y,
                    bbox_px=bbox_px,
                )

        w, h = scene.size
        x1, y1, x2, y2 = sanitize_bbox_pixels(bbox_px, width=w, height=h)
        if self.visual_prompting:
            scene = draw_head_bbox_prompt(scene, x1=x1, y1=y1, x2=x2, y2=y2)
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
        prompt = build_prompt(
            bbox_norm,
            self.prompt_template,
            self.prompt_text,
            num_classes=self.num_classes,
            point_decimals=4,
            coord_bins=self.coord_bins,
        )

        resolved_label_text = str(rec.label_text or "").strip()
        if (not resolved_label_text) and int(rec.label_id) >= 0:
            resolved_label_text = str(self.id2label.get(int(rec.label_id), "")).strip()

        target_text, obj_id, target_text_valid, target_object_valid = format_structured_target_text(
            label_text=resolved_label_text,
            label_id=int(rec.label_id),
            id2label=self.id2label,
            vocab2id=self.vocab2id,
            vocab2id_lower=self.vocab2id_lower,
            num_classes=self.num_classes,
            point_x=float(gaze_x),
            point_y=float(gaze_y),
            coord_bins=self.coord_bins,
            reasoning_text=reasoning_text,
            force_reasoning_format=self.force_reasoning_format,
            target_order=self.target_order,
        )

        bx = quantize_coord(float(gaze_x), bins=self.coord_bins)
        by = quantize_coord(float(gaze_y), bins=self.coord_bins)

        return {
            "scene_image": scene,
            "text_input": prompt,
            "target_text": target_text,
            "target_text_valid": torch.tensor(target_text_valid, dtype=torch.float32),
            "target_point_valid": torch.tensor(1.0, dtype=torch.float32),
            "target_object_valid": torch.tensor(target_object_valid, dtype=torch.float32),
            "target_label": int(rec.label_id),
            "target_label_ids": [int(obj_id)] if int(obj_id) >= 0 else [],
            "target_point": torch.tensor([float(gaze_x), float(gaze_y)], dtype=torch.float32),
            "target_point_bin": torch.tensor([bx, by], dtype=torch.long),
            "target_object_id": torch.tensor(max(0, obj_id), dtype=torch.long),
            "target_structured_valid": torch.tensor(target_text_valid, dtype=torch.float32),
            "target_format_valid": torch.tensor(1.0, dtype=torch.float32),
            "gt_points": torch.tensor([[float(gaze_x), float(gaze_y)]], dtype=torch.float32),
            "target_label_text": resolved_label_text,
            "image_rel": str(rec.image_rel),
            "reasoning_text": reasoning_text or "",
            "has_reasoning": has_reasoning_file,
        }


class MultiViewGazeDataset(Dataset):
    """Two-view dataset: one direct view and one reasoning view per record.

    Direct views use full augmentation and point_object target order.
    Reasoning views use safe augmentation and reasoning_point_object target order.
    Both views are stored in self._views for full dual-view training.
    """

    def __init__(
        self,
        records: list[Record],
        prompt_template: str,
        prompt_text_direct: str,
        prompt_text_reasoning: str,
        id2label: dict[int, str] | None = None,
        vocab2id: dict[str, int] | None = None,
        vocab2id_lower: dict[str, int] | None = None,
        num_classes: int = 0,
        visual_prompting: bool = False,
        image_cache_size: int = 0,
        filter_invalid_object_samples: bool = True,
        coord_bins: int = 1000,
        reasoning_index: dict[str, Any] | None = None,
        max_reasoning_words: int = 30,
        max_reasoning_chars: int = 220,
    ) -> None:
        from pathlib import Path as _Path

        self.prompt_template = str(prompt_template or "")
        self.prompt_text_direct = str(prompt_text_direct or "")
        self.prompt_text_reasoning = str(prompt_text_reasoning or "")
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.coord_bins = int(coord_bins)
        self.visual_prompting = bool(visual_prompting)
        self.reasoning_index: dict[str, Any] | None = reasoning_index
        self.max_reasoning_words = int(max_reasoning_words)
        self.max_reasoning_chars = int(max_reasoning_chars)
        self._image_cache: _ImageLRUCache | None = (
            _ImageLRUCache(int(image_cache_size)) if int(image_cache_size) > 0 else None
        )

        if bool(filter_invalid_object_samples):
            before = len(records)
            records = [
                r for r in records
                if _resolve_obj_id(
                    label_text=str(r.label_text or "").strip(),
                    label_id=int(r.label_id),
                    id2label=self.id2label,
                    vocab2id=self.vocab2id,
                    vocab2id_lower=self.vocab2id_lower,
                    num_classes=self.num_classes,
                ) >= 0
            ]
            dropped = before - len(records)
            if dropped > 0:
                print(
                    f"[INFO] MultiViewGazeDataset: filtered {dropped}/{before} records "
                    f"with invalid object labels"
                )
        self.records = records

        # Build view list: (record_index, view_type)
        self._direct_views: list[tuple[int, str]] = [(i, "direct") for i in range(len(self.records))]
        self._reasoning_views: list[tuple[int, str]] = []

        if reasoning_index is not None:
            for i, rec in enumerate(self.records):
                folder = _Path(rec.image_rel).parent.name
                stem = _Path(rec.image_rel).stem
                key = f"{folder}/{stem}_{int(rec.sample_id)}"
                if reasoning_index.get(key) is not None:
                    self._reasoning_views.append((i, "reasoning"))

        self._views: list[tuple[int, str]] = self._direct_views + self._reasoning_views

    def get_view_counts(self) -> tuple[int, int]:
        """Return (n_direct, n_reasoning) view counts."""
        return len(self._direct_views), len(self._reasoning_views)

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from pathlib import Path as _Path
        from .utils.gaze_tokens import normalize_reasoning_text

        rec_idx, view_type = self._views[idx]
        rec = self.records[rec_idx]

        path_str = str(rec.image_path)
        scene: Image.Image | None = None
        if self._image_cache is not None:
            scene = self._image_cache.get(path_str)
        if scene is None:
            with Image.open(rec.image_path) as img:
                scene = img.convert("RGB")
            if self._image_cache is not None:
                self._image_cache.put(path_str, scene)

        gaze_x = float(rec.gaze_x)
        gaze_y = float(rec.gaze_y)
        bbox_px = rec.bbox_px

        if view_type == "reasoning":
            # Reasoning view: safe (color-jitter-only) augmentation
            scene, gaze_x, gaze_y, bbox_px = apply_safe_augmentation(
                scene=scene, gaze_x=gaze_x, gaze_y=gaze_y, bbox_px=bbox_px,
            )
            # Load reasoning text lazily; fall back to direct format if file is malformed.
            reasoning_text: str | None = None
            if self.reasoning_index is not None:
                folder = _Path(rec.image_rel).parent.name
                stem = _Path(rec.image_rel).stem
                key = f"{folder}/{stem}_{int(rec.sample_id)}"
                rpath = self.reasoning_index.get(key)
                if rpath is not None:
                    raw = load_reasoning_text(rpath)
                    if raw:
                        reasoning_text = normalize_reasoning_text(
                            raw,
                            max_words=self.max_reasoning_words,
                            max_chars=self.max_reasoning_chars,
                        )
            if reasoning_text:
                target_order = "reasoning_point_object"
                prompt_text = self.prompt_text_reasoning
            else:
                # Malformed or missing content → treat as direct view
                target_order = "point_object"
                prompt_text = self.prompt_text_direct
        else:
            # Direct view: full augmentation
            scene, gaze_x, gaze_y, bbox_px = apply_train_augmentation(
                scene=scene, gaze_x=gaze_x, gaze_y=gaze_y, bbox_px=bbox_px,
            )
            reasoning_text = None
            target_order = "point_object"
            prompt_text = self.prompt_text_direct

        w, h = scene.size
        x1, y1, x2, y2 = sanitize_bbox_pixels(bbox_px, width=w, height=h)
        if self.visual_prompting:
            scene = draw_head_bbox_prompt(scene, x1=x1, y1=y1, x2=x2, y2=y2)
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
        prompt = build_prompt(
            bbox_norm,
            self.prompt_template,
            prompt_text,
            num_classes=self.num_classes,
            point_decimals=4,
            coord_bins=self.coord_bins,
        )

        resolved_label_text = str(rec.label_text or "").strip()
        if (not resolved_label_text) and int(rec.label_id) >= 0:
            resolved_label_text = str(self.id2label.get(int(rec.label_id), "")).strip()

        target_text, obj_id, target_text_valid, target_object_valid = format_structured_target_text(
            label_text=resolved_label_text,
            label_id=int(rec.label_id),
            id2label=self.id2label,
            vocab2id=self.vocab2id,
            vocab2id_lower=self.vocab2id_lower,
            num_classes=self.num_classes,
            point_x=float(gaze_x),
            point_y=float(gaze_y),
            coord_bins=self.coord_bins,
            reasoning_text=reasoning_text,
            force_reasoning_format=False,
            target_order=target_order,
        )

        bx = quantize_coord(float(gaze_x), bins=self.coord_bins)
        by = quantize_coord(float(gaze_y), bins=self.coord_bins)

        return {
            "scene_image": scene,
            "text_input": prompt,
            "target_text": target_text,
            "target_text_valid": torch.tensor(target_text_valid, dtype=torch.float32),
            "target_point_valid": torch.tensor(1.0, dtype=torch.float32),
            "target_object_valid": torch.tensor(target_object_valid, dtype=torch.float32),
            "target_label": int(rec.label_id),
            "target_label_ids": [int(obj_id)] if int(obj_id) >= 0 else [],
            "target_point": torch.tensor([float(gaze_x), float(gaze_y)], dtype=torch.float32),
            "target_point_bin": torch.tensor([bx, by], dtype=torch.long),
            "target_object_id": torch.tensor(max(0, obj_id), dtype=torch.long),
            "target_structured_valid": torch.tensor(target_text_valid, dtype=torch.float32),
            "target_format_valid": torch.tensor(1.0, dtype=torch.float32),
            "gt_points": torch.tensor([[float(gaze_x), float(gaze_y)]], dtype=torch.float32),
            "target_label_text": resolved_label_text,
            "image_rel": str(rec.image_rel),
            "reasoning_text": reasoning_text or "",
            "has_reasoning": reasoning_text is not None,
            "view_type": view_type,
        }


class GazeTestDataset(Dataset):
    def __init__(
        self,
        groups: list[TestGroup],
        prompt_template: str,
        prompt_text: str,
        id2label: dict[int, str] | None = None,
        vocab2id: dict[str, int] | None = None,
        vocab2id_lower: dict[str, int] | None = None,
        num_classes: int = 0,
        visual_prompting: bool = False,
        image_cache_size: int = 0,
        coord_bins: int = 1000,
        force_reasoning_format: bool = False,
        target_order: str = "object_point",
        # deprecated args kept for backward compat (ignored)
        answer_template: str = "",
        fallback_target_text: str = "",
        point_decimals: int = 4,
    ) -> None:
        self.groups = groups
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.coord_bins = int(coord_bins)
        self.visual_prompting = bool(visual_prompting)
        self.force_reasoning_format = bool(force_reasoning_format)
        self.target_order = str(target_order or "object_point")
        self._image_cache: _ImageLRUCache | None = (
            _ImageLRUCache(int(image_cache_size)) if int(image_cache_size) > 0 else None
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = self.groups[idx]
        path_str = str(g.image_path)
        scene: Image.Image | None = None
        if self._image_cache is not None:
            scene = self._image_cache.get(path_str)
        if scene is None:
            with Image.open(g.image_path) as img:
                scene = img.convert("RGB")
            if self._image_cache is not None:
                self._image_cache.put(path_str, scene)
        w, h = scene.size

        x1, y1, x2, y2 = sanitize_bbox_pixels(g.bbox_px, width=w, height=h)
        if self.visual_prompting:
            scene = draw_head_bbox_prompt(scene, x1=x1, y1=y1, x2=x2, y2=y2)
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
        prompt = build_prompt(
            bbox_norm,
            self.prompt_template,
            self.prompt_text,
            num_classes=self.num_classes,
            point_decimals=4,
            coord_bins=self.coord_bins,
        )
        if g.gt_points:
            px = sum(float(x) for x, _ in g.gt_points) / float(len(g.gt_points))
            py = sum(float(y) for _, y in g.gt_points) / float(len(g.gt_points))
        else:
            px, py = 0.5, 0.5

        target_text, obj_id, target_text_valid, target_object_valid = format_structured_target_text(
            label_text=g.label_text,
            label_id=int(g.label_id),
            id2label=self.id2label,
            vocab2id=self.vocab2id,
            vocab2id_lower=self.vocab2id_lower,
            num_classes=self.num_classes,
            point_x=px,
            point_y=py,
            coord_bins=self.coord_bins,
            force_reasoning_format=self.force_reasoning_format,
            target_order=self.target_order,
        )

        bx = quantize_coord(float(px), bins=self.coord_bins)
        by = quantize_coord(float(py), bins=self.coord_bins)

        return {
            "scene_image": scene,
            "text_input": prompt,
            "target_text": target_text,
            "target_text_valid": torch.tensor(target_text_valid, dtype=torch.float32),
            "target_point_valid": torch.tensor(1.0, dtype=torch.float32),
            "target_object_valid": torch.tensor(target_object_valid, dtype=torch.float32),
            "target_label": int(g.label_id),
            "target_label_ids": [int(x) for x in (g.label_ids or []) if int(x) >= 0],
            "target_point": torch.tensor([float(px), float(py)], dtype=torch.float32),
            "target_point_bin": torch.tensor([bx, by], dtype=torch.long),
            "target_object_id": torch.tensor(max(0, obj_id), dtype=torch.long),
            "target_structured_valid": torch.tensor(target_text_valid, dtype=torch.float32),
            "target_format_valid": torch.tensor(1.0, dtype=torch.float32),
            "gt_points": torch.tensor(g.gt_points, dtype=torch.float32),
            "target_label_text": str(g.label_text),
            "image_rel": str(g.image_rel),
        }
