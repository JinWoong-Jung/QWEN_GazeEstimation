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
    build_prompt,
    sanitize_bbox_pixels,
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

def format_target_text(
    label_text: str,
    label_id: int,
    id2label: dict[int, str] | None,
    vocab2id: dict[str, int] | None,
    vocab2id_lower: dict[str, int] | None,
    num_classes: int,
    answer_template: str,
    fallback_target_text: str,
    point_x: float,
    point_y: float,
    point_decimals: int,
) -> tuple[str, float]:
    raw = str(label_text or "").strip()
    if not raw and (id2label is not None) and int(label_id) >= 0:
        raw = str(id2label.get(int(label_id), "")).strip()
    if not raw:
        raw = str(fallback_target_text)

    obj_id = int(label_id)
    if obj_id < 0:
        v = vocab2id or {}
        vl = vocab2id_lower or {}
        if raw in v:
            obj_id = int(v[raw])
        else:
            obj_id = int(vl.get(raw.lower(), -1))

    if int(num_classes) > 0:
        is_valid_obj = 0 <= int(obj_id) < int(num_classes)
    else:
        is_valid_obj = int(obj_id) >= 0
    is_valid = 1.0 if is_valid_obj else 0.0

    dec = max(0, int(point_decimals))
    px = f"{float(point_x):.{dec}f}"
    py = f"{float(point_y):.{dec}f}"
    tpl = str(answer_template or "Point: {point_x} {point_y}\nObject: {label_text}")
    try:
        text = tpl.format(label_text=raw, point_x=px, point_y=py)
    except Exception:
        text = f"Point: {px} {py}\nObject: {raw}"
    return str(text), float(is_valid)


def draw_head_bbox_prompt(
    scene: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Image.Image:
    w, h = scene.size
    # Draw inclusive rectangle coordinates, clamped to valid image extent.
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
        answer_template: str = "Point: {point_x} {point_y}\nObject: {label_text}",
        fallback_target_text: str = "unknown",
        point_decimals: int = 4,
        visual_prompting: bool = False,
        image_cache_size: int = 0,
    ) -> None:
        self.records = records
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.apply_augmentation = bool(apply_augmentation)
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.answer_template = str(answer_template or "Point: {point_x} {point_y}\nObject: {label_text}")
        self.fallback_target_text = str(fallback_target_text)
        self.point_decimals = int(point_decimals)
        self.visual_prompting = bool(visual_prompting)
        self._image_cache: _ImageLRUCache | None = (
            _ImageLRUCache(int(image_cache_size)) if int(image_cache_size) > 0 else None
        )

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
        if self.apply_augmentation:
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
        prompt = build_prompt(bbox_norm, self.prompt_template, self.prompt_text)
        resolved_label_text = str(rec.label_text or "").strip()
        if (not resolved_label_text) and int(rec.label_id) >= 0:
            resolved_label_text = str(self.id2label.get(int(rec.label_id), "")).strip()
        target_text, target_valid = format_target_text(
            label_text=resolved_label_text,
            label_id=int(rec.label_id),
            id2label=self.id2label,
            vocab2id=self.vocab2id,
            vocab2id_lower=self.vocab2id_lower,
            num_classes=self.num_classes,
            answer_template=self.answer_template,
            fallback_target_text=self.fallback_target_text,
            point_x=float(gaze_x),
            point_y=float(gaze_y),
            point_decimals=self.point_decimals,
        )

        return {
            "scene_image": scene,
            "text_input": prompt,
            "target_text": target_text,
            # target_text_valid controls whether point/format masks are built.
            # Always 1.0: even when the object label is invalid we still want
            # to supervise the gaze coordinate (localization is the primary goal).
            "target_text_valid": torch.tensor(1.0, dtype=torch.float32),
            "target_point_valid": torch.tensor(1.0, dtype=torch.float32),
            # target_object_valid is decoupled from point validity so that
            # object retrieval loss is suppressed only when the label is bad.
            "target_object_valid": torch.tensor(target_valid, dtype=torch.float32),
            "target_label": int(rec.label_id),
            "target_label_ids": [int(rec.label_id)] if int(rec.label_id) >= 0 else [],
            "target_point": torch.tensor([float(gaze_x), float(gaze_y)], dtype=torch.float32),
            "gt_points": torch.tensor([[float(gaze_x), float(gaze_y)]], dtype=torch.float32),
            "target_label_text": resolved_label_text,
            "image_rel": str(rec.image_rel),
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
        answer_template: str = "Point: {point_x} {point_y}\nObject: {label_text}",
        fallback_target_text: str = "unknown",
        point_decimals: int = 4,
        visual_prompting: bool = False,
        image_cache_size: int = 0,
    ) -> None:
        self.groups = groups
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.answer_template = str(answer_template or "Point: {point_x} {point_y}\nObject: {label_text}")
        self.fallback_target_text = str(fallback_target_text)
        self.point_decimals = int(point_decimals)
        self.visual_prompting = bool(visual_prompting)
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
        prompt = build_prompt(bbox_norm, self.prompt_template, self.prompt_text)
        if g.gt_points:
            px = sum(float(x) for x, _ in g.gt_points) / float(len(g.gt_points))
            py = sum(float(y) for _, y in g.gt_points) / float(len(g.gt_points))
        else:
            px, py = 0.5, 0.5
        target_text, target_valid = format_target_text(
            label_text=g.label_text,
            label_id=int(g.label_id),
            id2label=self.id2label,
            vocab2id=self.vocab2id,
            vocab2id_lower=self.vocab2id_lower,
            num_classes=self.num_classes,
            answer_template=self.answer_template,
            fallback_target_text=self.fallback_target_text,
            point_x=px,
            point_y=py,
            point_decimals=self.point_decimals,
        )

        return {
            "scene_image": scene,
            "text_input": prompt,
            "target_text": target_text,
            "target_text_valid": torch.tensor(target_valid, dtype=torch.float32),
            "target_point_valid": torch.tensor(1.0, dtype=torch.float32),
            "target_object_valid": torch.tensor(target_valid, dtype=torch.float32),
            "target_label": int(g.label_id),
            "target_label_ids": [int(x) for x in (g.label_ids or []) if int(x) >= 0],
            "target_point": torch.tensor([float(px), float(py)], dtype=torch.float32),
            "gt_points": torch.tensor(g.gt_points, dtype=torch.float32),
            "target_label_text": str(g.label_text),
            "image_rel": str(g.image_rel),
        }
