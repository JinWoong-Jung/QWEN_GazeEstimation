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
    alias_label_text,
    build_prompt,
    crop_head_region,
    load_reasoning_record,
    sanitize_bbox_pixels,
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


class _ReasoningTextLRUCache:
    """Lazy cache for SDFT teacher-demonstration text."""

    def __init__(
        self,
        index: dict[str, Any] | None,
        *,
        maxsize: int = 8192,
    ) -> None:
        self._index = index or {}
        self._cache: OrderedDict[str, dict[str, str | None]] = OrderedDict()
        self._maxsize = max(1, int(maxsize))

    def __bool__(self) -> bool:
        return bool(self._index)

    def get(self, key: str) -> dict[str, str | None]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self._index.get(key)
        raw = load_reasoning_record(path) if path is not None else {}
        record = {
            "object_text": _normalize_aux_text(raw.get("object_text")),
            "reasoning_text": _normalize_aux_text(raw.get("reasoning_text")),
        }
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = record
        return record


def _normalize_aux_text(text: str | None) -> str | None:
    text = " ".join(str(text or "").split())
    return text or None


def _reasoning_key(image_rel: str, sample_id: int) -> str:
    from pathlib import Path as _Path

    path = _Path(image_rel)
    return f"{path.parent.name}/{path.stem}_{int(sample_id)}"


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
    target_order: str = "point_object",
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
        target_order=str(target_order or "point_object"),
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
        prompt_text_teacher: str = "",
        apply_augmentation: bool = False,
        id2label: dict[int, str] | None = None,
        vocab2id: dict[str, int] | None = None,
        vocab2id_lower: dict[str, int] | None = None,
        num_classes: int = 0,
        visual_prompting: bool = False,
        image_cache_size: int = 0,
        filter_invalid_object_samples: bool = True,
        coord_bins: int = 1000,
        train_augmentation_mode: str = "full",
        target_order: str = "point_object",
        reasoning_index: dict[str, Any] | None = None,
        use_head_crop: bool = False,
        head_crop_padding: float = 0.3,
        head_crop_size: int = 224,
        # deprecated args kept for backward compat (ignored)
        answer_template: str = "",
        fallback_target_text: str = "",
        point_decimals: int = 4,
    ) -> None:
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.prompt_text_teacher = str(prompt_text_teacher or "")
        self.apply_augmentation = bool(apply_augmentation)
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.coord_bins = int(coord_bins)
        self.train_augmentation_mode = str(train_augmentation_mode or "full")
        self.visual_prompting = bool(visual_prompting)
        self.target_order = str(target_order or "point_object")
        self.use_head_crop = bool(use_head_crop)
        self.head_crop_padding = float(head_crop_padding)
        self.head_crop_size = int(head_crop_size)
        self._image_cache: _ImageLRUCache | None = (
            _ImageLRUCache(int(image_cache_size)) if int(image_cache_size) > 0 else None
        )
        self._reasoning_text_cache = _ReasoningTextLRUCache(reasoning_index)

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
        reasoning_text: str | None = None
        object_text: str | None = None
        if self._reasoning_text_cache:
            reasoning_record = self._reasoning_text_cache.get(
                _reasoning_key(str(rec.image_rel), int(rec.sample_id))
            )
            reasoning_text = reasoning_record.get("reasoning_text")
            object_text = reasoning_record.get("object_text")

        if self.apply_augmentation:
            scene, gaze_x, gaze_y, bbox_px = apply_train_augmentation(
                scene=scene,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                bbox_px=bbox_px,
                mode=self.train_augmentation_mode,
            )

        w, h = scene.size
        x1, y1, x2, y2 = sanitize_bbox_pixels(bbox_px, width=w, height=h)
        head_crop: Image.Image | None = None
        if self.use_head_crop:
            head_crop = crop_head_region(
                scene, (x1, y1, x2, y2),
                padding=self.head_crop_padding,
                target_size=self.head_crop_size,
            )
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
        prompt_teacher = build_prompt(
            bbox_norm,
            self.prompt_template,
            self.prompt_text_teacher,
            num_classes=self.num_classes,
            point_decimals=4,
            coord_bins=self.coord_bins,
        ) if self.prompt_text_teacher else prompt

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
            target_order=self.target_order,
        )

        bx = quantize_coord(float(gaze_x), bins=self.coord_bins)
        by = quantize_coord(float(gaze_y), bins=self.coord_bins)

        return {
            "scene_image": scene,
            "head_crop_image": head_crop,
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
            "object_text": object_text or resolved_label_text,
            "reasoning_text": reasoning_text or "",
            "text_input_teacher": prompt_teacher,
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
        target_order: str = "point_object",
        use_head_crop: bool = False,
        head_crop_padding: float = 0.3,
        head_crop_size: int = 224,
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
        self.target_order = str(target_order or "point_object")
        self.use_head_crop = bool(use_head_crop)
        self.head_crop_padding = float(head_crop_padding)
        self.head_crop_size = int(head_crop_size)
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
        head_crop: Image.Image | None = None
        if self.use_head_crop:
            head_crop = crop_head_region(
                scene, (x1, y1, x2, y2),
                padding=self.head_crop_padding,
                target_size=self.head_crop_size,
            )
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
            target_order=self.target_order,
        )

        bx = quantize_coord(float(px), bins=self.coord_bins)
        by = quantize_coord(float(py), bins=self.coord_bins)

        return {
            "scene_image": scene,
            "head_crop_image": head_crop,
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
