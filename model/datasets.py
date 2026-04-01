from __future__ import annotations

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
from .utils.object_tokens import build_object_token, format_answer, object_token_width


def _format_target_text(
    label_text: str,
    label_id: int,
    id2label: dict[int, str] | None,
    vocab2id: dict[str, int] | None,
    vocab2id_lower: dict[str, int] | None,
    num_classes: int,
    answer_template: str,
    fallback_target_text: str,
    fallback_object_id: int,
    point_x: float,
    point_y: float,
    point_decimals: int,
) -> tuple[str, float]:
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

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
    obj_w = object_token_width(int(num_classes))
    if is_valid_obj:
        obj_token = build_object_token(int(obj_id), width=obj_w)
    else:
        if int(num_classes) > 0:
            safe_fallback_id = int(fallback_object_id)
            if (safe_fallback_id < 0) or (safe_fallback_id >= int(num_classes)):
                safe_fallback_id = 0
        else:
            safe_fallback_id = max(0, int(fallback_object_id))
        obj_token = build_object_token(int(safe_fallback_id), width=obj_w)

    dec = max(0, int(point_decimals))
    px = f"{_clamp01(point_x):.{dec}f}"
    py = f"{_clamp01(point_y):.{dec}f}"
    tpl = str(answer_template or "Point: {point_x} {point_y}\nObject: {object_token}")
    try:
        text = tpl.format(
            label_text=raw,
            label_id=int(label_id),
            point_x=px,
            point_y=py,
            object_id=int(obj_id),
            object_token=str(obj_token),
        )
    except Exception:
        text = format_answer(
            point_x=float(point_x),
            point_y=float(point_y),
            label_id=int(safe_fallback_id if (not is_valid_obj) else int(obj_id)),
            point_decimals=dec,
            width=int(obj_w),
        )
    return str(text), float(is_valid)


def _draw_head_bbox_prompt(
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
        answer_template: str = "Point: {point_x} {point_y}\nObject: {object_token}",
        fallback_target_text: str = "unknown",
        fallback_object_id: int = -1,
        point_decimals: int = 4,
        visual_prompting: bool = False,
    ) -> None:
        self.records = records
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.apply_augmentation = bool(apply_augmentation)
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.answer_template = str(answer_template or "Point: {point_x} {point_y}\nObject: {object_token}")
        self.fallback_target_text = str(fallback_target_text)
        self.fallback_object_id = int(fallback_object_id)
        self.point_decimals = int(point_decimals)
        self.visual_prompting = bool(visual_prompting)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        with Image.open(rec.image_path) as img:
            scene = img.convert("RGB")

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
            scene = _draw_head_bbox_prompt(scene, x1=x1, y1=y1, x2=x2, y2=y2)
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
        prompt = build_prompt(bbox_norm, self.prompt_template, self.prompt_text)
        target_text, target_valid = _format_target_text(
            label_text=rec.label_text,
            label_id=int(rec.label_id),
            id2label=self.id2label,
            vocab2id=self.vocab2id,
            vocab2id_lower=self.vocab2id_lower,
            num_classes=self.num_classes,
            answer_template=self.answer_template,
            fallback_target_text=self.fallback_target_text,
            fallback_object_id=self.fallback_object_id,
            point_x=float(gaze_x),
            point_y=float(gaze_y),
            point_decimals=self.point_decimals,
        )

        return {
            "scene_image": scene,
            "text_input": prompt,
            "target_text": target_text,
            "target_text_valid": torch.tensor(target_valid, dtype=torch.float32),
            "target_point_valid": torch.tensor(1.0, dtype=torch.float32),
            "target_object_valid": torch.tensor(target_valid, dtype=torch.float32),
            "target_label": int(rec.label_id),
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
        answer_template: str = "Point: {point_x} {point_y}\nObject: {object_token}",
        fallback_target_text: str = "unknown",
        fallback_object_id: int = -1,
        point_decimals: int = 4,
        visual_prompting: bool = False,
    ) -> None:
        self.groups = groups
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.id2label = id2label or {}
        self.vocab2id = vocab2id or {}
        self.vocab2id_lower = vocab2id_lower or {}
        self.num_classes = int(num_classes)
        self.answer_template = str(answer_template or "Point: {point_x} {point_y}\nObject: {object_token}")
        self.fallback_target_text = str(fallback_target_text)
        self.fallback_object_id = int(fallback_object_id)
        self.point_decimals = int(point_decimals)
        self.visual_prompting = bool(visual_prompting)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = self.groups[idx]
        with Image.open(g.image_path) as img:
            scene = img.convert("RGB")
        w, h = scene.size

        x1, y1, x2, y2 = sanitize_bbox_pixels(g.bbox_px, width=w, height=h)
        if self.visual_prompting:
            scene = _draw_head_bbox_prompt(scene, x1=x1, y1=y1, x2=x2, y2=y2)
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
        prompt = build_prompt(bbox_norm, self.prompt_template, self.prompt_text)
        if g.gt_points:
            px = sum(float(x) for x, _ in g.gt_points) / float(len(g.gt_points))
            py = sum(float(y) for _, y in g.gt_points) / float(len(g.gt_points))
        else:
            px, py = 0.5, 0.5
        target_text, target_valid = _format_target_text(
            label_text=g.label_text,
            label_id=int(g.label_id),
            id2label=self.id2label,
            vocab2id=self.vocab2id,
            vocab2id_lower=self.vocab2id_lower,
            num_classes=self.num_classes,
            answer_template=self.answer_template,
            fallback_target_text=self.fallback_target_text,
            fallback_object_id=self.fallback_object_id,
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
