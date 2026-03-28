from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .recognition_objectives import is_embedding_recognition_objective
from .utils.data_utils import (
    Record,
    TestGroup,
    apply_train_augmentation,
    build_prompt,
    gaussian_heatmap,
    sanitize_bbox_pixels,
)


class GazeDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        heatmap_size: tuple[int, int],
        heatmap_sigma: float,
        prompt_template: str,
        prompt_text: str = "",
        apply_augmentation: bool = False,
        recognition_objective: str = "ce",
        label_embed_dir: Path | None = None,
        label_emb_dim: int = 512,
        normalize_label_emb: bool = True,
    ) -> None:
        self.records = records
        self.heatmap_size = (int(heatmap_size[0]), int(heatmap_size[1]))
        self.heatmap_sigma = float(heatmap_sigma)
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")
        self.apply_augmentation = bool(apply_augmentation)
        self.recognition_objective = str(recognition_objective).strip().lower()
        self.label_embed_dir = label_embed_dir
        self.label_emb_dim = int(label_emb_dim)
        self.normalize_label_emb = bool(normalize_label_emb)
        self._label_emb_cache: dict[str, torch.Tensor] = {}
        self._label_emb_warn_count = 0

    def __len__(self) -> int:
        return len(self.records)

    def _warn_label_emb(self, msg: str) -> None:
        if self._label_emb_warn_count < 20:
            print(f"[GazeDataset][label_emb] {msg}")
            self._label_emb_warn_count += 1
            if self._label_emb_warn_count == 20:
                print("[GazeDataset][label_emb] warning log limit reached; suppressing further messages.")

    def _load_label_embedding(self, label_text: str) -> torch.Tensor:
        txt = str(label_text).strip()
        if not txt:
            return torch.zeros((self.label_emb_dim,), dtype=torch.float32)
        if txt in self._label_emb_cache:
            return self._label_emb_cache[txt].clone()
        if self.label_embed_dir is None:
            return torch.zeros((self.label_emb_dim,), dtype=torch.float32)
        p = self.label_embed_dir / f"{txt}-emb.pt"
        if not p.exists():
            self._warn_label_emb(f"missing embedding file: {p}")
            return torch.zeros((self.label_emb_dim,), dtype=torch.float32)
        try:
            emb = torch.load(p, map_location="cpu")
            if not torch.is_tensor(emb):
                raise TypeError(f"not tensor: {type(emb)}")
            emb = emb.to(dtype=torch.float32).flatten()
            if emb.numel() != self.label_emb_dim:
                raise ValueError(f"dim mismatch: got {emb.numel()}, expected {self.label_emb_dim}")
            if self.normalize_label_emb:
                emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
            self._label_emb_cache[txt] = emb
            return emb.clone()
        except Exception as e:
            self._warn_label_emb(f"failed to load {p}: {e}")
            return torch.zeros((self.label_emb_dim,), dtype=torch.float32)

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
        head = scene.crop((x1, y1, x2, y2))
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
        prompt = build_prompt(bbox_norm, self.prompt_template, self.prompt_text)
        heatmap = gaussian_heatmap(
            x_norm=gaze_x,
            y_norm=gaze_y,
            size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )
        if is_embedding_recognition_objective(self.recognition_objective):
            target_label_emb = self._load_label_embedding(rec.label_text)
            target_label_valid = float((target_label_emb.abs().sum().item() > 0.0) and bool(rec.label_text.strip()))
        else:
            target_label_emb = torch.zeros((self.label_emb_dim,), dtype=torch.float32)
            target_label_valid = 0.0

        return {
            "scene_image": scene,
            "head_image": head,
            "text_input": prompt,
            "target_point": torch.tensor([gaze_x, gaze_y], dtype=torch.float32),
            "target_heatmap": heatmap,
            "target_label": int(rec.label_id),
            "target_label_emb": target_label_emb,
            "target_label_valid": torch.tensor(target_label_valid, dtype=torch.float32),
        }


class GazeTestDataset(Dataset):
    def __init__(
        self,
        groups: list[TestGroup],
        prompt_template: str,
        prompt_text: str,
    ) -> None:
        self.groups = groups
        self.prompt_template = str(prompt_template or "")
        self.prompt_text = str(prompt_text or "")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = self.groups[idx]
        with Image.open(g.image_path) as img:
            scene = img.convert("RGB")
        w, h = scene.size

        x1, y1, x2, y2 = sanitize_bbox_pixels(g.bbox_px, width=w, height=h)
        head = scene.crop((x1, y1, x2, y2))
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
        prompt = build_prompt(bbox_norm, self.prompt_template, self.prompt_text)

        return {
            "scene_image": scene,
            "head_image": head,
            "text_input": prompt,
            "gt_points": torch.tensor(g.gt_points, dtype=torch.float32),
            "target_label": int(g.label_id),
            "target_label_ids": torch.tensor(g.label_ids, dtype=torch.long),
            "target_label_text": str(g.label_text),
        }
