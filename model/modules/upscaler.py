from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def softargmax2d(heatmap: torch.Tensor) -> torch.Tensor:
    if heatmap.dim() != 3:
        raise ValueError(f"heatmap must be [B, H, W], got shape={tuple(heatmap.shape)}")
    b, h, w = heatmap.shape
    flat = heatmap.reshape(b, -1)
    probs = torch.softmax(flat, dim=-1)

    xs = torch.linspace(0.0, 1.0, w, device=heatmap.device, dtype=heatmap.dtype)
    ys = torch.linspace(0.0, 1.0, h, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    gx = grid_x.reshape(1, -1)
    gy = grid_y.reshape(1, -1)
    x = torch.sum(probs * gx, dim=-1)
    y = torch.sum(probs * gy, dim=-1)
    return torch.stack([x, y], dim=-1).clamp(min=0.0, max=1.0)


def argmax2d(heatmap: torch.Tensor) -> torch.Tensor:
    if heatmap.dim() != 3:
        raise ValueError(f"heatmap must be [B, H, W], got shape={tuple(heatmap.shape)}")
    b, h, w = heatmap.shape
    idx = torch.argmax(heatmap.reshape(b, -1), dim=-1)
    y = idx // w
    x = idx % w
    x = x.to(dtype=heatmap.dtype) / max(w - 1, 1)
    y = y.to(dtype=heatmap.dtype) / max(h - 1, 1)
    return torch.stack([x, y], dim=-1)


class HeatmapUpscaler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        scene_grid_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] = (512, 512),
        upsample_mode: str = "bilinear",
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        if scene_grid_size is None:
            self.scene_h = -1
            self.scene_w = -1
        else:
            self.scene_h, self.scene_w = int(scene_grid_size[0]), int(scene_grid_size[1])
        self.out_h, self.out_w = int(output_size[0]), int(output_size[1])
        self.upsample_mode = upsample_mode
        self.apply_sigmoid = bool(apply_sigmoid)
        self.patch_scorer = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _infer_square_grid(n_tokens: int) -> tuple[int, int]:
        s = int(round(float(n_tokens) ** 0.5))
        if s > 0 and s * s == int(n_tokens):
            return s, s
        raise ValueError(
            f"Cannot infer 2D scene grid from N={n_tokens}. "
            "Provide scene_grid_size from backbone vision metadata."
        )

    def _resolve_scene_grid(
        self,
        n_tokens: int,
        scene_grid_size: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        if scene_grid_size is not None:
            h, w = int(scene_grid_size[0]), int(scene_grid_size[1])
        elif self.scene_h > 0 and self.scene_w > 0:
            h, w = int(self.scene_h), int(self.scene_w)
        else:
            h, w = self._infer_square_grid(n_tokens)

        if h <= 0 or w <= 0 or (h * w != int(n_tokens)):
            raise ValueError(
                f"Invalid scene grid size {(h, w)} for token count N={n_tokens}"
            )
        return h, w

    def _reshape_to_2d(
        self,
        patch_scores: torch.Tensor,
        scene_grid_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if patch_scores.dim() != 2:
            raise ValueError(f"patch_scores must be [B, N], got shape={tuple(patch_scores.shape)}")
        n_tokens = int(patch_scores.shape[1])
        scene_h, scene_w = self._resolve_scene_grid(n_tokens, scene_grid_size=scene_grid_size)
        coarse = patch_scores.view(patch_scores.shape[0], 1, scene_h, scene_w)
        return coarse, (scene_h, scene_w)

    def _upsample(self, coarse_map: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            coarse_map,
            size=(self.out_h, self.out_w),
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in {"bilinear", "bicubic"} else None,
        )

    def forward(
        self,
        scene_hidden: torch.Tensor,
        *,
        scene_grid_size: tuple[int, int] | None = None,
        use_softargmax: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if scene_hidden.dim() != 3:
            raise ValueError(f"scene_hidden must be [B, N, D], got shape={tuple(scene_hidden.shape)}")

        patch_logits = self.patch_scorer(scene_hidden).squeeze(-1)  # [B, N]
        coarse_logits, scene_hw = self._reshape_to_2d(
            patch_logits,
            scene_grid_size=scene_grid_size,
        )  # [B,1,Hs,Ws]
        heatmap_logits = self._upsample(coarse_logits)  # [B,1,H,W]
        heatmap = torch.sigmoid(heatmap_logits) if self.apply_sigmoid else heatmap_logits

        patch_weights = torch.softmax(patch_logits, dim=-1)
        map_for_point = heatmap[:, 0, :, :]
        point_soft = softargmax2d(map_for_point)
        point_hard = argmax2d(map_for_point)

        select_soft = self.training if use_softargmax is None else bool(use_softargmax)
        point = point_soft if select_soft else point_hard

        return {
            "patch_logits": patch_logits,
            "patch_weights": patch_weights,
            "coarse_logits": coarse_logits,
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "point": point,
            "point_soft": point_soft,
            "point_hard": point_hard,
            "scene_grid_hw": torch.tensor(scene_hw, dtype=torch.long, device=scene_hidden.device),
        }
