from __future__ import annotations

import torch
import torch.nn as nn


class TokenPooler(nn.Module):
    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"mean", "cls"}:
            raise ValueError(f"unsupported pooling mode: {mode}")

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be [B, N, D], got shape={tuple(hidden.shape)}")
        if self.mode == "cls":
            return hidden[:, 0, :]
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(f"mask must be [B, N], got shape={tuple(mask.shape)}")
            if mask.shape[:2] != hidden.shape[:2]:
                raise ValueError(
                    f"mask/hidden shape mismatch: mask={tuple(mask.shape)} hidden={tuple(hidden.shape)}"
                )
            w = mask.to(device=hidden.device, dtype=hidden.dtype).clamp(min=0.0)
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (hidden * w.unsqueeze(-1)).sum(dim=1) / denom
        return hidden.mean(dim=1)


class SubjectSummary(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pool_mode: str = "mean",
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head_pool = TokenPooler(pool_mode)
        self.text_pool = TokenPooler(pool_mode)

        inner = int(mlp_hidden_dim or (hidden_dim * 2))
        self.fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
        )

    def forward(
        self,
        head_hidden: torch.Tensor,
        text_hidden: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_h = self.head_pool(head_hidden, mask=head_mask)
        z_t = self.text_pool(text_hidden, mask=text_mask)
        z = self.fuser(torch.cat([z_h, z_t], dim=-1))
        return {"z_h": z_h, "z_t": z_t, "z": z}
