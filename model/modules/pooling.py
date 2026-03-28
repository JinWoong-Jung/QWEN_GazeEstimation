from __future__ import annotations

import torch
import torch.nn as nn


class TokenPooler(nn.Module):
    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"mean", "cls"}:
            raise ValueError(f"unsupported pooling mode: {mode}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be [B, N, D], got shape={tuple(hidden.shape)}")
        if self.mode == "cls":
            return hidden[:, 0, :]
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
    ) -> dict[str, torch.Tensor]:
        z_h = self.head_pool(head_hidden)
        z_t = self.text_pool(text_hidden)
        z = self.fuser(torch.cat([z_h, z_t], dim=-1))
        return {"z_h": z_h, "z_t": z_t, "z": z}
