from __future__ import annotations

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        inner = int(mlp_hidden_dim or (hidden_dim * 2))
        self.param_gen = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim * 2),
        )

    def forward(self, scene_hidden: torch.Tensor, subject_token: torch.Tensor) -> dict[str, torch.Tensor]:
        if scene_hidden.dim() != 3:
            raise ValueError(f"scene_hidden must be [B, N, D], got shape={tuple(scene_hidden.shape)}")
        if subject_token.dim() != 2:
            raise ValueError(f"subject_token must be [B, D], got shape={tuple(subject_token.shape)}")

        gamma_beta = self.param_gen(subject_token)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        conditioned = gamma * scene_hidden + beta
        return {"scene_hidden": conditioned, "gamma": gamma, "beta": beta}


class CrossAttentionConditioner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pair_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.use_pair_tokens = bool(use_pair_tokens)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scene_hidden: torch.Tensor,
        subject_token: torch.Tensor | None = None,
        head_token: torch.Tensor | None = None,
        text_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if scene_hidden.dim() != 3:
            raise ValueError(f"scene_hidden must be [B, N, D], got shape={tuple(scene_hidden.shape)}")

        if self.use_pair_tokens and head_token is not None and text_token is not None:
            if head_token.dim() != 2 or text_token.dim() != 2:
                raise ValueError("head_token and text_token must be [B, D]")
            kv = torch.stack([head_token, text_token], dim=1)
        else:
            if subject_token is None or subject_token.dim() != 2:
                raise ValueError("subject_token must be [B, D] when pair tokens are not used")
            kv = subject_token.unsqueeze(1)

        attended, attn_weights = self.attn(query=scene_hidden, key=kv, value=kv, need_weights=True)
        conditioned = self.norm(scene_hidden + attended)
        return {"scene_hidden": conditioned, "attn_weights": attn_weights}


class SubjectConditioning(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mode: str = "film",
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pair_tokens: bool = True,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.mode = str(mode).strip().lower()
        self.num_layers = max(1, int(num_layers))
        if self.mode not in {"film", "cross_attn", "cross_attention"}:
            raise ValueError(f"unsupported conditioning mode: {mode}")

        def _build_impl() -> nn.Module:
            if self.mode == "film":
                return FiLMConditioner(hidden_dim=hidden_dim, dropout=dropout)
            return CrossAttentionConditioner(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_pair_tokens=use_pair_tokens,
            )

        self.layers = nn.ModuleList([_build_impl() for _ in range(self.num_layers)])

    def forward(
        self,
        scene_hidden: torch.Tensor,
        subject_token: torch.Tensor,
        head_token: torch.Tensor | None = None,
        text_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = scene_hidden
        out: dict[str, torch.Tensor] = {"scene_hidden": hidden}
        for layer in self.layers:
            if self.mode == "film":
                out = layer(scene_hidden=hidden, subject_token=subject_token)
            else:
                out = layer(
                    scene_hidden=hidden,
                    subject_token=subject_token,
                    head_token=head_token,
                    text_token=text_token,
                )
            hidden = out["scene_hidden"]
        out["scene_hidden"] = hidden
        return out
