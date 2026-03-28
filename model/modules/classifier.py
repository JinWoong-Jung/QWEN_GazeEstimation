from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SemGazeStyleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        n = int(max(1, num_layers))
        h = [int(hidden_dim)] * (n - 1)
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip([int(input_dim)] + h, h + [int(output_dim)])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class GazeRecognitionClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int | None = None,
        output_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        mlp_num_layers: int = 2,
        dropout: float = 0.1,
        use_subject_context: bool = True,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.use_subject_context = bool(use_subject_context)
        in_dim = hidden_dim * 3 if self.use_subject_context else hidden_dim
        out_dim = int(output_dim or hidden_dim)
        inner = int(mlp_hidden_dim or (hidden_dim * 2))
        n_layers = int(max(1, mlp_num_layers))
        self.normalize_output = bool(normalize_output)
        if n_layers == 2:
            self.projector = nn.Sequential(
                nn.Linear(in_dim, inner),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(inner, out_dim),
            )
        else:
            # SemGaze label_mlp style: ReLU MLP without dropout, typically 6 layers.
            self.projector = _SemGazeStyleMLP(
                input_dim=in_dim,
                hidden_dim=inner,
                output_dim=out_dim,
                num_layers=n_layers,
            )
        self.output_dim = out_dim
        self.classifier = (
            nn.Linear(out_dim, int(num_classes))
            if (num_classes is not None and int(num_classes) > 0)
            else None
        )

    def _weighted_pool(self, scene_hidden: torch.Tensor, patch_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_hidden.dim() != 3:
            raise ValueError(f"scene_hidden must be [B, N, D], got shape={tuple(scene_hidden.shape)}")
        if patch_logits.dim() != 2:
            raise ValueError(f"patch_logits must be [B, N], got shape={tuple(patch_logits.shape)}")
        if scene_hidden.shape[:2] != patch_logits.shape:
            raise ValueError("scene_hidden and patch_logits token count must match")

        weights = torch.softmax(patch_logits, dim=-1)  # [B, N]
        f_g = torch.sum(scene_hidden * weights.unsqueeze(-1), dim=1)  # [B, D]
        return f_g, weights

    def forward(
        self,
        scene_hidden: torch.Tensor,
        patch_logits: torch.Tensor,
        head_token: torch.Tensor | None = None,
        text_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        f_g, weights = self._weighted_pool(scene_hidden, patch_logits)

        if self.use_subject_context:
            if head_token is None or text_token is None:
                raise ValueError("head_token and text_token are required when use_subject_context=True")
            if head_token.dim() != 2 or text_token.dim() != 2:
                raise ValueError("head_token/text_token must be [B, D]")
            fused_input = torch.cat([f_g, head_token, text_token], dim=-1)
        else:
            fused_input = f_g

        emb = self.projector(fused_input)
        if self.normalize_output:
            emb = F.normalize(emb, p=2, dim=-1)
        out: dict[str, torch.Tensor] = {
            "patch_weights": weights,
            "f_g": f_g,
            "f_r": emb,
        }
        if self.classifier is not None:
            logits = self.classifier(emb)
            pred = torch.argmax(logits, dim=-1)
            out["logits"] = logits
            out["pred"] = pred
        return out
