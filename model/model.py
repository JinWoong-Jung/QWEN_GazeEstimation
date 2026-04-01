from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class QwenTextGenerationModel(nn.Module):
    """Minimal wrapper: image+prompt -> Qwen(LoRA) -> text tokens/logits."""

    def __init__(self, qwen_model: nn.Module) -> None:
        super().__init__()
        self.qwen = qwen_model

    def forward(
        self,
        *,
        joint_inputs: dict[str, Any],
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        kwargs = dict(joint_inputs)
        kwargs["return_dict"] = True
        kwargs["use_cache"] = bool(use_cache)
        if labels is not None:
            kwargs["labels"] = labels
        out = self.qwen(**kwargs)
        return {
            "loss": getattr(out, "loss", None),
            "logits": getattr(out, "logits", None),
        }

    def generate(
        self,
        *,
        joint_inputs: dict[str, Any],
        max_new_tokens: int = 16,
        do_sample: bool = False,
        num_beams: int = 1,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
        kwargs = dict(joint_inputs)
        return self.qwen.generate(
            **kwargs,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=bool(do_sample),
            num_beams=max(1, int(num_beams)),
            num_return_sequences=max(1, int(num_return_sequences)),
            use_cache=True,
        )


# Backward-compatible alias for older imports.
QwenGazeIntegratedModel = QwenTextGenerationModel
