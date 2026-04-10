from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class QwenTextGenerationModel(nn.Module):
    """Minimal LoRA wrapper: image + prompt → Qwen → text logits.

    No auxiliary heads (object projector, point head) — pure autoregressive
    text generation is the single objective.
    """

    def __init__(self, qwen_model: nn.Module) -> None:
        super().__init__()
        self.qwen = qwen_model

    def forward(
        self,
        *,
        joint_inputs: dict[str, Any],
        use_cache: bool = False,
    ) -> dict[str, Any]:
        kwargs = dict(joint_inputs)
        kwargs["return_dict"] = True
        kwargs["use_cache"] = bool(use_cache)
        out = self.qwen(**kwargs)
        return {
            "logits": getattr(out, "logits", None),
        }

    def generate(
        self,
        *,
        joint_inputs: dict[str, Any],
        max_new_tokens: int = 16,
        do_sample: bool = False,
        num_beams: int = 1,
        **generate_kwargs: Any,
    ) -> torch.Tensor | Any:
        kwargs = dict(joint_inputs)
        return self.qwen.generate(
            **kwargs,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=bool(do_sample),
            num_beams=max(1, int(num_beams)),
            use_cache=True,
            **generate_kwargs,
        )


# Backward-compatible alias.
QwenGazeIntegratedModel = QwenTextGenerationModel
