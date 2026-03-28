from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_bce_loss(
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    pred_heatmap_logits: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = pred_heatmap_logits
    if logits is None:
        probs = pred_heatmap.clamp(min=1e-6, max=1.0 - 1e-6)
        logits = torch.logit(probs)
    return F.binary_cross_entropy_with_logits(
        logits,
        target_heatmap.to(dtype=logits.dtype, device=logits.device),
        reduction=reduction,
    )


def compute_localization_loss(
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    pred_heatmap_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    l_hm = heatmap_bce_loss(
        pred_heatmap=pred_heatmap,
        target_heatmap=target_heatmap,
        pred_heatmap_logits=pred_heatmap_logits,
    )
    return {"loss": l_hm, "l_hm": l_hm}
