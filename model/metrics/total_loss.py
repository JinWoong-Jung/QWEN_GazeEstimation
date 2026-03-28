from __future__ import annotations

import torch

from ..recognition_objectives import (
    is_batch_local_infonce_objective,
    is_full_vocab_contrastive_objective,
    normalize_recognition_objective,
)
from .localization import compute_localization_loss
from .recognition import classification_ce_loss, info_nce_batch_local_loss


def compute_total_loss(
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    pred_heatmap_logits: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    target_label: torch.Tensor | None = None,
    pred_label_emb: torch.Tensor | None = None,
    target_label_emb: torch.Tensor | None = None,
    target_label_valid: torch.Tensor | None = None,
    *,
    lambda_cls: float = 1.0,
    label_smoothing: float = 0.0,
    cls_ignore_index: int = -100,
    recognition_objective: str = "ce",
    logit_scale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    loc = compute_localization_loss(
        pred_heatmap=pred_heatmap,
        target_heatmap=target_heatmap,
        pred_heatmap_logits=pred_heatmap_logits,
    )
    loss = loc["loss"]
    out: dict[str, torch.Tensor] = {
        "loss": loss,
        "l_hm": loc["l_hm"],
    }

    obj = normalize_recognition_objective(recognition_objective)
    if is_batch_local_infonce_objective(obj):
        if pred_label_emb is not None and target_label_emb is not None:
            l_cls = info_nce_batch_local_loss(
                emb_pred=pred_label_emb,
                emb_gt=target_label_emb,
                valid_mask=target_label_valid,
                logit_scale=logit_scale,
            )
            loss = loss + float(lambda_cls) * l_cls
            out["loss"] = loss
            out["l_cls"] = l_cls
    elif is_full_vocab_contrastive_objective(obj):
        if logits is not None and target_label is not None:
            l_cls = classification_ce_loss(
                logits=logits,
                target_label=target_label,
                label_smoothing=label_smoothing,
                ignore_index=cls_ignore_index,
            )
            loss = loss + float(lambda_cls) * l_cls
            out["loss"] = loss
            out["l_cls"] = l_cls
        elif pred_label_emb is not None and target_label_emb is not None:
            # Fallback for runs without ready full-vocab logits.
            l_cls = info_nce_batch_local_loss(
                emb_pred=pred_label_emb,
                emb_gt=target_label_emb,
                valid_mask=target_label_valid,
                logit_scale=logit_scale,
            )
            loss = loss + float(lambda_cls) * l_cls
            out["loss"] = loss
            out["l_cls"] = l_cls
    else:
        if logits is not None and target_label is not None:
            l_cls = classification_ce_loss(
                logits=logits,
                target_label=target_label,
                label_smoothing=label_smoothing,
                ignore_index=cls_ignore_index,
            )
            loss = loss + float(lambda_cls) * l_cls
            out["loss"] = loss
            out["l_cls"] = l_cls

    return out
