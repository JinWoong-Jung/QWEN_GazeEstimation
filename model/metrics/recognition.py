from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_ce_loss(
    logits: torch.Tensor,
    target_label: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        target_label,
        label_smoothing=float(label_smoothing),
        ignore_index=int(ignore_index),
    )


def info_nce_batch_local_loss(
    emb_pred: torch.Tensor,
    emb_gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    logit_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        mask = torch.ones((emb_pred.shape[0],), dtype=torch.bool, device=emb_pred.device)
    else:
        mask = valid_mask.to(device=emb_pred.device).bool().view(-1)
    if emb_pred.shape[0] != emb_gt.shape[0]:
        raise ValueError(
            f"emb_pred/emb_gt batch mismatch: pred={tuple(emb_pred.shape)} gt={tuple(emb_gt.shape)}"
        )
    if mask.shape[0] != emb_pred.shape[0]:
        raise ValueError(
            f"valid_mask shape mismatch: mask={tuple(mask.shape)} pred_batch={emb_pred.shape[0]}"
        )
    if int(mask.sum().item()) <= 0:
        return torch.zeros((), device=emb_pred.device, dtype=emb_pred.dtype)

    pred = emb_pred[mask]
    gt = emb_gt[mask].to(dtype=pred.dtype, device=pred.device)
    unique_gt, labels = torch.unique(gt, dim=0, return_inverse=True)
    logits = torch.matmul(pred, unique_gt.t())
    if logit_scale is not None:
        logits = logits * logit_scale.exp().to(dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits, labels.long())
