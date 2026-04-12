from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    """Causal-LM cross-entropy restricted to tokens where mask is True.

    Returns (loss_scalar, n_valid_tokens). Returns (0, 0) when no valid tokens.
    """
    if mask is None:
        z = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return z, 0
    if (not torch.is_tensor(mask)) or mask.dim() != 2:
        z = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return z, 0

    m = mask.to(device=logits.device, dtype=torch.bool)
    y = labels.to(device=logits.device)
    # Causal LM next-token alignment: logits[:, t] predicts labels[:, t+1].
    shift_logits = logits[:, :-1, :]
    shift_labels = y[:, 1:]
    shift_mask = m[:, 1:]

    valid = shift_mask & shift_labels.ne(-100)
    n_valid = int(valid.sum().item())
    if n_valid <= 0:
        z = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return z, 0

    ce = F.cross_entropy(
        shift_logits.reshape(-1, int(shift_logits.shape[-1])),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    loss = ce[valid].mean()
    return loss, n_valid


def retrieval_ce_full_bank(
    pred_object_emb: torch.Tensor | None,
    target_label: torch.Tensor | None,
    target_object_valid: torch.Tensor | None,
    label_embedding_bank: torch.Tensor | None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, int]:
    """InfoNCE-style retrieval CE over the full label bank.

    pred_object_emb [B, D] (L2-normalized) × label_embedding_bank [V, D] → cross-entropy.
    Samples where target_object_valid <= 0 or target_label < 0 or >= V are excluded.

    Returns (loss_scalar, n_valid_samples).
    """
    z = torch.zeros((), dtype=torch.float32)
    if not torch.is_tensor(pred_object_emb) or pred_object_emb.dim() != 2:
        return z, 0
    if not torch.is_tensor(label_embedding_bank) or label_embedding_bank.dim() != 2:
        return z, 0
    if not torch.is_tensor(target_label):
        return z, 0

    device = pred_object_emb.device
    z = z.to(device)
    V = int(label_embedding_bank.shape[0])
    bank = label_embedding_bank.to(device=device, dtype=pred_object_emb.dtype)
    tgt = target_label.to(device=device)

    valid_mask = tgt.ge(0) & tgt.lt(V)
    if torch.is_tensor(target_object_valid):
        valid_mask = valid_mask & target_object_valid.to(device=device, dtype=torch.float32).gt(0)

    n_valid = int(valid_mask.sum().item())
    if n_valid <= 0:
        return z, 0

    pred = pred_object_emb[valid_mask]
    labels = tgt[valid_mask]
    t = max(float(temperature), 1e-6)
    logits = (pred @ bank.t()) / t  # [n_valid, V]
    loss = F.cross_entropy(logits, labels)
    return loss, n_valid


def compute_answer_loss(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_mask_answer: torch.Tensor | None,
    weight_answer: float,
) -> dict[str, Any]:
    """Compute masked NLL over answer tokens (sole training objective).

    Returns dict with keys: loss, loss_answer, n_answer_tokens.
    """
    device = labels.device
    dtype = torch.float32
    if torch.is_tensor(logits):
        device = logits.device
        dtype = logits.dtype
    z = torch.zeros((), device=device, dtype=dtype)

    if torch.is_tensor(logits) and logits.dim() == 3:
        l_answer, n_answer = masked_token_ce(logits, labels, loss_mask_answer)
    else:
        l_answer, n_answer = z, 0

    w = float(weight_answer)
    total = w * l_answer if n_answer > 0 else z

    return {
        "loss": total,
        "loss_answer": l_answer,
        "n_answer_tokens": int(n_answer),
    }
