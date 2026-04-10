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
