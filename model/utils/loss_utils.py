from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _masked_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
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


def compute_structured_losses(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_mask_answer: torch.Tensor | None,
    loss_mask_point: torch.Tensor | None,
    loss_mask_objectid: torch.Tensor | None,
    weight_answer: float,
    weight_point: float,
    weight_objectid: float,
    fallback_loss: torch.Tensor | None = None,
) -> dict[str, Any]:
    device = labels.device
    dtype = torch.float32
    if torch.is_tensor(logits):
        device = logits.device
        dtype = logits.dtype

    z = torch.zeros((), device=device, dtype=dtype)
    if (not torch.is_tensor(logits)) or logits.dim() != 3:
        out_loss = fallback_loss if torch.is_tensor(fallback_loss) else z
        return {
            "loss": out_loss,
            "loss_answer": z,
            "loss_localization": z,
            "loss_recognition": z,
            "n_answer_tokens": 0,
            "n_point_tokens": 0,
            "n_objectid_tokens": 0,
            "used_fallback": True,
        }

    l_answer, n_answer = _masked_token_ce(logits, labels, loss_mask_answer)
    l_point, n_point = _masked_token_ce(logits, labels, loss_mask_point)
    l_obj, n_obj = _masked_token_ce(logits, labels, loss_mask_objectid)

    w_answer = float(weight_answer)
    w_point = float(weight_point)
    w_obj = float(weight_objectid)
    total = (w_answer * l_answer) + (w_point * l_point) + (w_obj * l_obj)

    used_fallback = False
    if (n_answer + n_point + n_obj) <= 0:
        if torch.is_tensor(fallback_loss):
            total = fallback_loss.to(device=device, dtype=dtype)
            used_fallback = True
        else:
            total = z

    return {
        "loss": total,
        "loss_answer": l_answer,
        "loss_localization": l_point,
        "loss_recognition": l_obj,
        "n_answer_tokens": int(n_answer),
        "n_point_tokens": int(n_point),
        "n_objectid_tokens": int(n_obj),
        "used_fallback": bool(used_fallback),
    }

