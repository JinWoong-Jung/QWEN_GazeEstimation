from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    if mask is None:
        return torch.zeros((), device=logits.device, dtype=logits.dtype), 0
    if (not torch.is_tensor(mask)) or mask.dim() != 2:
        return torch.zeros((), device=logits.device, dtype=logits.dtype), 0

    m = mask.to(device=logits.device, dtype=torch.bool)
    y = labels.to(device=logits.device)
    # Causal LM next-token alignment: logits[:, t] predicts labels[:, t+1].
    shift_logits = logits[:, :-1, :]
    shift_labels = y[:, 1:]
    shift_mask = m[:, 1:]

    valid = shift_mask & shift_labels.ne(-100)
    n_valid = int(valid.sum().item())
    if n_valid <= 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype), 0

    ce = F.cross_entropy(
        shift_logits.reshape(-1, int(shift_logits.shape[-1])),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    return ce[valid].mean(), n_valid


def retrieval_ce_full_bank(
    pred_object_emb: torch.Tensor | None,
    target_label: torch.Tensor | None,
    target_object_valid: torch.Tensor | None,
    label_embedding_bank: torch.Tensor | None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, int]:
    if (not torch.is_tensor(pred_object_emb)) or pred_object_emb.dim() != 2:
        dev = pred_object_emb.device if torch.is_tensor(pred_object_emb) else "cpu"
        return torch.zeros((), device=dev), 0
    if (not torch.is_tensor(label_embedding_bank)) or label_embedding_bank.dim() != 2:
        return torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype), 0
    if (not torch.is_tensor(target_label)) or target_label.dim() != 1:
        return torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype), 0

    bank = label_embedding_bank.to(device=pred_object_emb.device, dtype=pred_object_emb.dtype)
    if int(bank.shape[0]) <= 0 or int(bank.shape[1]) != int(pred_object_emb.shape[1]):
        return torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype), 0

    y = target_label.to(device=pred_object_emb.device, dtype=torch.long).flatten()
    if int(y.numel()) != int(pred_object_emb.shape[0]):
        return torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype), 0

    valid = (y >= 0) & (y < int(bank.shape[0]))
    if torch.is_tensor(target_object_valid):
        v = target_object_valid.to(device=pred_object_emb.device, dtype=torch.float32).flatten()
        if int(v.numel()) == int(valid.numel()):
            valid = valid & v.gt(0)
    n_valid = int(valid.sum().item())
    if n_valid <= 0:
        return torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype), 0

    q = F.normalize(pred_object_emb[valid], p=2, dim=-1)
    k = F.normalize(bank, p=2, dim=-1)
    logits = q @ k.t() / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, y[valid], reduction="mean"), n_valid


def compute_structured_losses(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_mask_fmt: torch.Tensor | None,
    loss_mask_point: torch.Tensor | None,
    loss_mask_object: torch.Tensor | None,
    pred_object_emb: torch.Tensor | None = None,
    target_label: torch.Tensor | None = None,
    target_object_valid: torch.Tensor | None = None,
    label_embedding_bank: torch.Tensor | None = None,
    object_temperature: float = 0.07,
    weight_fmt: float,
    weight_point: float,
    weight_object: float,
    fallback_loss: torch.Tensor | None = None,
) -> dict[str, Any]:
    device = labels.device
    dtype = torch.float32
    if torch.is_tensor(logits):
        device = logits.device
        dtype = logits.dtype

    z = torch.zeros((), device=device, dtype=dtype)
    has_logits = torch.is_tensor(logits) and logits.dim() == 3
    if has_logits:
        l_fmt, n_fmt = masked_token_ce(logits, labels, loss_mask_fmt)
        l_point, n_point = masked_token_ce(logits, labels, loss_mask_point)
    else:
        l_fmt, n_fmt = z, 0
        l_point, n_point = z, 0

    # L_obj-ret: InfoNCE retrieval loss on <obj_emb> hidden state.
    l_obj, n_obj = retrieval_ce_full_bank(
        pred_object_emb=pred_object_emb,
        target_label=target_label,
        target_object_valid=target_object_valid,
        label_embedding_bank=label_embedding_bank,
        temperature=float(object_temperature),
    )

    total = float(weight_fmt) * l_fmt + float(weight_point) * l_point + float(weight_object) * l_obj

    used_fallback = False
    if (n_fmt + n_point + n_obj) <= 0:
        if torch.is_tensor(fallback_loss):
            total = fallback_loss.to(device=device, dtype=dtype)
            used_fallback = True
        else:
            total = z

    return {
        "loss": total,
        "loss_total": total,
        "loss_fmt": l_fmt,
        "loss_point": l_point,
        "loss_object": l_obj,
        "n_fmt_tokens": int(n_fmt),
        "n_point_tokens": int(n_point),
        "n_object_tokens": int(n_obj),
        "used_fallback": bool(used_fallback),
    }
