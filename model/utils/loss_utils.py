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
    if (not torch.is_tensor(pred_object_emb)) or pred_object_emb.dim() != 2:
        z = torch.zeros((), device=pred_object_emb.device if torch.is_tensor(pred_object_emb) else "cpu")
        return z, 0
    if (not torch.is_tensor(label_embedding_bank)) or label_embedding_bank.dim() != 2:
        z = torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype)
        return z, 0
    if (not torch.is_tensor(target_label)) or target_label.dim() != 1:
        z = torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype)
        return z, 0

    bank = label_embedding_bank.to(device=pred_object_emb.device, dtype=pred_object_emb.dtype)
    if int(bank.shape[0]) <= 0 or int(bank.shape[1]) != int(pred_object_emb.shape[1]):
        z = torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype)
        return z, 0

    y = target_label.to(device=pred_object_emb.device, dtype=torch.long).flatten()
    if int(y.numel()) != int(pred_object_emb.shape[0]):
        z = torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype)
        return z, 0

    valid = (y >= 0) & (y < int(bank.shape[0]))
    if torch.is_tensor(target_object_valid):
        v = target_object_valid.to(device=pred_object_emb.device, dtype=torch.float32).flatten()
        if int(v.numel()) == int(valid.numel()):
            valid = valid & v.gt(0)
    n_valid = int(valid.sum().item())
    if n_valid <= 0:
        z = torch.zeros((), device=pred_object_emb.device, dtype=pred_object_emb.dtype)
        return z, 0

    q = F.normalize(pred_object_emb[valid], p=2, dim=-1)
    k = F.normalize(bank, p=2, dim=-1)
    logits = q @ k.t()
    t = max(float(temperature), 1e-6)
    logits = logits / t
    loss = F.cross_entropy(logits, y[valid], reduction="mean")
    return loss, n_valid


def point_regression_loss(
    pred_point_xy: torch.Tensor | None,
    target_point: torch.Tensor | None,
    target_point_valid: torch.Tensor | None,
    point_head_valid: torch.Tensor | None,
    loss_type: str = "smooth_l1",
) -> tuple[torch.Tensor, int]:
    """Compute regression loss between predicted and GT normalized gaze coordinates.

    Args:
        pred_point_xy      : [B, 2] predicted (x, y) ∈ [0, 1]² from point head (sigmoid output)
        target_point       : [B, 2] GT normalized gaze coordinates ∈ [0, 1]²
        target_point_valid : [B] float — 1.0 if this sample has a valid GT point
        point_head_valid   : [B] bool  — True if the point_span_mask had tokens for this sample
        loss_type          : "smooth_l1" (SmoothL1Loss) or "mse" (MSELoss)

    Returns:
        (loss scalar, n_valid_samples)
    """
    dev = "cpu"
    dtype = torch.float32
    if torch.is_tensor(pred_point_xy):
        dev = pred_point_xy.device
        dtype = pred_point_xy.dtype

    z = torch.zeros((), device=dev, dtype=dtype)

    if not torch.is_tensor(pred_point_xy) or pred_point_xy.dim() != 2 or int(pred_point_xy.shape[1]) != 2:
        return z, 0
    if not torch.is_tensor(target_point) or target_point.dim() != 2 or int(target_point.shape[1]) != 2:
        return z, 0

    bsz = int(pred_point_xy.shape[0])
    if int(target_point.shape[0]) != bsz:
        return z, 0

    # Build validity mask: both point_head pooling succeeded AND GT is valid.
    valid = torch.ones((bsz,), dtype=torch.bool, device=dev)
    if torch.is_tensor(point_head_valid):
        phv = point_head_valid.to(device=dev, dtype=torch.bool).flatten()
        if int(phv.numel()) == bsz:
            valid = valid & phv
    if torch.is_tensor(target_point_valid):
        tpv = target_point_valid.to(device=dev, dtype=torch.float32).flatten()
        if int(tpv.numel()) == bsz:
            valid = valid & tpv.gt(0)

    n_valid = int(valid.sum().item())
    if n_valid <= 0:
        return z, 0

    pred_v = pred_point_xy[valid].to(dtype=torch.float32)
    tgt_v = target_point[valid].to(device=dev, dtype=torch.float32)

    lt = str(loss_type or "smooth_l1").strip().lower()
    if lt == "mse":
        loss = F.mse_loss(pred_v, tgt_v, reduction="mean")
    else:
        # SmoothL1 (beta=1.0) by default
        loss = F.smooth_l1_loss(pred_v, tgt_v, reduction="mean")

    return loss.to(dtype=dtype), n_valid


def compute_structured_losses(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_mask_answer: torch.Tensor | None,
    loss_mask_point: torch.Tensor | None,
    loss_mask_object: torch.Tensor | None,
    pred_object_emb: torch.Tensor | None = None,
    target_label: torch.Tensor | None = None,
    target_object_valid: torch.Tensor | None = None,
    label_embedding_bank: torch.Tensor | None = None,
    object_loss_mode: str = "token_ce",
    object_temperature: float = 0.07,
    weight_answer: float,
    weight_point: float,
    weight_object: float,
    weight_slot: float = 0.0,
    # ── Point regression head ──────────────────────────────────────────────
    pred_point_xy: torch.Tensor | None = None,
    target_point: torch.Tensor | None = None,
    target_point_valid: torch.Tensor | None = None,
    point_head_valid: torch.Tensor | None = None,
    weight_point_reg: float = 0.0,
    point_reg_loss_type: str = "smooth_l1",
    # ──────────────────────────────────────────────────────────────────────
    fallback_loss: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute all structured losses and return a dict with individual components.

    Loss components:
        L_text   : masked NLL over full answer tokens (weight_answer)
        L_point_ce : masked NLL over Point x/y coordinate tokens (weight_point, usually 0)
        L_object : retrieval CE over object-slot hidden state (weight_object)
        L_slot   : masked NLL over slot tokens (weight_slot, legacy)
        L_point_reg : SmoothL1 / MSE regression on point head output (weight_point_reg)

    Total loss = w_answer * L_text + w_point * L_point_ce + w_object * L_object
                 + w_slot * L_slot + w_point_reg * L_point_reg
    """
    device = labels.device
    dtype = torch.float32
    if torch.is_tensor(logits):
        device = logits.device
        dtype = logits.dtype

    z = torch.zeros((), device=device, dtype=dtype)
    has_logits = torch.is_tensor(logits) and logits.dim() == 3
    if has_logits:
        l_answer, n_answer = masked_token_ce(logits, labels, loss_mask_answer)
        l_point, n_point = masked_token_ce(logits, labels, loss_mask_point)
    else:
        l_answer, n_answer = z, 0
        l_point, n_point = z, 0

    mode = str(object_loss_mode or "token_ce").strip().lower()
    l_slot, n_slot = z, 0
    if mode in {"retrieval", "infonce", "full_bank_retrieval"}:
        l_obj, n_obj = retrieval_ce_full_bank(
            pred_object_emb=pred_object_emb,
            target_label=target_label,
            target_object_valid=target_object_valid,
            label_embedding_bank=label_embedding_bank,
            temperature=float(object_temperature),
        )
        # Slot token CE: forces the model to actually generate <obj_emb> at inference time.
        # Without this, the retrieval loss only trains the hidden-state projector but
        # applies no direct LM pressure to emit the slot token.
        if has_logits and float(weight_slot) > 0.0:
            l_slot, n_slot = masked_token_ce(logits, labels, loss_mask_object)
    else:
        if has_logits:
            l_obj, n_obj = masked_token_ce(logits, labels, loss_mask_object)
        else:
            l_obj, n_obj = z, 0

    # ── Point regression head loss ─────────────────────────────────────────
    l_point_reg, n_point_reg = point_regression_loss(
        pred_point_xy=pred_point_xy,
        target_point=target_point,
        target_point_valid=target_point_valid,
        point_head_valid=point_head_valid,
        loss_type=str(point_reg_loss_type),
    )
    l_point_reg = l_point_reg.to(device=device, dtype=dtype)

    w_answer = float(weight_answer)
    w_point = float(weight_point)
    w_obj = float(weight_object)
    w_slot = float(weight_slot)
    w_point_reg = float(weight_point_reg)
    total = (
        (w_answer * l_answer)
        + (w_point * l_point)
        + (w_obj * l_obj)
        + (w_slot * l_slot)
        + (w_point_reg * l_point_reg)
    )

    used_fallback = False
    if (n_answer + n_point + n_obj + n_slot + n_point_reg) <= 0:
        if torch.is_tensor(fallback_loss):
            total = fallback_loss.to(device=device, dtype=dtype)
            used_fallback = True
        else:
            total = z

    return {
        "loss": total,
        "loss_total": total,
        "loss_answer": l_answer,
        "loss_point": l_point,
        "loss_object": l_obj,
        "loss_slot": l_slot,
        "loss_point_reg": l_point_reg,
        "n_answer_tokens": int(n_answer),
        "n_point_tokens": int(n_point),
        "n_object_tokens": int(n_obj),
        "n_slot_tokens": int(n_slot),
        "n_point_reg_samples": int(n_point_reg),
        "used_fallback": bool(used_fallback),
    }
