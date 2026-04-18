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


def masked_sample_exact_rate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    """Teacher-forced exact-match rate over masked tokens, computed per sample.

    A sample counts as correct iff all valid masked next-token predictions match
    the ground-truth labels. Returns (rate_scalar, n_valid_samples).
    """
    if mask is None:
        z = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return z, 0
    if (not torch.is_tensor(mask)) or mask.dim() != 2:
        z = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return z, 0

    m = mask.to(device=logits.device, dtype=torch.bool)
    y = labels.to(device=logits.device)
    shift_logits = logits[:, :-1, :]
    shift_labels = y[:, 1:]
    shift_mask = m[:, 1:]

    valid = shift_mask & shift_labels.ne(-100)
    valid_samples = valid.any(dim=1)
    n_valid_samples = int(valid_samples.sum().item())
    if n_valid_samples <= 0:
        z = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return z, 0

    preds = shift_logits.argmax(dim=-1)
    token_correct = preds.eq(shift_labels) | (~valid)
    sample_exact = token_correct.all(dim=1) & valid_samples
    rate = sample_exact[valid_samples].to(dtype=logits.dtype).mean()
    return rate, n_valid_samples


def compute_structured_loss(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_mask_point: torch.Tensor | None,
    loss_mask_object: torch.Tensor | None,
    loss_mask_format: torch.Tensor | None,
    weight_point: float = 1.0,
    weight_object: float = 1.0,
    weight_format: float = 0.25,
) -> dict[str, Any]:
    """Compute L_SFT = w_p * L_point + w_o * L_object + w_f * L_format.

    Returns dict with keys:
        loss, loss_point, loss_object, loss_format,
        n_point_tokens, n_object_tokens, n_format_tokens,
        format_valid_rate, n_format_samples
    """
    device = labels.device
    dtype = torch.float32
    if torch.is_tensor(logits):
        device = logits.device
        dtype = logits.dtype
    z = torch.zeros((), device=device, dtype=dtype)

    if not (torch.is_tensor(logits) and logits.dim() == 3):
        return {
            "loss": z,
            "loss_point": z,
            "loss_object": z,
            "loss_format": z,
            "n_point_tokens": 0,
            "n_object_tokens": 0,
            "n_format_tokens": 0,
            "format_valid_rate": z,
            "n_format_samples": 0,
        }

    l_pt, n_pt = masked_token_ce(logits, labels, loss_mask_point)
    l_obj, n_obj = masked_token_ce(logits, labels, loss_mask_object)
    l_fmt, n_fmt = masked_token_ce(logits, labels, loss_mask_format)
    fmt_valid_rate, n_fmt_samples = masked_sample_exact_rate(logits, labels, loss_mask_format)

    wp, wo, wf = float(weight_point), float(weight_object), float(weight_format)
    total = z
    if n_pt > 0:
        total = total + wp * l_pt
    if n_obj > 0:
        total = total + wo * l_obj
    if n_fmt > 0:
        total = total + wf * l_fmt

    return {
        "loss": total,
        "loss_point": l_pt,
        "loss_object": l_obj,
        "loss_format": l_fmt,
        "n_point_tokens": int(n_pt),
        "n_object_tokens": int(n_obj),
        "n_format_tokens": int(n_fmt),
        "format_valid_rate": fmt_valid_rate,
        "n_format_samples": int(n_fmt_samples),
    }


def compute_answer_loss(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_mask_answer: torch.Tensor | None = None,
    loss_mask_point: torch.Tensor | None = None,
    loss_mask_object: torch.Tensor | None = None,
    loss_mask_format: torch.Tensor | None = None,
    weight_answer: float = 1.0,
    weight_point: float = 1.0,
    weight_object: float = 1.0,
    weight_format: float = 0.25,
) -> dict[str, Any]:
    """Compute structured SFT loss when structured masks are present,
    else fall back to full-answer CE (legacy).

    Returns dict with keys: loss, loss_point, loss_object, loss_format,
    loss_answer, n_point_tokens, n_object_tokens, n_format_tokens, n_answer_tokens.
    """
    device = labels.device
    dtype = torch.float32
    if torch.is_tensor(logits):
        device = logits.device
        dtype = logits.dtype
    z = torch.zeros((), device=device, dtype=dtype)

    has_structured = (
        torch.is_tensor(loss_mask_point) and torch.is_tensor(loss_mask_object)
        and torch.is_tensor(loss_mask_format)
    )

    if has_structured:
        d = compute_structured_loss(
            logits=logits,
            labels=labels,
            loss_mask_point=loss_mask_point,
            loss_mask_object=loss_mask_object,
            loss_mask_format=loss_mask_format,
            weight_point=float(weight_point),
            weight_object=float(weight_object),
            weight_format=float(weight_format),
        )
        d["loss_answer"] = d["loss"]
        d["n_answer_tokens"] = d["n_point_tokens"] + d["n_object_tokens"] + d["n_format_tokens"]
        return d

    # Legacy path: single answer mask CE
    if torch.is_tensor(logits) and logits.dim() == 3:
        l_answer, n_answer = masked_token_ce(logits, labels, loss_mask_answer)
    else:
        l_answer, n_answer = z, 0

    w = float(weight_answer)
    total = w * l_answer if n_answer > 0 else z

    return {
        "loss": total,
        "loss_answer": l_answer,
        "loss_point": z,
        "loss_object": z,
        "loss_format": z,
        "n_answer_tokens": int(n_answer),
        "n_point_tokens": 0,
        "n_object_tokens": 0,
        "n_format_tokens": 0,
        "format_valid_rate": z,
        "n_format_samples": 0,
    }
