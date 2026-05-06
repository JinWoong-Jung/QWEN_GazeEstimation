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


def gaussian_soft_label_ce(
    logits_at_pt: torch.Tensor,
    gt_loc_ids: torch.Tensor,
    loc_token_ids: torch.Tensor,
    sigma: float = 7.0,
) -> torch.Tensor:
    """Gaussian soft-label CE over coordinate bins.

    Each loc token is treated as a bin; bins near the GT bin receive
    partial credit proportional to a Gaussian centered at the GT bin.
    log_softmax is computed over the full vocabulary; only the C loc
    token positions contribute to the loss.

    Args:
        logits_at_pt:  [N, V] logits at the N point-token positions.
        gt_loc_ids:    [N]    GT loc token vocab IDs for each position.
        loc_token_ids: [C]    All loc vocab IDs in bin order (C = coord_bins).
        sigma:         Gaussian width in bin units.
    Returns:
        Scalar loss.
    """
    C = int(loc_token_ids.shape[0])
    N = int(logits_at_pt.shape[0])
    if N <= 0 or C <= 0:
        return torch.zeros((), device=logits_at_pt.device, dtype=logits_at_pt.dtype)

    loc_ids_dev = loc_token_ids.to(device=logits_at_pt.device)
    gt_ids_dev = gt_loc_ids.to(device=logits_at_pt.device)

    # GT bin index (0..C-1) for each of the N positions
    gt_bin = (gt_ids_dev.unsqueeze(1) == loc_ids_dev.unsqueeze(0)).long().argmax(dim=1)  # [N]

    # Gaussian soft labels [N, C]
    k = torch.arange(C, device=logits_at_pt.device, dtype=logits_at_pt.dtype)
    diff = k.unsqueeze(0) - gt_bin.to(dtype=logits_at_pt.dtype).unsqueeze(1)  # [N, C]
    soft = torch.exp(-0.5 * diff ** 2 / (float(sigma) ** 2))
    soft = soft / soft.sum(dim=1, keepdim=True)  # [N, C], sums to 1

    # log_softmax over full vocab, then slice to loc token positions only
    log_p = F.log_softmax(logits_at_pt, dim=-1)[:, loc_ids_dev]  # [N, C]
    return -(soft * log_p).sum(dim=-1).mean()


def compute_structured_loss(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_mask_point: torch.Tensor | None,
    loss_mask_object: torch.Tensor | None,
    loss_mask_format: torch.Tensor | None,
    loss_mask_reasoning: torch.Tensor | None = None,
    weight_point: float = 1.0,
    weight_object: float = 1.0,
    weight_format: float = 0.25,
    weight_reasoning: float = 0.3,
    compute_format_rate: bool = False,
    loc_token_ids: torch.Tensor | None = None,
    gaussian_sigma: float = 0.0,
) -> dict[str, Any]:
    """Compute structured SFT loss.

    CE is computed once on the union of valid mask positions, then split by
    category. compute_format_rate gates the expensive argmax-based format
    exact-match rate — pass True only at wandb logging steps.

    Returns dict with keys:
        loss, loss_point, loss_object, loss_format, loss_reasoning,
        n_point_tokens, n_object_tokens, n_format_tokens, n_reasoning_tokens,
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
            "loss_reasoning": z,
            "n_point_tokens": 0,
            "n_object_tokens": 0,
            "n_format_tokens": 0,
            "n_reasoning_tokens": 0,
            "format_valid_rate": z,
            "n_format_samples": 0,
        }

    # Causal LM shift - done once for all categories.
    shift_logits = logits[:, :-1, :]   # [B, L-1, V]
    shift_labels = labels[:, 1:]        # [B, L-1]
    not_padding = shift_labels.ne(-100)

    def _shift_mask(m: torch.Tensor | None) -> torch.Tensor:
        if m is None or not torch.is_tensor(m) or m.dim() != 2:
            return torch.zeros(shift_labels.shape, dtype=torch.bool, device=device)
        return m.to(device=device, dtype=torch.bool)[:, 1:]

    sm_pt  = _shift_mask(loss_mask_point)
    sm_obj = _shift_mask(loss_mask_object)
    sm_fmt = _shift_mask(loss_mask_format)
    sm_rsn = _shift_mask(loss_mask_reasoning)

    pt_valid  = sm_pt  & not_padding
    obj_valid = sm_obj & not_padding
    fmt_valid = sm_fmt & not_padding
    rsn_valid = sm_rsn & not_padding

    n_pt  = int(pt_valid.sum().item())
    n_obj = int(obj_valid.sum().item())
    n_fmt = int(fmt_valid.sum().item())
    n_rsn = int(rsn_valid.sum().item())

    # Build union mask and compute CE exactly once on valid token positions.
    valid_union = (sm_pt | sm_obj | sm_fmt | sm_rsn) & not_padding   # [B, L-1]

    if valid_union.any():
        ce_valid = F.cross_entropy(
            shift_logits[valid_union],   # [N_valid, V]
            shift_labels[valid_union],   # [N_valid]
            reduction="none",
        )  # [N_valid]

        def _cat_mean(cat_valid: torch.Tensor, n_cat: int) -> torch.Tensor:
            if n_cat <= 0:
                return z
            return ce_valid[cat_valid[valid_union]].mean()

        # Point loss: Gaussian soft-label CE when loc_token_ids and sigma are provided,
        # otherwise fall back to standard hard CE.
        pt_logits = None
        pt_labels = None
        if n_pt > 0:
            pt_in_union = pt_valid[valid_union]           # [N_valid] bool
            pt_logits = shift_logits[valid_union][pt_in_union]
            pt_labels = shift_labels[valid_union][pt_in_union]

        if (
            loc_token_ids is not None
            and float(gaussian_sigma) > 0.0
            and n_pt > 0
        ):
            l_pt = gaussian_soft_label_ce(
                pt_logits,                                # [N_pt, V]
                pt_labels,                                # [N_pt]
                loc_token_ids.to(device=device),
                sigma=float(gaussian_sigma),
            )
        else:
            l_pt = _cat_mean(pt_valid, n_pt)

        l_obj = _cat_mean(obj_valid, n_obj)
        l_fmt = _cat_mean(fmt_valid, n_fmt)
        l_rsn = _cat_mean(rsn_valid, n_rsn)
    else:
        l_pt = l_obj = l_fmt = l_rsn = z

    fmt_valid_rate, n_fmt_samples = (
        masked_sample_exact_rate(logits, labels, loss_mask_format)
        if compute_format_rate
        else (z, 0)
    )

    wp, wo, wf, wr = float(weight_point), float(weight_object), float(weight_format), float(weight_reasoning)
    total = z
    if n_pt > 0:
        total = total + wp * l_pt
    if n_obj > 0:
        total = total + wo * l_obj
    if n_fmt > 0:
        total = total + wf * l_fmt
    if n_rsn > 0:
        total = total + wr * l_rsn

    return {
        "loss": total,
        "loss_point": l_pt,
        "loss_object": l_obj,
        "loss_format": l_fmt,
        "loss_reasoning": l_rsn,
        "n_point_tokens": int(n_pt),
        "n_object_tokens": int(n_obj),
        "n_format_tokens": int(n_fmt),
        "n_reasoning_tokens": int(n_rsn),
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
    loss_mask_reasoning: torch.Tensor | None = None,
    weight_answer: float = 1.0,
    weight_point: float = 1.0,
    weight_object: float = 1.0,
    weight_format: float = 0.25,
    weight_reasoning: float = 0.3,
    compute_format_rate: bool = False,
    loc_token_ids: torch.Tensor | None = None,
    gaussian_sigma: float = 0.0,
) -> dict[str, Any]:
    """Compute structured SFT loss when structured masks are present,
    else fall back to full-answer CE (legacy).

    Returns dict with keys: loss, loss_point, loss_object, loss_format,
    loss_reasoning, loss_answer, n_point_tokens, n_object_tokens,
    n_format_tokens, n_reasoning_tokens, n_answer_tokens.
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
            loss_mask_reasoning=loss_mask_reasoning,
            weight_point=float(weight_point),
            weight_object=float(weight_object),
            weight_format=float(weight_format),
            weight_reasoning=float(weight_reasoning),
            compute_format_rate=compute_format_rate,
            loc_token_ids=loc_token_ids,
            gaussian_sigma=float(gaussian_sigma),
        )
        d["loss_answer"] = d["loss"]
        d["n_answer_tokens"] = (
            d["n_point_tokens"] + d["n_object_tokens"]
            + d["n_format_tokens"] + d["n_reasoning_tokens"]
        )
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
        "loss_reasoning": z,
        "n_answer_tokens": int(n_answer),
        "n_point_tokens": 0,
        "n_object_tokens": 0,
        "n_format_tokens": 0,
        "n_reasoning_tokens": 0,
        "format_valid_rate": z,
        "n_format_samples": 0,
    }
