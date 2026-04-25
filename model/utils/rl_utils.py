from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .eval_utils import l2_stats


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

def compute_point_reward(min_l2: float, beta: float) -> float:
    """r_point = exp(-beta * min_l2). Approaches 1 as min_l2→0."""
    return math.exp(-float(beta) * max(0.0, float(min_l2)))


def compute_object_reward(pred_obj_id: int | None, gt_obj_ids: list[int]) -> float:
    """r_object = 1.0 if pred matches any GT object id, else 0.0."""
    if pred_obj_id is None or not gt_obj_ids:
        return 0.0
    return 1.0 if int(pred_obj_id) in [int(x) for x in gt_obj_ids] else 0.0


def compute_total_reward(
    parsed: dict[str, Any],
    gt_points: torch.Tensor | None,
    gt_obj_ids: list[int],
    *,
    reward_point_weight: float = 1.0,
    reward_object_weight: float = 0.75,
    reward_joint_bonus: float = 0.25,
    reward_extra_penalty: float = 0.5,
    reward_point_beta: float = 10.0,
) -> dict[str, float]:
    """Compute per-rollout reward decomposed into point/object/joint/extra components.

    Format is a hard gate: invalid format → r_total = -1.0, all others = 0.
    This prevents RL from undoing the SFT-learned format structure.

    Returns dict with: reward_total, reward_point, reward_object,
    reward_joint, reward_extra, valid_format, has_extra_text.
    """
    has_extra = bool(parsed.get("has_extra_text", False))
    base: dict[str, float] = {
        "reward_total": 0.0,
        "reward_point": 0.0,
        "reward_object": 0.0,
        "reward_joint": 0.0,
        "reward_extra": 0.0,
        "valid_format": False,
        "has_extra_text": has_extra,
    }

    if not bool(parsed.get("valid_format", False)):
        base["reward_total"] = -1.0
        return base

    base["valid_format"] = True

    # Point reward: exp(-beta * min_l2)
    r_point = 0.0
    min_l2_val: float | None = None
    if parsed.get("point_xy") is not None and gt_points is not None:
        stats = l2_stats(parsed["point_xy"], gt_points)
        if stats is not None:
            min_l2_val = float(stats[1])
            r_point = compute_point_reward(min_l2_val, float(reward_point_beta))
    base["reward_point"] = float(r_point)

    # Object reward: exact match (supports multi-label)
    r_obj = compute_object_reward(parsed.get("object_id"), gt_obj_ids)
    base["reward_object"] = float(r_obj)

    # Joint bonus: both correct AND point close (<0.1 normalised L2)
    r_joint = 0.0
    if r_obj > 0.5 and min_l2_val is not None and min_l2_val < 0.1:
        r_joint = 1.0
    base["reward_joint"] = float(r_joint)

    # Extra-text penalty
    r_extra = 1.0 if has_extra else 0.0
    base["reward_extra"] = float(r_extra)

    r_total = (
        float(reward_point_weight) * r_point
        + float(reward_object_weight) * r_obj
        + float(reward_joint_bonus) * r_joint
        - float(reward_extra_penalty) * r_extra
    )
    base["reward_total"] = float(r_total)
    return base


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def group_normalize_advantages(
    rewards: list[float],
    eps: float = 1e-8,
) -> list[float]:
    """GRPO group-relative advantage: A_i = (r_i - mean(group)) / (std(group) + eps)."""
    if not rewards:
        return []
    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(max(0.0, variance))
    return [(r - mean) / (std + eps) for r in rewards]


# ---------------------------------------------------------------------------
# Adaptive KL Controller (Rex-Omni / TRL 방식)
# ---------------------------------------------------------------------------

class AdaptiveKLController:
    """KL coefficient that adapts to keep KL near a target value.

    Based on: https://arxiv.org/pdf/1909.08593.pdf
    Adapted from Rex-Omni verl/trainer/core_algos.py
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float) -> None:
        self.kl_coef = float(init_kl_coef)
        self.target = float(target_kl)
        self.horizon = float(horizon)

    def update(self, current_kl: float, n_steps: int) -> None:
        proportional_error = float(np.clip(current_kl / self.target - 1, -0.2, 0.2))
        mult = 1.0 + proportional_error * float(n_steps) / self.horizon
        self.kl_coef *= mult


class FixedKLController:
    """Fixed KL coefficient (no adaptation)."""

    def __init__(self, init_kl_coef: float) -> None:
        self.kl_coef = float(init_kl_coef)

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def build_kl_controller(
    kl_type: str,
    init_kl_coef: float,
    target_kl: float = 0.1,
    horizon: float = 10000.0,
) -> AdaptiveKLController | FixedKLController:
    if kl_type == "adaptive":
        return AdaptiveKLController(
            init_kl_coef=init_kl_coef,
            target_kl=target_kl,
            horizon=horizon,
        )
    return FixedKLController(init_kl_coef=init_kl_coef)


# ---------------------------------------------------------------------------
# Log-prob utilities
# ---------------------------------------------------------------------------

def build_answer_span_mask(
    generated_ids: torch.Tensor,
    prompt_len: int,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Bool mask [B, L] that is True for generated answer token positions.

    Covers positions >= prompt_len, excluding padding.
    """
    bsz, seqlen = int(generated_ids.shape[0]), int(generated_ids.shape[1])
    mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=generated_ids.device)
    if int(prompt_len) < seqlen:
        mask[:, int(prompt_len):] = True
    if pad_token_id is not None:
        mask &= generated_ids.ne(int(pad_token_id))
    return mask


def compute_token_logprobs_sum(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of per-token log-probs at answer positions (causal-LM shift).

    Returns (sum_logprobs [B], n_tokens [B]).
    """
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:].clamp(min=0)
    shift_mask = answer_mask[:, 1:].to(device=logits.device)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_lp = log_probs.gather(2, shift_ids.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * shift_mask.float()

    sum_lp = token_lp.sum(dim=1)
    n_tokens = shift_mask.float().sum(dim=1)
    return sum_lp, n_tokens


def compute_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token log-probs for all positions (causal-LM shift).

    Returns token_logprobs [B, L-1] — position i gives log P(token i+1 | context).
    Masking (response_mask) is applied externally by the loss function.
    """
    shift_logits = logits[:, :-1, :]                   # [B, L-1, V]
    shift_ids = input_ids[:, 1:].clamp(min=0)          # [B, L-1]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    return log_probs.gather(2, shift_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]


# ---------------------------------------------------------------------------
# KL divergence (Rex-Omni / verl style — per-token)
# ---------------------------------------------------------------------------

def compute_kl_per_token(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_penalty: str = "low_var_kl",
) -> torch.Tensor:
    """Per-token KL divergence estimate.

    Args:
        log_probs:     [B, L] current policy per-token log-probs
        ref_log_probs: [B, L] reference policy per-token log-probs
        kl_penalty:    one of "kl" | "abs" | "mse" | "low_var_kl"

    Returns:
        kl: [B, L] per-token KL estimates

    Adapted from Rex-Omni verl/trainer/core_algos.py::compute_kl()
    """
    log_probs = log_probs.float()
    ref_log_probs = ref_log_probs.float()

    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    if kl_penalty == "low_var_kl":
        # Joschu approximation: low-variance, numerically stable
        # http://joschu.net/blog/kl-approx.html
        kl = ref_log_probs - log_probs
        kld = (kl.exp() - kl - 1.0).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    raise NotImplementedError(f"Unknown kl_penalty: {kl_penalty}")


# ---------------------------------------------------------------------------
# Policy loss (Rex-Omni verl style — per-token, dual-clip)
# ---------------------------------------------------------------------------

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked positions; returns 0 if mask is all-False."""
    mask = mask.to(dtype=torch.bool, device=x.device)
    n = mask.float().sum()
    if n < 1.0:
        return x.new_zeros(())
    return (x * mask.float()).sum() / n


def compute_policy_loss_per_token(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    clip_ratio_dual: float = 3.0,
    kl_beta: float = 0.01,
    kl_penalty: str = "low_var_kl",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-token GRPO policy loss with dual-clip and KL penalty.

    Implements Rex-Omni verl/trainer/core_algos.py::compute_policy_loss() adapted
    for single-GPU training with built-in KL regularisation.

    Args:
        old_log_probs:  [B, L] log-probs cached at rollout time
        new_log_probs:  [B, L] log-probs of current policy (with grad)
        ref_log_probs:  [B, L] log-probs of frozen reference model
        advantages:     [B] scalar advantage per sample (broadcast to token level)
        response_mask:  [B, L] True at response token positions
        clip_ratio_low: lower PPO clip bound (1 - eps)
        clip_ratio_high: upper PPO clip bound (1 + eps); can be > low for DAPO
        clip_ratio_dual: dual-clip threshold (prevents large negative loss when adv<0)
        kl_beta:        KL penalty coefficient
        kl_penalty:     KL approximation mode

    Returns:
        (total_loss, stats_dict)
    """
    # Broadcast scalar advantage → token level [B, L]
    adv_token = advantages.float().to(device=new_log_probs.device).unsqueeze(-1).expand_as(new_log_probs)

    # Importance sampling ratio per token
    neg_approx_kl = new_log_probs - old_log_probs          # [B, L]
    ratio = torch.exp(neg_approx_kl)                        # [B, L]
    clipped_ratio = torch.exp(
        torch.clamp(
            neg_approx_kl,
            math.log(1.0 - clip_ratio_low),
            math.log(1.0 + clip_ratio_high),
        )
    )

    # PPO surrogate losses
    pg_loss_unclipped = -adv_token * ratio                  # [B, L]
    pg_loss_clipped   = -adv_token * clipped_ratio          # [B, L]
    pg_loss_dual      = -adv_token * clip_ratio_dual        # [B, L] — dual-clip floor

    # Higher clip: max(unclipped, clipped) — penalises ratio > 1+eps when adv>0
    pg_loss_higher = torch.max(pg_loss_unclipped, pg_loss_clipped)
    # Dual clip: only active when adv < 0 — prevents over-suppression of unlikely actions
    pg_loss_final = torch.where(adv_token < 0, torch.min(pg_loss_higher, pg_loss_dual), pg_loss_higher)

    pg_loss = _masked_mean(pg_loss_final, response_mask)

    # Clip fraction statistics
    clip_frac_high = _masked_mean((pg_loss_unclipped < pg_loss_clipped).float(), response_mask)
    clip_frac_low  = _masked_mean(
        (pg_loss_higher > pg_loss_dual).float() * (adv_token < 0).float(),
        response_mask,
    )
    approx_kl_mean = _masked_mean(-neg_approx_kl, response_mask)

    # KL penalty from reference model
    kl_tokens = compute_kl_per_token(new_log_probs, ref_log_probs, kl_penalty=kl_penalty)
    kl_loss = _masked_mean(kl_tokens, response_mask)

    total_loss = pg_loss + float(kl_beta) * kl_loss

    return total_loss, {
        "pg_loss":         float(pg_loss.detach().item()),
        "kl_mean":         float(kl_loss.detach().item()),
        "approx_kl":       float(approx_kl_mean.detach().item()),
        "clip_frac_high":  float(clip_frac_high.detach().item()),
        "clip_frac_low":   float(clip_frac_low.detach().item()),
        "ratio_mean":      float(_masked_mean(ratio, response_mask).detach().item()),
    }


# ---------------------------------------------------------------------------
# Legacy GRPO objective (sequence-level) — kept for backward compatibility
# ---------------------------------------------------------------------------

def compute_grpo_loss(
    new_lp_sum: torch.Tensor,
    old_lp_sum: torch.Tensor,
    ref_lp_sum: torch.Tensor,
    n_tokens: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sequence-level GRPO loss (legacy). Prefer compute_policy_loss_per_token."""
    n = n_tokens.clamp(min=1.0).to(device=new_lp_sum.device, dtype=torch.float32)
    adv = advantages.to(device=new_lp_sum.device, dtype=torch.float32)
    new_lp = new_lp_sum.to(dtype=torch.float32)
    old_lp = old_lp_sum.to(device=new_lp_sum.device, dtype=torch.float32)
    ref_lp = ref_lp_sum.to(device=new_lp_sum.device, dtype=torch.float32)

    log_ratio = (new_lp - old_lp) / n
    ratio = log_ratio.exp().clamp(0.0, 10.0)

    pg_unclipped = ratio * adv
    pg_clipped = ratio.clamp(1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
    pg_loss = -torch.min(pg_unclipped, pg_clipped).mean()

    kl_per_sample = (new_lp - ref_lp) / n
    kl_loss = kl_per_sample.mean()

    total_loss = pg_loss + float(kl_beta) * kl_loss

    return total_loss, {
        "pg_loss":    float(pg_loss.detach().item()),
        "kl_mean":    float(kl_loss.detach().item()),
        "ratio_mean": float(ratio.detach().mean().item()),
        "ratio_std":  float(ratio.detach().std().item()) if ratio.numel() > 1 else 0.0,
    }
