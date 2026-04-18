from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .eval_utils import l2_stats
from .gaze_tokens import parse_structured_output_text


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
    Both tensors are on the same device as logits.
    """
    # Causal LM: logits[:, t] predicts input_ids[:, t+1]
    shift_logits = logits[:, :-1, :]                       # [B, L-1, V]
    shift_ids = input_ids[:, 1:].clamp(min=0)              # [B, L-1]
    shift_mask = answer_mask[:, 1:].to(device=logits.device)  # [B, L-1]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_lp = log_probs.gather(2, shift_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    token_lp = token_lp * shift_mask.float()

    sum_lp = token_lp.sum(dim=1)               # [B]
    n_tokens = shift_mask.float().sum(dim=1)   # [B]
    return sum_lp, n_tokens


# ---------------------------------------------------------------------------
# GRPO objective
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
    """GRPO clipped policy-gradient loss + KL-from-ref penalty.

    All logprob arguments are per-sequence sums; normalised by n_tokens internally.

    L = -mean[ min(r*A, clip(r,1-eps,1+eps)*A) ] + kl_beta * mean[log(pi/pi_ref)]

    Returns (loss, stats_dict).
    """
    n = n_tokens.clamp(min=1.0).to(device=new_lp_sum.device, dtype=torch.float32)
    adv = advantages.to(device=new_lp_sum.device, dtype=torch.float32)
    new_lp = new_lp_sum.to(dtype=torch.float32)
    old_lp = old_lp_sum.to(device=new_lp_sum.device, dtype=torch.float32)
    ref_lp = ref_lp_sum.to(device=new_lp_sum.device, dtype=torch.float32)

    # Per-token average log-ratio (current vs old-at-rollout)
    log_ratio = (new_lp - old_lp) / n
    ratio = log_ratio.exp().clamp(0.0, 10.0)  # safety clamp against exp overflow

    pg_unclipped = ratio * adv
    pg_clipped = ratio.clamp(1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
    pg_loss = -torch.min(pg_unclipped, pg_clipped).mean()

    # KL: E[log(pi_theta / pi_ref)] evaluated on rollout samples
    kl_per_sample = (new_lp - ref_lp) / n
    kl_loss = kl_per_sample.mean()

    total_loss = pg_loss + float(kl_beta) * kl_loss

    return total_loss, {
        "pg_loss": float(pg_loss.detach().item()),
        "kl_mean": float(kl_loss.detach().item()),
        "ratio_mean": float(ratio.detach().mean().item()),
        "ratio_std": float(ratio.detach().std().item()) if ratio.numel() > 1 else 0.0,
    }
