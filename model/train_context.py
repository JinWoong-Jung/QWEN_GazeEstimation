from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainRuntime:
    device: torch.device
    amp_dtype: torch.dtype
    processor: Any
    coord_bins: int
    num_classes: int
    scene_size: tuple[int, int] | None


@dataclass
class LossConfig:
    weight_point: float
    weight_object: float
    weight_format: float
    gaussian_point_sigma: float
    loc_token_ids: torch.Tensor | None
    object_token_ids: torch.Tensor | None


@dataclass
class EvalConfig:
    checkpoint_monitor: str
    checkpoint_monitor_mode: str
    eval_target_order: str
    gen_max_tokens: int
    num_beams: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    stop_at_object_end: bool
    constrained_decoding: bool
    constrained_temperature: float
    constrained_loc_decoding: str


@dataclass
class SdftConfig:
    mode: str
    ce_weight: float
    distil_kl_weight: float
    distil_temperature: float
    distil_teacher_eval_mode: bool
    distil_teacher_suffix: str
    rollout_max_new_tokens: int
    rollout_do_sample: bool
    rollout_temperature: float
    rollout_top_p: float
    rollout_constrained_decoding: bool
    rollout_constrained_loc_decoding: str
    skip_invalid_rollouts: bool
    skip_truncated_rollouts: bool
    kl_on_point: bool
    kl_on_object: bool
    kl_on_format: bool
