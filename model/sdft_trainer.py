from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils.common import to_device


class EMATeacher:
    """EMA copy of trainable parameters only (LoRA adapters + gaze tokens).

    ϕ ← τ·ϕ + (1−τ)·θ  after each optimizer step.
    Base model weights are frozen in both student and teacher so they are not tracked.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.ema_params: dict[str, torch.Tensor] = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                self.ema_params[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module):
        """Context manager: swap in EMA weights, yield, restore student weights."""
        ema = self

        class _Ctx:
            def __enter__(self_inner):
                self._originals: dict[str, torch.Tensor] = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and name in ema.ema_params:
                        self._originals[name] = param.data
                        param.data = ema.ema_params[name]
                return self_inner

            def __exit__(self_inner, *_):
                for name, param in model.named_parameters():
                    if name in self._originals:
                        param.data = self._originals[name]

        return _Ctx()
from .utils.eval_utils import constrained_generate_structured, decode_generated
from .utils.gaze_tokens import parse_structured_output_text
from .utils.loss_utils import compute_answer_loss, compute_kl_distil_loss, compute_kl_distil_loss_legacy_union
from .utils.processor_collate import build_infer_inputs, build_train_inputs
from .utils.special_tokens import OBJECT_END_MARKER


def train_step_sdft_rollout(
    *,
    model: torch.nn.Module,
    processor: Any,
    batch: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    distil_kl_weight: float,
    distil_temperature: float,
    distil_teacher_eval_mode: bool,
    distil_teacher_suffix: str,
    ema_teacher: EMATeacher | None = None,
    loc_token_ids_for_loss: torch.Tensor | None,
    object_token_ids_for_loss: torch.Tensor | None,
    loss_weights: dict[str, float],
    gaussian_point_sigma: float,
    sdft_ce_weight: float,
    rollout_max_new_tokens: int,
    rollout_do_sample: bool,
    rollout_temperature: float,
    rollout_top_p: float,
    rollout_constrained_decoding: bool,
    rollout_constrained_loc_decoding: str,
    rollout_target_order: str,
    skip_invalid_rollouts: bool,
    skip_truncated_rollouts: bool,
    kl_on_point: bool,
    kl_on_object: bool,
    num_classes: int,
    coord_bins: int,
    max_text_length: int,
) -> dict[str, Any] | None:
    """Run one on-policy rollout SDFT step.

    The student generates from the direct point-object prompt. The same generated
    completion is then re-forwarded through the student prompt with gradients and
    the teacher prompt without gradients; KL is applied to the generated point and
    object token positions.
    """
    ce_loss: torch.Tensor | None = None
    if sdft_ce_weight > 0.0:
        gt_labels = batch["labels"].to(device)
        if not torch.all(gt_labels.eq(-100)):
            gt_joint = to_device(batch["joint_inputs"], device=device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                gt_out = model(joint_inputs=gt_joint, use_cache=False)
                ce_result = compute_answer_loss(
                    logits=gt_out.get("logits"),
                    labels=gt_labels,
                    loss_mask_point=batch.get("loss_mask_point"),
                    loss_mask_object=batch.get("loss_mask_object"),
                    loss_mask_format=batch.get("loss_mask_format"),
                    weight_point=loss_weights["point"],
                    weight_object=loss_weights["object"],
                    weight_format=loss_weights["format"],
                    loc_token_ids=loc_token_ids_for_loss,
                    gaussian_sigma=gaussian_point_sigma,
                )
            ce_loss = ce_result["loss"]

    scene_images: list[Any] = batch["scene_images"]
    text_inputs: list[str] = batch["text_input"]
    reasoning_texts: list[str] = batch.get("reasoning_texts", [""] * len(text_inputs))
    object_texts: list[str] = batch.get("object_texts", [""] * len(text_inputs))
    teacher_base_inputs: list[str] = batch.get("teacher_base_inputs", text_inputs)

    infer_inputs = build_infer_inputs(
        processor=processor,
        scene_images=scene_images,
        text_inputs=text_inputs,
        max_text_length=max_text_length,
    )
    infer_inputs_dev = to_device(infer_inputs, device=device)
    prompt_input_ids_cpu = infer_inputs["input_ids"].detach().cpu()
    prompt_attn_cpu = infer_inputs.get("attention_mask", None)
    if prompt_attn_cpu is not None:
        prompt_attn_cpu = prompt_attn_cpu.detach().cpu()

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            if bool(rollout_constrained_decoding):
                generated_texts = constrained_generate_structured(
                    model=model,
                    joint=infer_inputs_dev,
                    processor=processor,
                    num_classes=int(num_classes),
                    coord_bins=int(coord_bins),
                    amp_dtype=amp_dtype,
                    target_order=str(rollout_target_order),
                    temperature=float(rollout_temperature),
                    loc_selection_strategy=str(rollout_constrained_loc_decoding),
                )
            else:
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
                ):
                    gen_kwargs: dict[str, Any] = {}
                    if rollout_do_sample:
                        gen_kwargs["temperature"] = rollout_temperature
                        gen_kwargs["top_p"] = rollout_top_p
                    generated_ids = model.generate(
                        joint_inputs=infer_inputs_dev,
                        max_new_tokens=rollout_max_new_tokens,
                        do_sample=rollout_do_sample,
                        **gen_kwargs,
                    )
                generated_texts = decode_generated(
                    processor=processor,
                    generated_ids=generated_ids.detach().cpu(),
                    input_ids=prompt_input_ids_cpu,
                    attention_mask=prompt_attn_cpu,
                    num_return_sequences=1,
                )
    finally:
        if was_training:
            model.train()

    valid_indices: list[int] = []
    valid_generated_texts: list[str] = []
    bsz = len(generated_texts)

    for i, text in enumerate(generated_texts):
        if skip_truncated_rollouts and OBJECT_END_MARKER not in text:
            continue
        parsed = parse_structured_output_text(
            text, num_classes=num_classes, coord_bins=coord_bins
        )
        if skip_invalid_rollouts and not parsed.get("valid_format", False):
            continue
        valid_indices.append(i)
        valid_generated_texts.append(text)

    valid_frac = float(len(valid_indices)) / float(max(1, bsz))

    if not valid_indices:
        if ce_loss is not None:
            loss = sdft_ce_weight * ce_loss
            return {
                "loss": loss,
                "kl_loss": None,
                "ce_loss": ce_loss.detach(),
                "rollout_format_valid": valid_frac,
                "valid_kl_sample_frac": 0.0,
            }
        return None

    valid_scene_images = [scene_images[i] for i in valid_indices]
    valid_direct_prompts = [text_inputs[i] for i in valid_indices]
    valid_teacher_base_prompts = [teacher_base_inputs[i] for i in valid_indices]
    valid_reasoning_texts = [reasoning_texts[i] for i in valid_indices]
    valid_object_texts = [object_texts[i] for i in valid_indices]
    n_valid = len(valid_indices)
    all_ones = torch.ones(n_valid, dtype=torch.float32)

    student_inputs, _student_labels, student_mask_pt, student_mask_obj, _student_mask_fmt = build_train_inputs(
        processor=processor,
        scene_images=valid_scene_images,
        text_inputs=valid_direct_prompts,
        target_texts=valid_generated_texts,
        target_text_valid=all_ones,
        target_point_valid=all_ones,
        target_object_valid=all_ones,
        target_format_valid=all_ones,
        max_text_length=max_text_length,
    )

    _has_any_reasoning = any(rt or ot for rt, ot in zip(valid_reasoning_texts, valid_object_texts))
    if _has_any_reasoning:
        teacher_text_inputs = [
            (base + distil_teacher_suffix.format(reasoning_text=rt, object_text=ot))
            if (rt or ot) else base
            for base, rt, ot in zip(valid_teacher_base_prompts, valid_reasoning_texts, valid_object_texts)
        ]
        teacher_inputs, _teacher_labels, teacher_mask_pt, teacher_mask_obj, _teacher_mask_fmt = build_train_inputs(
            processor=processor,
            scene_images=valid_scene_images,
            text_inputs=teacher_text_inputs,
            target_texts=valid_generated_texts,
            target_text_valid=all_ones,
            target_point_valid=all_ones,
            target_object_valid=all_ones,
            target_format_valid=all_ones,
            max_text_length=max_text_length,
        )
        teacher_inputs_dev = to_device(teacher_inputs, device=device)
    else:
        teacher_mask_pt = student_mask_pt
        teacher_mask_obj = student_mask_obj

    student_inputs_dev = to_device(student_inputs, device=device)
    if not _has_any_reasoning:
        teacher_inputs_dev = student_inputs_dev

    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
    ):
        student_out = model(joint_inputs=student_inputs_dev, use_cache=False)

        was_training = bool(model.training)
        if distil_teacher_eval_mode:
            model.eval()
        try:
            with torch.no_grad(), (ema_teacher.apply(model) if ema_teacher is not None else nullcontext()):
                teacher_out = model(joint_inputs=teacher_inputs_dev, use_cache=False)
        finally:
            if distil_teacher_eval_mode and was_training:
                model.train()

        kl_loss = compute_kl_distil_loss(
            student_logits=student_out.get("logits"),
            teacher_logits=teacher_out.get("logits"),
            student_mask_point=student_mask_pt if kl_on_point else None,
            teacher_mask_point=teacher_mask_pt if kl_on_point else None,
            student_mask_object=student_mask_obj if kl_on_object else None,
            teacher_mask_object=teacher_mask_obj if kl_on_object else None,
            has_distil=None,
            loc_token_ids=loc_token_ids_for_loss,
            object_token_ids=object_token_ids_for_loss,
            temperature=distil_temperature,
        )

        if not kl_loss.requires_grad:
            if ce_loss is not None:
                loss = sdft_ce_weight * ce_loss
                return {
                    "loss": loss,
                    "kl_loss": None,
                    "ce_loss": ce_loss.detach(),
                    "rollout_format_valid": valid_frac,
                    "valid_kl_sample_frac": 0.0,
                }
            return None

        loss = distil_kl_weight * kl_loss
        if ce_loss is not None:
            loss = loss + sdft_ce_weight * ce_loss

    return {
        "loss": loss,
        "kl_loss": kl_loss.detach(),
        "ce_loss": ce_loss.detach() if ce_loss is not None else None,
        "rollout_format_valid": valid_frac,
        "valid_kl_sample_frac": valid_frac,
    }


@dataclass
class SdftEpochResult:
    global_step: int
    train_loss: float
    skipped_all_ignore: int
    step_count: int
    t_data: float
    t_fwd: float
    t_bwd: float
    t_step: float


def run_sdft_epoch(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    trainable_params: list[torch.nn.Parameter],
    device: torch.device,
    amp_dtype: torch.dtype,
    sdft_mode: str,
    loss_weights: dict[str, float],
    loc_token_ids_for_loss: torch.Tensor | None,
    object_token_ids_for_loss: torch.Tensor | None,
    gaussian_point_sigma: float,
    distil_kl_weight: float,
    distil_temperature: float,
    distil_teacher_eval_mode: bool,
    distil_teacher_suffix: str,
    ema_teacher: EMATeacher | None = None,
    sdft_ce_weight: float,
    rollout_max_new_tokens: int,
    rollout_do_sample: bool,
    rollout_temperature: float,
    rollout_top_p: float,
    rollout_constrained_decoding: bool,
    rollout_constrained_loc_decoding: str,
    rollout_target_order: str,
    skip_invalid_rollouts: bool,
    skip_truncated_rollouts: bool,
    kl_on_point: bool,
    kl_on_object: bool,
    num_classes: int,
    coord_bins: int,
    max_text_length: int,
    accum_steps: int,
    max_grad_norm: float,
    wandb_run: Any,
    epoch: int,
    num_epochs: int,
    num_train_batches: int,
    updates_per_epoch: int,
    global_step: int,
    train_log_every: int,
) -> SdftEpochResult:
    """Run one SDFT epoch (rollout or teacher-forcing mode)."""
    model.train()
    sum_loss = 0.0
    sample_count = 0
    step_count = 0
    updates_done_in_epoch = 0
    skipped_all_ignore = 0
    _t_fwd_sum = 0.0
    _t_bwd_sum = 0.0
    _t_data_sum = 0.0
    _t_step_sum = 0.0

    optimizer.zero_grad(set_to_none=True)
    _accum_backward_count = 0
    train_iter = tqdm(
        train_loader,
        desc=f"Train {epoch}/{num_epochs}",
        leave=False,
        dynamic_ncols=True,
        disable=False,
    )

    _t_data0 = time.perf_counter()
    for step, batch in enumerate(train_iter, start=1):
        _t_step0 = time.perf_counter()
        _t_data_sum += _t_step0 - _t_data0
        bsz = int(batch["labels"].shape[0])
        is_last_batch = (step == num_train_batches)

        # ----------------------------------------------------------------
        # Rollout SDFT branch
        # ----------------------------------------------------------------
        if sdft_mode == "rollout":
            _t_fwd0 = time.perf_counter()
            rollout_result = train_step_sdft_rollout(
                model=model,
                processor=processor,
                batch=batch,
                device=device,
                amp_dtype=amp_dtype,
                distil_kl_weight=distil_kl_weight,
                distil_temperature=distil_temperature,
                distil_teacher_eval_mode=distil_teacher_eval_mode,
                distil_teacher_suffix=distil_teacher_suffix,
                ema_teacher=ema_teacher,
                loc_token_ids_for_loss=loc_token_ids_for_loss,
                object_token_ids_for_loss=object_token_ids_for_loss,
                loss_weights=loss_weights,
                gaussian_point_sigma=gaussian_point_sigma,
                sdft_ce_weight=sdft_ce_weight,
                rollout_max_new_tokens=rollout_max_new_tokens,
                rollout_do_sample=rollout_do_sample,
                rollout_temperature=rollout_temperature,
                rollout_top_p=rollout_top_p,
                rollout_constrained_decoding=rollout_constrained_decoding,
                rollout_constrained_loc_decoding=rollout_constrained_loc_decoding,
                rollout_target_order=rollout_target_order,
                skip_invalid_rollouts=skip_invalid_rollouts,
                skip_truncated_rollouts=skip_truncated_rollouts,
                kl_on_point=kl_on_point,
                kl_on_object=kl_on_object,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                max_text_length=max_text_length,
            )
            _t_fwd_sum += time.perf_counter() - _t_fwd0

            if rollout_result is None:
                # All rollouts invalid, no CE fallback: skip this batch without backward.
                skipped_all_ignore += 1
                _t_data0 = time.perf_counter()
                continue

            raw_loss = rollout_result["loss"]
            kl_loss_val = rollout_result.get("kl_loss") or torch.zeros((), device=device)
            losses: dict[str, Any] = {}  # structured SFT losses not computed in rollout mode

            scaled_loss = raw_loss / float(max(accum_steps, 1))
            _t_bwd0 = time.perf_counter()
            scaled_loss.backward()
            _accum_backward_count += 1
            _t_bwd_sum += time.perf_counter() - _t_bwd0

            should_step = (
                _accum_backward_count >= int(accum_steps)
                or (is_last_batch and _accum_backward_count > 0)
            )
            if should_step:
                if 0 < _accum_backward_count < int(accum_steps):
                    _grad_scale = float(accum_steps) / float(_accum_backward_count)
                    for _p in trainable_params:
                        if _p.grad is not None:
                            _p.grad.mul_(_grad_scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()
                if ema_teacher is not None:
                    ema_teacher.update(model)
                optimizer.zero_grad(set_to_none=True)
                _accum_backward_count = 0
                global_step += 1
                updates_done_in_epoch += 1

                if wandb_run is not None:
                    should_log_train = (global_step % train_log_every) == 0 or is_last_batch
                    if should_log_train:
                        grad_norm_value = (
                            float(grad_norm.detach().item())
                            if torch.is_tensor(grad_norm) else float(grad_norm)
                        )
                        epoch_progress = (float(epoch) - 1.0) + (
                            float(updates_done_in_epoch) / max(float(updates_per_epoch), 1.0)
                        )
                        _kl_v = float(kl_loss_val.detach().item()) if torch.is_tensor(kl_loss_val) else 0.0
                        _ce_v = rollout_result.get("ce_loss")
                        log_payload = {
                            "train/loss": float(raw_loss.detach().item()),
                            "train/sdft_mode": 1.0,  # 1=rollout, 0=teacher_forcing
                            "train/sdft_kl_loss": _kl_v,
                            "train/rollout_format_valid": float(rollout_result.get("rollout_format_valid", 0.0)),
                            "train/valid_kl_sample_frac": float(rollout_result.get("valid_kl_sample_frac", 0.0)),
                            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "train/grad_norm": grad_norm_value,
                            "train/epoch": epoch_progress,
                        }
                        if _ce_v is not None and torch.is_tensor(_ce_v):
                            log_payload["train/sdft_ce_loss"] = float(_ce_v.detach().item())
                        wandb_run.log(log_payload, step=global_step)

        # ----------------------------------------------------------------
        # Teacher-forcing SDFT branch
        # ----------------------------------------------------------------
        else:
            labels = batch["labels"].to(device)
            if torch.all(labels.eq(-100)):
                skipped_all_ignore += 1
                _t_data0 = time.perf_counter()
                continue

            joint_inputs = to_device(batch["joint_inputs"], device=device)

            _t_fwd0 = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(joint_inputs=joint_inputs, use_cache=False)
                losses = compute_answer_loss(
                    logits=out.get("logits", None),
                    labels=labels,
                    loss_mask_point=batch.get("loss_mask_point", None),
                    loss_mask_object=batch.get("loss_mask_object", None),
                    loss_mask_format=batch.get("loss_mask_format", None),
                    weight_point=loss_weights["point"],
                    weight_object=loss_weights["object"],
                    weight_format=loss_weights["format"],
                    compute_format_rate=False,
                    loc_token_ids=loc_token_ids_for_loss,
                    gaussian_sigma=gaussian_point_sigma,
                )
                raw_loss = losses["loss"]

                # Self-distillation KL loss: KL(teacher || student).
                kl_loss_val = torch.zeros((), device=device, dtype=raw_loss.dtype)
                if distil_kl_weight > 0.0 and "teacher_joint_inputs" in batch:
                    teacher_joint_inputs = to_device(batch["teacher_joint_inputs"], device=device)
                    was_training = bool(model.training)
                    if distil_teacher_eval_mode:
                        model.eval()
                    try:
                        with torch.no_grad(), (ema_teacher.apply(model) if ema_teacher is not None else nullcontext()):
                            teacher_out = model(joint_inputs=teacher_joint_inputs, use_cache=False)
                    finally:
                        if distil_teacher_eval_mode and was_training:
                            model.train()
                    kl_loss_val = compute_kl_distil_loss_legacy_union(
                        student_logits=out.get("logits"),
                        teacher_logits=teacher_out.get("logits"),
                        student_mask_point=batch.get("loss_mask_point"),
                        teacher_mask_point=batch.get("teacher_loss_mask_point"),
                        student_mask_object=batch.get("loss_mask_object"),
                        teacher_mask_object=batch.get("teacher_loss_mask_object"),
                        has_distil=batch.get("has_distil"),
                    )
                    raw_loss = raw_loss + distil_kl_weight * kl_loss_val

                scaled_loss = raw_loss / float(max(accum_steps, 1))
            _t_fwd_sum += time.perf_counter() - _t_fwd0

            _t_bwd0 = time.perf_counter()
            scaled_loss.backward()
            _accum_backward_count += 1
            _t_bwd_sum += time.perf_counter() - _t_bwd0

            should_step = (
                _accum_backward_count >= int(accum_steps)
                or (is_last_batch and _accum_backward_count > 0)
            )
            if should_step:
                if 0 < _accum_backward_count < int(accum_steps):
                    _grad_scale = float(accum_steps) / float(_accum_backward_count)
                    for _p in trainable_params:
                        if _p.grad is not None:
                            _p.grad.mul_(_grad_scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()
                if ema_teacher is not None:
                    ema_teacher.update(model)
                optimizer.zero_grad(set_to_none=True)
                _accum_backward_count = 0
                global_step += 1
                updates_done_in_epoch += 1

                if wandb_run is not None:
                    should_log_train = (global_step % train_log_every) == 0 or is_last_batch
                    if should_log_train:
                        grad_norm_value = (
                            float(grad_norm.detach().item())
                            if torch.is_tensor(grad_norm)
                            else float(grad_norm)
                        )
                        epoch_progress = (float(epoch) - 1.0) + (
                            float(updates_done_in_epoch) / max(float(updates_per_epoch), 1.0)
                        )
                        log_payload = {
                            "train/loss": float(raw_loss.detach().item()),
                            "train/loss_point": float(losses["loss_point"].detach().item()),
                            "train/loss_object": float(losses["loss_object"].detach().item()),
                            "train/loss_format": float(losses["loss_format"].detach().item()),
                            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "train/grad_norm": grad_norm_value,
                            "train/epoch": epoch_progress,
                        }
                        if distil_kl_weight > 0.0:
                            log_payload["train/loss_kl"] = float(kl_loss_val.detach().item())
                        wandb_run.log(log_payload, step=global_step)

        sum_loss += float(raw_loss.detach().item()) * float(bsz)
        sample_count += bsz
        step_count += 1
        _t_step_sum += time.perf_counter() - _t_step0
        _t_data0 = time.perf_counter()
        train_iter.set_postfix(loss=f"{(sum_loss / max(sample_count, 1)):.4f}")

    if step_count == 0 or sample_count == 0:
        raise RuntimeError(
            "No effective training batches were produced. "
            "All batches may have empty target text (all labels ignored)."
        )

    return SdftEpochResult(
        global_step=global_step,
        train_loss=float(sum_loss / float(sample_count)),
        skipped_all_ignore=skipped_all_ignore,
        step_count=step_count,
        t_data=_t_data_sum,
        t_fwd=_t_fwd_sum,
        t_bwd=_t_bwd_sum,
        t_step=_t_step_sum,
    )
