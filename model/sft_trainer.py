from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils.common import to_device
from .utils.loss_utils import compute_answer_loss


@dataclass
class SftEpochResult:
    global_step: int
    train_loss: float
    skipped_all_ignore: int
    step_count: int
    t_data: float
    t_fwd: float
    t_bwd: float
    t_step: float


def run_sft_epoch(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    trainable_params: list[torch.nn.Parameter],
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_weights: dict[str, float],
    loc_token_ids_for_loss: torch.Tensor | None,
    gaussian_point_sigma: float,
    accum_steps: int,
    max_grad_norm: float,
    wandb_run: Any,
    epoch: int,
    num_epochs: int,
    num_train_batches: int,
    updates_per_epoch: int,
    global_step: int,
    train_log_every: int,
) -> SftEpochResult:
    """Run one SFT epoch. Pure CE loss, no distillation."""
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
                    log_payload = {
                        "train/loss": float(raw_loss.detach().item()),
                        "train/loss_point": float(losses["loss_point"].detach().item()),
                        "train/loss_object": float(losses["loss_object"].detach().item()),
                        "train/loss_format": float(losses["loss_format"].detach().item()),
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "train/grad_norm": grad_norm_value,
                        "train/epoch": epoch_progress,
                    }
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

    return SftEpochResult(
        global_step=global_step,
        train_loss=float(sum_loss / float(sample_count)),
        skipped_all_ignore=skipped_all_ignore,
        step_count=step_count,
        t_data=_t_data_sum,
        t_fwd=_t_fwd_sum,
        t_bwd=_t_bwd_sum,
        t_step=_t_step_sum,
    )
