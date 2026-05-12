from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from peft import PeftModel

from .model import QwenTextGenerationModel
from .utils.checkpoint import (
    checkpoint_monitor_value,
    infer_checkpoint_monitor_mode,
    load_added_token_rows,
    load_token_rows,
    save_checkpoint,
)
from .utils.common import to_device
from .utils.model_init import init_base_model
from .utils.eval_utils import (
    decode_generated,
    maybe_save_generation_preview,
    run_test_metrics,
)
from .utils.gaze_tokens import parse_structured_output_text
from .utils.processor_collate import (
    QwenRLCollator,
    build_answer_mask,
    build_train_inputs,
)
from .utils.rl_utils import (
    build_kl_controller,
    compute_policy_loss_per_token,
    compute_token_logprobs,
    compute_total_reward,
    group_normalize_advantages,
    infer_logprobs_chunked,
)
from .utils.wandb_utils import val_metric_log_payload


def run_rl_training(
    *,
    args: argparse.Namespace,
    policy_model: torch.nn.Module,
    processor: Any,
    train_ds: Any,
    val_metric_loader: DataLoader | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_classes: int,
    out_dir: Path,
    base_vocab_size: int,
    new_vocab_size: int,
    model_path: str,
    model_kwargs: dict[str, Any],
    checkpoint_dir: Path | None,
    wandb_run: Any,
    scene_size: tuple[int, int] | None,
    coord_bins: int = 1000,
    token_ids_to_save: list[int] | None = None,
) -> tuple[int, float]:
    """GRPO-based RL post-training (Stage 2).

    Assumes policy_model is already loaded from an SFT checkpoint with trainable
    LoRA + new token embeddings.  A frozen copy of the same checkpoint is loaded
    as the reference model for the KL penalty term.

    Returns (global_step, best_monitor_value).
    """
    # ------------------------------------------------------------------
    # RL hyperparameters (Rex-Omni / verl style)
    # ------------------------------------------------------------------
    rl_group_size = max(1, int(getattr(args, "rl_group_size", 4)))
    rl_lr = float(getattr(args, "rl_lr", getattr(args, "lr", 1e-5)))
    rl_weight_decay = float(getattr(args, "weight_decay", 0.01))
    rl_max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
    reward_point_weight = float(getattr(args, "reward_point_weight", 1.0))
    reward_object_weight = float(getattr(args, "reward_object_weight", 0.75))
    reward_joint_bonus = float(getattr(args, "reward_joint_bonus", 0.25))
    reward_extra_penalty = float(getattr(args, "reward_extra_penalty", 0.5))
    reward_point_beta = float(getattr(args, "reward_point_beta", 10.0))
    reward_point_mode = str(getattr(args, "reward_point_mode", "l2"))
    reward_point_box_radius = float(getattr(args, "reward_point_box_radius", 0.05))
    rl_temperature = float(getattr(args, "rl_temperature", 0.7))
    rl_top_p = float(getattr(args, "rl_top_p", 0.9))
    rl_epochs = max(1, int(getattr(args, "epochs", 5)))
    rl_max_new_tokens = max(16, int(getattr(args, "generation_max_new_tokens", 16)))
    _eval_target_order = "point_object"

    # PPO clipping — asymmetric + dual-clip (Rex-Omni)
    rl_clip_ratio_low  = float(getattr(args, "rl_clip_ratio_low",  getattr(args, "rl_clip_eps", 0.2)))
    rl_clip_ratio_high = float(getattr(args, "rl_clip_ratio_high", getattr(args, "rl_clip_eps", 0.2)))
    rl_clip_ratio_dual = float(getattr(args, "rl_clip_ratio_dual", 3.0))

    # KL controller — adaptive or fixed (Rex-Omni AdaptiveKLController)
    kl_ctrl = build_kl_controller(
        kl_type   = str(getattr(args, "rl_kl_type",    "fixed")),
        init_kl_coef = float(getattr(args, "rl_kl_beta",   0.01)),
        target_kl = float(getattr(args, "rl_kl_target", 0.1)),
        horizon   = float(getattr(args, "rl_kl_horizon", 10000.0)),
    )
    rl_kl_penalty = str(getattr(args, "rl_kl_penalty", "low_var_kl"))

    # Multi-epoch rollout reuse
    n_ppo_epochs = max(1, int(getattr(args, "rl_n_ppo_epochs", 1)))

    # Logprob micro-batch size: limits peak VRAM during ref/old forward passes.
    # -1 = full B*G batch at once. Set to rl_group_size (e.g. 4) to process one
    # prompt group at a time — safest option for large batch_size.
    lp_micro_bsz = int(getattr(args, "rl_logprob_micro_batch_size", -1))
    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 4)))
    run_val_metrics_every_n = max(1, int(getattr(args, "run_val_metrics_every_n_epochs", 1)))
    checkpoint_monitor = str(getattr(args, "checkpoint_monitor", "val_dist")).strip() or "val_dist"
    checkpoint_monitor_mode = infer_checkpoint_monitor_mode(
        checkpoint_monitor,
        str(getattr(args, "checkpoint_monitor_mode", "auto")),
    )
    best_monitor_value = float("inf") if checkpoint_monitor_mode == "min" else -float("inf")

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id: int | None = getattr(tokenizer, "pad_token_id", None)

    # ------------------------------------------------------------------
    # Ref model: frozen copy of the SFT checkpoint — kept on GPU for speed.
    # ------------------------------------------------------------------
    print("[RL] loading frozen reference model …")
    _ref_base = init_base_model(model_path=model_path, model_kwargs=model_kwargs)
    _ref_base.resize_token_embeddings(int(new_vocab_size))
    if checkpoint_dir is not None:
        _adapter_dir = checkpoint_dir / "lora_adapter"
        if _adapter_dir.exists():
            _ref_qwen: Any = PeftModel.from_pretrained(
                _ref_base, model_id=str(_adapter_dir), is_trainable=False,
            )
            _ref_tmp = QwenTextGenerationModel(qwen_model=_ref_qwen)
            if not load_token_rows(ckpt_dir=checkpoint_dir, model=_ref_tmp, device=torch.device("cpu")):
                load_added_token_rows(ckpt_dir=checkpoint_dir, model=_ref_tmp, device=torch.device("cpu"))
        else:
            _ref_qwen = _ref_base
            print("[RL][WARN] no lora_adapter found in checkpoint_dir; using base model as ref.")
    else:
        _ref_qwen = _ref_base
        print("[RL][WARN] checkpoint_dir not set; KL ref is the base (non-SFT) model.")
    ref_model: torch.nn.Module = QwenTextGenerationModel(qwen_model=_ref_qwen).to(device)
    ref_model.eval()
    for _p in ref_model.parameters():
        _p.requires_grad_(False)
    print("[RL] reference model loaded and frozen on GPU.")

    # ------------------------------------------------------------------
    # RL dataloader (inference-mode prompts + GT labels + raw images)
    # ------------------------------------------------------------------
    rl_collator = QwenRLCollator(
        processor=processor,
        max_text_length=int(getattr(args, "max_text_length", 256)),
        scene_size=scene_size,
    )
    _nw = int(getattr(args, "num_workers", 4))
    rl_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(getattr(args, "batch_size", 4))),
        shuffle=True,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=rl_collator,
        persistent_workers=(_nw > 0),
        prefetch_factor=(2 if _nw > 0 else None),
    )

    # ------------------------------------------------------------------
    # Optimizer (lower lr than SFT, as per TODO recommendation)
    # ------------------------------------------------------------------
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=rl_lr, weight_decay=rl_weight_decay,
    )
    num_rl_batches = len(rl_loader)
    updates_per_epoch = max(1, math.ceil(num_rl_batches / accum_steps))
    total_updates = max(1, updates_per_epoch * rl_epochs)
    warmup_steps = int(total_updates * float(getattr(args, "warmup_ratio", 0.05)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    print(
        f"[RL] starting GRPO training: epochs={rl_epochs} group_size={rl_group_size} "
        f"clip_low={rl_clip_ratio_low} clip_high={rl_clip_ratio_high} clip_dual={rl_clip_ratio_dual} "
        f"kl_beta={kl_ctrl.kl_coef} kl_type={getattr(args, 'rl_kl_type', 'fixed')} lr={rl_lr} "
        f"n_ppo_epochs={n_ppo_epochs} "
        f"reward=(pt={reward_point_weight} obj={reward_object_weight} beta={reward_point_beta})"
    )

    global_step = 0
    start_time = time.time()

    for epoch in range(1, rl_epochs + 1):
        # eval() is set per-batch before rollout; train() is restored after the loop.
        policy_model.train()
        sum_reward = 0.0
        sum_reward_pt = 0.0
        sum_reward_obj = 0.0
        sum_reward_joint = 0.0
        sum_invalid_fmt = 0.0
        sum_extra_txt = 0.0
        sum_pg_loss = 0.0
        sum_kl = 0.0
        rollout_count = 0
        update_count = 0
        optimizer.zero_grad(set_to_none=True)

        it = tqdm(
            rl_loader,
            desc=f"RL {epoch}/{rl_epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=False,
        )

        for step, batch in enumerate(it, start=1):
            joint = to_device(batch["joint_inputs"], device=device)
            scene_images_b: list[Any] = batch["scene_images"]
            text_inputs_b: list[str] = batch["text_input"]
            gt_points_b: list[Any] = batch["gt_points"]         # list[Tensor[N,2]]
            target_label_ids_b: torch.Tensor = batch["target_label_ids"]  # [B, 5]
            target_object_valid_b: torch.Tensor = batch["target_object_valid"]
            B = len(text_inputs_b)
            G = rl_group_size
            prompt_len = int(joint["input_ids"].shape[1])

            # -------------------------------------------------------
            # 1. Rollout: sample G responses per prompt (no grad, eval)
            # eval() disables LoRA dropout so rollout is deterministic
            # and consistent with the old-logprob forward pass below.
            # -------------------------------------------------------
            policy_model.eval()
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    generated_ids = policy_model.generate(
                        joint_inputs=joint,
                        max_new_tokens=rl_max_new_tokens,
                        do_sample=True,
                        temperature=rl_temperature,
                        top_p=rl_top_p,
                        num_return_sequences=G,
                        num_beams=1,
                    )
            generated_ids_cpu = generated_ids.detach().cpu()
            input_ids_cpu = joint["input_ids"].detach().cpu()
            attn_cpu = joint.get("attention_mask", None)
            if torch.is_tensor(attn_cpu):
                attn_cpu = attn_cpu.detach().cpu()

            # -------------------------------------------------------
            # 2. Decode all B*G outputs
            # -------------------------------------------------------
            preds = decode_generated(
                processor=processor,
                generated_ids=generated_ids_cpu,
                input_ids=input_ids_cpu,
                attention_mask=attn_cpu,
                num_return_sequences=G,
            )
            preds = preds[: B * G]

            # -------------------------------------------------------
            # 3. Parse + reward computation
            # -------------------------------------------------------
            rewards_list: list[dict[str, float]] = []
            for k in range(B * G):
                b = k // G
                parsed = parse_structured_output_text(preds[k], int(num_classes), coord_bins=int(coord_bins))
                gt_pts = gt_points_b[b] if b < len(gt_points_b) else None
                # GT object ids: use target_label_ids (multi-label aware)
                raw_ids = target_label_ids_b[b].tolist() if b < int(target_label_ids_b.shape[0]) else []
                gt_ids = [int(v) for v in raw_ids if int(v) >= 0]
                # If object GT is invalid, skip object reward
                obj_valid = (
                    b < int(target_object_valid_b.numel())
                    and float(target_object_valid_b[b].item()) > 0.0
                )
                if not obj_valid:
                    gt_ids = []

                rwd = compute_total_reward(
                    parsed=parsed,
                    gt_points=gt_pts,
                    gt_obj_ids=gt_ids,
                    reward_point_weight=reward_point_weight,
                    reward_object_weight=reward_object_weight,
                    reward_joint_bonus=reward_joint_bonus,
                    reward_extra_penalty=reward_extra_penalty,
                    reward_point_beta=reward_point_beta,
                    reward_point_mode=reward_point_mode,
                    reward_point_box_radius=reward_point_box_radius,
                )
                rewards_list.append(rwd)
                sum_reward += rwd["reward_total"]
                sum_reward_pt += rwd["reward_point"]
                sum_reward_obj += rwd["reward_object"]
                sum_reward_joint += rwd["reward_joint"]
                sum_invalid_fmt += 0.0 if rwd["valid_format"] else 1.0
                sum_extra_txt += rwd["reward_extra"]
                rollout_count += 1

            # -------------------------------------------------------
            # 4. Group-normalise → advantages
            # -------------------------------------------------------
            advantages_flat: list[float] = []
            for b in range(B):
                group_rwds = [rewards_list[b * G + g]["reward_total"] for g in range(G)]
                advantages_flat.extend(group_normalize_advantages(group_rwds))
            adv_tensor = torch.tensor(advantages_flat, dtype=torch.float32)

            # -------------------------------------------------------
            # 5. Build logprob joint_inputs: B*G (prompt + sampled answer)
            # -------------------------------------------------------
            exp_scenes: list[Any] = [scene_images_b[k // G] for k in range(B * G)]
            exp_texts: list[str] = [text_inputs_b[k // G] for k in range(B * G)]
            tv_ones = torch.ones(B * G, dtype=torch.float32)

            lp_joint, _labels, _mpt, _mobj, _mfmt = build_train_inputs(
                processor=processor,
                scene_images=exp_scenes,
                text_inputs=exp_texts,
                target_texts=preds,
                target_text_valid=tv_ones,
                target_point_valid=tv_ones,
                target_object_valid=tv_ones,
                target_format_valid=tv_ones,
                max_text_length=int(getattr(args, "max_text_length", 256)),
            )

            # response_mask: all generated token positions (Rex-Omni style)
            # Falls back to structured mask for structured tokens where available.
            _struct_mask = (_mpt | _mobj | _mfmt)          # [B*G, L]
            _has_struct = _struct_mask.any(dim=1)
            if not _has_struct.all():
                _fallback = build_answer_mask(
                    processor=processor,
                    joint_inputs=lp_joint,
                    target_texts=preds,
                    target_valid=tv_ones,
                )
                _struct_mask[~_has_struct] = _fallback[~_has_struct]
            response_mask = _struct_mask                    # [B*G, L]

            lp_input_ids  = lp_joint["input_ids"]            # [B*G, L]
            lp_joint_dev  = to_device(lp_joint, device=device)
            input_ids_dev = lp_input_ids.to(device=device)
            resp_mask_dev = response_mask[:, 1:].to(device=device)   # causal shift

            # -------------------------------------------------------
            # 6. Ref log-probs — per-token [B*G, L-1] (Rex-Omni style)
            # -------------------------------------------------------
            ref_log_probs = infer_logprobs_chunked(
                model=ref_model,
                lp_joint=lp_joint,
                input_ids=lp_input_ids,
                device=device,
                amp_dtype=amp_dtype,
                micro_bsz=lp_micro_bsz,
            )   # [B*G, L-1] on CPU

            # -------------------------------------------------------
            # 7. Old log-probs — per-token [B*G, L-1] cached at rollout time
            #    (Rex-Omni: old_lp is computed once, then reused across n_ppo_epochs)
            # -------------------------------------------------------
            policy_model.eval()
            old_log_probs = infer_logprobs_chunked(
                model=policy_model,
                lp_joint=lp_joint,
                input_ids=lp_input_ids,
                device=device,
                amp_dtype=amp_dtype,
                micro_bsz=lp_micro_bsz,
            )   # [B*G, L-1] on CPU

            # -------------------------------------------------------
            # 8. n_ppo_epochs gradient updates on cached rollout buffer
            #    (Rex-Omni core: 1 rollout → multiple policy updates)
            # -------------------------------------------------------
            is_last_batch = (step == num_rl_batches)
            remainder = num_rl_batches % accum_steps
            current_accum = (
                int(remainder)
                if (int(remainder) > 0 and step >= (num_rl_batches - int(remainder) + 1))
                else int(accum_steps)
            )

            for _ppo_ep in range(n_ppo_epochs):
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    _out_new = policy_model(joint_inputs=lp_joint_dev, use_cache=False)
                new_log_probs = compute_token_logprobs(
                    _out_new["logits"], input_ids_dev,
                )   # [B*G, L-1] — on GPU, with grad

                rl_loss, grpo_stats = compute_policy_loss_per_token(
                    old_log_probs=old_log_probs.to(device=device),
                    new_log_probs=new_log_probs,
                    ref_log_probs=ref_log_probs.to(device=device),
                    advantages=adv_tensor,
                    response_mask=resp_mask_dev,
                    clip_ratio_low=rl_clip_ratio_low,
                    clip_ratio_high=rl_clip_ratio_high,
                    clip_ratio_dual=rl_clip_ratio_dual,
                    kl_beta=kl_ctrl.kl_coef,
                    kl_penalty=rl_kl_penalty,
                )
                scaled_loss = rl_loss / float(max(current_accum * n_ppo_epochs, 1))
                scaled_loss.backward()

                sum_pg_loss += float(grpo_stats["pg_loss"])
                sum_kl      += float(grpo_stats["kl_mean"])

            # Adaptive KL update after last ppo epoch
            kl_ctrl.update(
                current_kl=float(grpo_stats["kl_mean"]),
                n_steps=accum_steps,
            )

            should_step = ((step % accum_steps) == 0) or is_last_batch
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=rl_max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                update_count += 1

                if wandb_run is not None and int(getattr(args, "wandb_log_every_steps", 20)) > 0:
                    if global_step % int(getattr(args, "wandb_log_every_steps", 20)) == 0:
                        n_roll = max(rollout_count, 1)
                        wandb_run.log(
                            {
                                "rl/reward_mean":        sum_reward / n_roll,
                                "rl/reward_point_mean":  sum_reward_pt / n_roll,
                                "rl/reward_object_mean": sum_reward_obj / n_roll,
                                "rl/reward_joint_mean":  sum_reward_joint / n_roll,
                                "rl/invalid_format_rate": sum_invalid_fmt / n_roll,
                                "rl/extra_text_rate":    sum_extra_txt / n_roll,
                                "rl/kl_mean":            sum_kl / max(update_count, 1),
                                "rl/kl_coef":            kl_ctrl.kl_coef,
                                "rl/policy_loss":        sum_pg_loss / max(update_count, 1),
                                "rl/clip_frac_high":     float(grpo_stats.get("clip_frac_high", 0.0)),
                                "rl/clip_frac_low":      float(grpo_stats.get("clip_frac_low", 0.0)),
                                "rl/ratio_mean":         float(grpo_stats.get("ratio_mean", 1.0)),
                                "rl/adv_std":            float(adv_tensor.std().item()),
                                "rl/learning_rate":      float(optimizer.param_groups[0]["lr"]),
                                "rl/epoch": (float(epoch) - 1.0) + float(update_count) / max(float(updates_per_epoch), 1.0),
                            },
                            step=global_step,
                        )

            if True:
                n_r = max(rollout_count, 1)
                it.set_postfix(
                    rwd=f"{(sum_reward / n_r):.3f}",
                    kl=f"{(sum_kl / max(update_count, 1)):.4f}",
                    kl_c=f"{kl_ctrl.kl_coef:.4f}",
                )

        # -------------------------------------------------------
        # End of RL epoch: validation
        # -------------------------------------------------------
        # Restore train() mode before generation-based validation.
        policy_model.train()
        n_roll = max(rollout_count, 1)
        n_upd = max(update_count, 1)
        epoch_msg = (
            f"[RL EPOCH {epoch}] reward={sum_reward / n_roll:.4f} "
            f"fmt_invalid={sum_invalid_fmt / n_roll:.3f} "
            f"kl={sum_kl / n_upd:.4f} pg_loss={sum_pg_loss / n_upd:.4f}"
        )

        val_gen_metrics: dict[str, float] | None = None
        _run_val = (
            val_metric_loader is not None
            and (epoch % run_val_metrics_every_n == 0 or epoch == rl_epochs)
        )
        if _run_val:
            val_gen_metrics = run_test_metrics(
                model=policy_model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                show_tqdm=True,
                desc=f"RL ValMetric {epoch}",
                max_new_tokens=rl_max_new_tokens,
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
                constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
                constrained_target_order=_eval_target_order,
                constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
                constrained_loc_decoding=str(getattr(args, "constrained_loc_decoding", "argmax")),
                include_l2_breakdown=False,
            )
            maybe_save_generation_preview(
                args=args,
                out_dir=out_dir,
                model=policy_model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                preview_attr="preview_val_samples",
                filename=f"val_generation_preview_epoch_{epoch:03d}.json",
                desc=f"RL Val preview {epoch}",
                log_prefix="VAL",
                max_new_tokens=rl_max_new_tokens,
                constrained_target_order=_eval_target_order,
            )
            epoch_msg += (
                f" val_dist={float(val_gen_metrics.get('Dist', 0.0)):.6f}"
                f" val_fmt={float(val_gen_metrics.get('FormatValid', 0.0)):.4f}"
                f" val_acc@1={float(val_gen_metrics.get('Acc@1', val_gen_metrics.get('ObjectAcc', 0.0))):.4f}"
                f" val_acc@3={float(val_gen_metrics.get('Acc@3', val_gen_metrics.get('Acc@1', val_gen_metrics.get('ObjectAcc', 0.0)))):.4f}"
            )
            if "DistExpected" in val_gen_metrics:
                epoch_msg += f" val_dist_expected={float(val_gen_metrics.get('DistExpected', 0.0)):.6f}"
        print(epoch_msg)

        if wandb_run is not None:
            _wlog: dict[str, float] = {
                "rl/epoch_reward_mean": sum_reward / n_roll,
                "rl/epoch_invalid_fmt": sum_invalid_fmt / n_roll,
            }
            if isinstance(val_gen_metrics, dict):
                _wlog.update(val_metric_log_payload(val_gen_metrics))
            wandb_run.log(_wlog, step=global_step)

        # Monitor-based checkpoint save
        monitor_value = checkpoint_monitor_value(
            checkpoint_monitor,
            val_gen_metrics=val_gen_metrics,
        )
        monitor_improved = (
            monitor_value is not None
            and (
                (checkpoint_monitor_mode == "min" and monitor_value < best_monitor_value)
                or (checkpoint_monitor_mode == "max" and monitor_value > best_monitor_value)
            )
        )
        if monitor_value is not None and monitor_improved:
            best_monitor_value = float(monitor_value)
            save_checkpoint(
                out_dir / "best", epoch, policy_model, processor,
                optimizer, scheduler, clear_dir=True,
                base_vocab_size=base_vocab_size,
                token_ids_to_save=token_ids_to_save,
            )
            print(f"[RL] new best checkpoint: {checkpoint_monitor}={best_monitor_value:.6f}")

        save_checkpoint(
            out_dir / "last", epoch, policy_model, processor,
            optimizer, scheduler, clear_dir=True,
            base_vocab_size=base_vocab_size,
            token_ids_to_save=token_ids_to_save,
        )

    elapsed = time.time() - start_time
    print(
        f"[RL DONE] global_step={global_step} best_{checkpoint_monitor}={best_monitor_value:.6f} "
        f"elapsed_sec={elapsed:.1f}"
    )
    return global_step, best_monitor_value
