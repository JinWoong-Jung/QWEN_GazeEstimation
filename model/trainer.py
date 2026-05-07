from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Flash Attention defaults to a non-deterministic algorithm.*",
    category=UserWarning,
    module=r"torch\.autograd\.graph",
)


import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .datasets import GazeDataset, GazeTestDataset, MultiViewGazeDataset
from .model import QwenTextGenerationModel
from .utils.checkpoint import (
    checkpoint_monitor_value,
    infer_checkpoint_monitor_mode,
    load_added_token_rows,
    load_checkpoint_for_eval,
    load_token_rows,
    save_checkpoint,
)
from .utils.common import env_flag, parse_dtype, resolve_path, set_seed, to_autocast_dtype, to_device
from .utils.model_init import (
    _download_model_to_local_dir,  # re-export: scripts/visualize_test_samples.py 등 외부 참조용
    enable_token_id_gradients,
    init_base_model,
    init_processor,
    peft_config_has_trainable_tokens,
    resolve_model_source,
)
from .utils.config_parser import (
    build_parser,
    load_yaml_config,
    normalize_run_name,
    output_dir_from_run_name,
)
from .utils.data_utils import (
    build_reasoning_index,
    load_label_map,
    load_label_text_map,
    load_records,
    load_test_groups,
    load_test_label_map,
    load_vocab2id,
)
from .utils.eval_utils import (
    collect_generation_samples,
    decode_generated,
    maybe_save_generation_preview,
    print_generation_samples,
    print_test_metrics_table,
    run_test_metrics,
)
from .utils.gaze_tokens import parse_structured_output_text
from .utils.special_tokens import GAZE_SCHEMA_MARKERS, register_gaze_special_tokens
from .utils.loss_utils import compute_answer_loss
from .utils.processor_collate import (
    QwenRLCollator,
    QwenTestCollator,
    QwenTrainCollator,
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
from .utils.wandb_utils import finish_wandb, init_wandb, test_log_payload, val_metric_log_payload


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_id2label(vocab2id: dict[str, int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for label_text, idx in vocab2id.items():
        idx_i = int(idx)
        if idx_i not in out:
            out[idx_i] = str(label_text)
    return out


def count_valid_targets(records: list[Any]) -> int:
    n = 0
    for r in records:
        label_id = int(getattr(r, "label_id", -1))
        txt = str(getattr(r, "label_text", "") or "").strip()
        if (label_id >= 0) or bool(txt):
            n += 1
    return n


def log_target_example(tag: str, dataset: Any) -> None:
    if not env_flag("QWEN_DEBUG_TARGET_EXAMPLE"):
        return
    try:
        n = len(dataset)
        if int(n) <= 0:
            print(f"[DEBUG] {tag} target example: dataset is empty.")
            return
        sample = dataset[0]
        tgt = str(sample.get("target_text", ""))
        print(f"[DEBUG] {tag} target example:\n{tgt}")
    except Exception as e:
        print(f"[DEBUG] {tag} target example unavailable: {e}")


def infer_num_classes(vocab2id: dict[str, int], vocab2id_path: Path) -> int:
    if (not isinstance(vocab2id, dict)) or (len(vocab2id) <= 0):
        raise RuntimeError(
            f"vocab2id is missing/empty: {vocab2id_path}. "
            "Object special tokens require a valid vocab2id mapping."
        )
    n = int(len(vocab2id))
    ids: list[int] = []
    for k, v in vocab2id.items():
        try:
            idx = int(v)
        except Exception as e:
            raise RuntimeError(
                f"vocab2id is malformed at key={k!r}, value={v!r}: {e}"
            ) from e
        ids.append(idx)
    expected = list(range(n))
    got = sorted(ids)
    if got != expected:
        first = got[:10]
        last = got[-10:] if len(got) > 10 else got
        raise RuntimeError(
            "vocab2id ids must be exactly contiguous 0..N-1. "
            f"path={vocab2id_path} N={n} got_min={min(got)} got_max={max(got)} "
            f"head={first} tail={last}"
        )
    return n


def _run_rl_training(
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
    rl_temperature = float(getattr(args, "rl_temperature", 0.7))
    rl_top_p = float(getattr(args, "rl_top_p", 0.9))
    rl_epochs = max(1, int(getattr(args, "epochs", 5)))
    rl_max_new_tokens = max(16, int(getattr(args, "generation_max_new_tokens", 16)))
    _rl_val_format = str(getattr(args, "val_test_output_format", "direct")).strip().lower()
    _eval_target_order = (
        "reasoning_point_object"
        if _rl_val_format == "reasoning"
        else str(getattr(args, "constrained_target_order", "point_object"))
    )

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

            lp_joint, _labels, _mpt, _mobj, _mfmt, _mrsn = build_train_inputs(
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
                max_reasoning_tokens=int(getattr(args, "max_reasoning_tokens", 80)),
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
                f" val_obj={float(val_gen_metrics.get('ObjectAcc', 0.0)):.4f}"
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


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="sft.yaml")
    cfg_args, _ = config_parser.parse_known_args()
    config_path = resolve_path(cfg_args.config)
    config_defaults = load_yaml_config(config_path)
    config_defaults["config"] = str(cfg_args.config)

    args = build_parser(defaults=config_defaults).parse_args()
    print(f"[INFO] loaded config: {resolve_path(args.config)}")
    set_seed(args.seed)

    run_name = normalize_run_name(getattr(args, "run_name", ""))
    if run_name:
        args.output_dir = output_dir_from_run_name(args.output_dir, run_name)
        args.wandb_run_name = run_name
        print(f"[INFO] run_name={run_name}")

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    wandb_run = init_wandb(args=args, root=ROOT)

    if args.device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but no GPU is available.")
    device = torch.device(args.device)

    model_path = resolve_model_source(args.model_path)
    checkpoint_dir = resolve_path(args.checkpoint_dir) if str(args.checkpoint_dir).strip() else None
    train_ann = resolve_path(args.train_ann)
    val_ann = resolve_path(args.val_ann)
    test_ann = resolve_path(args.test_ann)
    image_root = resolve_path(args.image_root)
    test_image_root = resolve_path(args.test_image_root)
    train_labels = resolve_path(args.train_labels)
    val_labels = resolve_path(args.val_labels)
    test_labels = resolve_path(args.test_labels)
    vocab2id_path = resolve_path(args.vocab2id)

    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    num_classes = infer_num_classes(vocab2id, vocab2id_path)
    id2label = build_id2label(vocab2id)
    print(f"[INFO] loaded vocab2id classes: {len(vocab2id)} (id_range=0..{max(num_classes - 1, 0)})")

    train_stage = str(getattr(args, "train_stage", "sft")).strip().lower()
    _inference_only = bool(getattr(args, "test_only", False)) or bool(getattr(args, "eval_only", False))
    if train_stage not in {"sft", "rl"}:
        raise ValueError(f"train_stage must be 'sft' or 'rl', got: {train_stage!r}")

    # prompt_text_direct: used for point_object views and val/test when val_test_output_format=direct
    # prompt_text_reasoning: used for reasoning views and val/test when val_test_output_format=reasoning
    # Falls back to prompt_text if the dedicated keys are absent
    _prompt_fallback = str(getattr(args, "prompt_text", "") or "")
    _prompt_text_direct = str(getattr(args, "prompt_text_direct", "") or "") or _prompt_fallback
    _prompt_text_reasoning_view = str(getattr(args, "prompt_text_reasoning", "") or "") or _prompt_fallback
    prompt_text_for_run = _prompt_text_direct

    # --- sample_mode: replaces use_reasoning ---
    _legacy_use_reasoning = bool(getattr(args, "use_reasoning", False))
    sample_mode = str(getattr(args, "sample_mode", "direct_only")).strip().lower()
    if _legacy_use_reasoning and sample_mode == "direct_only":
        sample_mode = "direct+reasoning"
        print("[WARN] use_reasoning=True (legacy flag); treating as sample_mode='direct+reasoning'")
    _VALID_SAMPLE_MODES = {"direct_only", "reasoning_only", "direct&reasoning", "direct+reasoning"}
    if sample_mode not in _VALID_SAMPLE_MODES:
        raise ValueError(
            f"train.sample_mode must be one of {sorted(_VALID_SAMPLE_MODES)}, got: {sample_mode!r}"
        )
    print(f"[INFO] sample_mode={sample_mode!r}")

    # --- val/test output format ---
    val_test_format = str(getattr(args, "val_test_output_format", "direct")).strip().lower()
    if val_test_format not in {"direct", "reasoning"}:
        raise ValueError(
            f"eval.val_test_output_format must be 'direct' or 'reasoning', got: {val_test_format!r}"
        )
    if val_test_format == "reasoning":
        _prompt_text_eval = _prompt_text_reasoning_view
        _eval_target_order = "reasoning_point_object"
        _force_eval = True
    else:
        _prompt_text_eval = _prompt_text_direct
        _eval_target_order = "point_object"
        _force_eval = False
    print(f"[INFO] val_test_output_format={val_test_format!r} → target_order={_eval_target_order!r}")

    # generation_max_new_tokens: auto-bump for reasoning eval
    _gen_max_tokens = int(getattr(args, "generation_max_new_tokens", 8))
    if val_test_format == "reasoning":
        _max_rsn_tokens = int(getattr(args, "max_reasoning_tokens", 80))
        _gen_max_tokens_eval = max(_gen_max_tokens, _max_rsn_tokens + 8)
        if _gen_max_tokens_eval > _gen_max_tokens:
            print(
                f"[INFO] val_test_output_format='reasoning': generation_max_new_tokens "
                f"auto-set {_gen_max_tokens} → {_gen_max_tokens_eval}"
            )
    else:
        _gen_max_tokens_eval = _gen_max_tokens

    filter_invalid = bool(getattr(args, "filter_invalid_object_samples", True))
    loss_weights = {
        "point": float(getattr(args, "loss_point_weight", 1.0)),
        "object": float(getattr(args, "loss_object_weight", 1.0)),
        "format": float(getattr(args, "loss_format_weight", 0.25)),
        "reasoning": float(getattr(args, "loss_reasoning_weight", 0.3)),
    }
    coord_bins = int(getattr(args, "coord_bins", 1000))
    if coord_bins <= 0:
        raise ValueError(f"coord_bins must be positive, got: {coord_bins}")

    load_dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
        print("[WARN] non-CUDA device detected; forcing model dtype to float32.")
        load_dtype = torch.float32
    amp_dtype = to_autocast_dtype(load_dtype)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if load_dtype != "auto":
        model_kwargs["dtype"] = load_dtype

    _resize_mode = str(getattr(args, "image_resize_mode", "native")).strip().lower()
    _fixed_resize = (_resize_mode == "fixed")
    _scene_size: tuple[int, int] | None = (
        (int(args.scene_h), int(args.scene_w)) if _fixed_resize else None
    )
    _proc_min_pixels: int | None = None if _fixed_resize else int(getattr(args, "min_pixels", 12544))
    _proc_max_pixels: int | None = None if _fixed_resize else int(getattr(args, "max_pixels", 2007040))

    # --- Phase 2 init order: processor → register tokens → base model → resize → LoRA ---
    processor = init_processor(
        model_path=model_path,
        checkpoint_dir=checkpoint_dir,
        min_pixels=_proc_min_pixels,
        max_pixels=_proc_max_pixels,
    )
    token_id_map = register_gaze_special_tokens(
        tokenizer=processor.tokenizer,
        num_classes=num_classes,
        coord_bins=coord_bins,
    )
    new_vocab_size = len(processor.tokenizer)
    gaze_token_ids = sorted({int(v) for v in token_id_map.values() if int(v) >= 0})
    print(
        f"[INFO] tokenizer extended: vocab_size={new_vocab_size} "
        f"(added loc tokens: {coord_bins}, obj tokens: {num_classes}, "
        f"fmt tokens: {len(GAZE_SCHEMA_MARKERS)})"
    )

    # Build ordered loc_token_ids tensor for Gaussian soft-label CE.
    # Entries are the vocab IDs of <loc_000>..<loc_{coord_bins-1}> in bin order.
    from .utils.special_tokens import format_loc_token, _loc_token_width
    _lw = _loc_token_width(coord_bins)
    _loc_id_list = [int(token_id_map.get(format_loc_token(b, _lw), -1)) for b in range(coord_bins)]
    loc_token_ids_tensor: torch.Tensor | None = (
        torch.tensor(_loc_id_list, dtype=torch.long)
        if all(i >= 0 for i in _loc_id_list) else None
    )
    loc_token_ids_for_loss: torch.Tensor | None = (
        loc_token_ids_tensor.to(device=device)
        if loc_token_ids_tensor is not None else None
    )
    gaussian_point_sigma = float(getattr(args, "gaussian_point_sigma", 0.0))

    if bool(getattr(args, "test_only", False)):
        print("[INFO] test_only=True; skipping train/val loading and training.")
        start_time = time.time()

        test_label_map, test_label_text_map, test_label_ids_map, test_label_stats = load_test_label_map(
            test_labels,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
        )
        print(
            "[INFO] test label map: "
            f"rows={test_label_stats['rows']} mapped={test_label_stats['mapped']} "
            f"missing_text={test_label_stats['missing_text']} unknown_text={test_label_stats['unknown_text']} "
            f"conflicts={test_label_stats['conflicts']}"
        )

        test_collator = QwenTestCollator(
            processor=processor,
            max_text_length=int(args.max_text_length),
            scene_size=_scene_size,
        )

        base_qwen = init_base_model(model_path=model_path, model_kwargs=model_kwargs)
        base_vocab_size = int(base_qwen.get_input_embeddings().weight.shape[0])
        base_qwen.resize_token_embeddings(new_vocab_size)

        adapter_dir = (checkpoint_dir / "lora_adapter") if checkpoint_dir is not None else None
        if adapter_dir is not None and adapter_dir.exists():
            qwen_model = PeftModel.from_pretrained(
                base_qwen, model_id=str(adapter_dir), is_trainable=False,
            )
            print(f"[INFO] loaded LoRA adapter from: {adapter_dir}")
            if checkpoint_dir is not None:
                _tmp_model = QwenTextGenerationModel(qwen_model=qwen_model)
                rows_loaded = load_token_rows(
                    ckpt_dir=checkpoint_dir,
                    model=_tmp_model,
                    device=device,
                )
                if rows_loaded:
                    print(f"[INFO] restored gaze token rows from: {checkpoint_dir / 'gaze_token_rows.pt'}")
                else:
                    rows_loaded = load_added_token_rows(
                        ckpt_dir=checkpoint_dir,
                        model=_tmp_model,
                        device=device,
                    )
                    if rows_loaded:
                        print(f"[INFO] restored added token rows from: {checkpoint_dir / 'added_token_rows.pt'}")
        else:
            qwen_model = base_qwen
            print("[INFO] adapter checkpoint not found; running zero-shot base model.")

        model = QwenTextGenerationModel(qwen_model=qwen_model).to(device)
        model.eval()

        test_groups = load_test_groups(
            annotation_file=test_ann,
            image_root=test_image_root,
            test_label_map=test_label_map,
            test_label_text_map=test_label_text_map,
            test_label_ids_map=test_label_ids_map,
            split_prefix=args.test_split_prefix,
            strip_split_prefix=bool(args.test_strip_split_prefix),
            bbox_round_decimals=int(args.test_bbox_round_decimals),
        )
        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(f"[TEST] groups={len(test_groups)}")
            test_ds = GazeTestDataset(
                groups=test_groups,
                prompt_template=args.prompt_template,
                prompt_text=_prompt_text_eval,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                visual_prompting=bool(args.visual_prompting),
                image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
                coord_bins=coord_bins,
                force_reasoning_format=_force_eval,
                target_order=_eval_target_order,
            )
            log_target_example("test_only", test_ds)
            _tnw = int(args.num_workers)
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=_tnw,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
                persistent_workers=(_tnw > 0),
                prefetch_factor=(2 if _tnw > 0 else None),
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                show_tqdm=True,
                desc="Test",
                max_new_tokens=_gen_max_tokens_eval,
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
                constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
                constrained_target_order=_eval_target_order,
                constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
                constrained_loc_decoding=str(getattr(args, "constrained_loc_decoding", "argmax")),
                max_reasoning_tokens=int(getattr(args, "max_reasoning_tokens", 80)),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(test_log_payload(test_metrics), step=0)
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            maybe_save_generation_preview(
                args=args,
                out_dir=out_dir,
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                max_new_tokens=_gen_max_tokens_eval,
                constrained_target_order=_eval_target_order,
            )

        elapsed = time.time() - start_time
        finish_wandb(wandb_run)
        print(f"[DONE] test_only=True elapsed_sec={elapsed:.1f}")
        return

    train_label_map, train_label_stats = load_label_map(
        train_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        use_embed_fallback=False,
    )
    val_label_map, val_label_stats = load_label_map(
        val_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        use_embed_fallback=False,
    )

    train_label_text_map, train_label_text_stats = load_label_text_map(
        train_labels, text_key="gaze_pseudo_label",
    )
    val_label_text_map, val_label_text_stats = load_label_text_map(
        val_labels, text_key="gaze_pseudo_label",
    )
    test_label_map, test_label_text_map, test_label_ids_map, test_label_stats = load_test_label_map(
        test_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
    )

    print(
        "[INFO] train label id coverage: "
        f"rows={train_label_stats['rows']} mapped={train_label_stats['mapped']} "
        f"missing_text={train_label_stats['missing_text']} unknown_text={train_label_stats['unknown_text']} "
        f"primary_mapped={train_label_stats.get('primary_mapped', 0)}"
    )
    print(
        "[INFO] val label id coverage: "
        f"rows={val_label_stats['rows']} mapped={val_label_stats['mapped']} "
        f"missing_text={val_label_stats['missing_text']} unknown_text={val_label_stats['unknown_text']}"
    )
    print(
        "[INFO] test label map: "
        f"rows={test_label_stats['rows']} mapped={test_label_stats['mapped']} "
        f"missing_text={test_label_stats['missing_text']} unknown_text={test_label_stats['unknown_text']} "
        f"conflicts={test_label_stats['conflicts']}"
    )

    train_records = load_records(
        annotation_file=train_ann,
        image_root=image_root,
        label_map=train_label_map,
        label_text_map=train_label_text_map,
        split_prefix=args.split_prefix,
        strip_split_prefix=bool(args.strip_split_prefix),
    )
    val_records = load_records(
        annotation_file=val_ann,
        image_root=image_root,
        label_map=val_label_map,
        label_text_map=val_label_text_map,
        split_prefix=args.split_prefix,
        strip_split_prefix=bool(args.strip_split_prefix),
    )
    if not train_records:
        raise RuntimeError("No train samples were loaded.")

    print(
        f"[INFO] train_records={len(train_records)} val_records={len(val_records)} "
        f"train_valid_targets={count_valid_targets(train_records)} "
        f"val_valid_targets={count_valid_targets(val_records)}"
    )

    image_cache_size = max(0, int(getattr(args, "image_cache_size", 0)))
    train_augmentation_mode = str(getattr(args, "train_augmentation_mode", "full") or "full").strip().lower()
    _aug_modes = {"full", "crop_flip_color", "default", "color", "color_only", "photometric", "safe", "no_crop", "flip_color", "hflip_color", "none", "no_aug", "off", "false"}
    if train_augmentation_mode not in _aug_modes:
        raise ValueError(
            f"unsupported train_augmentation_mode={train_augmentation_mode!r}; "
            "expected one of: full, color_only, no_crop, no_aug"
        )

    reasoning_index = None
    _needs_reasoning_index = sample_mode in {"reasoning_only", "direct&reasoning", "direct+reasoning"}
    if _needs_reasoning_index:
        reasoning_dir = resolve_path(str(getattr(args, "train_reasoning_dir", "")))
        if reasoning_dir.exists():
            reasoning_index = build_reasoning_index(reasoning_dir)
            print(
                f"[INFO] Loaded reasoning index: {len(reasoning_index)} entries "
                f"({len(reasoning_index)}/{max(1, len(train_records))} train samples)"
            )
            matched_reasoning = 0
            reasoning_hits: list[str] = []
            reasoning_misses: list[str] = []
            for rec in train_records:
                folder = Path(rec.image_rel).parent.name
                stem = Path(rec.image_rel).stem
                key = f"{folder}/{stem}_{int(rec.sample_id)}"
                rpath = reasoning_index.get(key)
                if rpath is not None:
                    matched_reasoning += 1
                    if len(reasoning_hits) < 3:
                        try:
                            rel_rpath = rpath.relative_to(reasoning_dir)
                        except ValueError:
                            rel_rpath = rpath
                        reasoning_hits.append(f"{key} -> {rel_rpath}")
                elif len(reasoning_misses) < 3:
                    reasoning_misses.append(key)
            missing_reasoning = len(train_records) - matched_reasoning
            print(
                "[INFO] Reasoning index match: "
                f"matched={matched_reasoning}/{len(train_records)} "
                f"missing={missing_reasoning} "
                f"hit_ratio={matched_reasoning / max(1, len(train_records)):.6f}"
            )
            if reasoning_hits:
                print("[INFO] Reasoning match examples: " + " ; ".join(reasoning_hits))
            if reasoning_misses:
                print("[WARN] Reasoning missing examples: " + " ; ".join(reasoning_misses))
            if matched_reasoning == 0 and len(train_records) > 0:
                print(
                    "[WARN] No train records matched the reasoning index. "
                    "Check that reasoning files are stored as folder/stem_sampleid.txt."
                )
        else:
            print(
                f"[WARN] sample_mode={sample_mode!r} requires reasoning data "
                f"but train_reasoning_dir not found: {reasoning_dir}"
            )

    _max_reasoning_words = int(getattr(args, "max_reasoning_words", 60))
    _max_reasoning_chars = int(getattr(args, "max_reasoning_chars", 500))
    _reasoning_view_ratio = float(getattr(args, "reasoning_view_ratio", 0.2))

    if sample_mode == "direct_only":
        train_ds: GazeDataset | MultiViewGazeDataset = GazeDataset(
            records=train_records,
            prompt_template=args.prompt_template,
            prompt_text=prompt_text_for_run,
            apply_augmentation=True,
            id2label=id2label,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
            num_classes=int(num_classes),
            visual_prompting=bool(args.visual_prompting),
            image_cache_size=image_cache_size,
            filter_invalid_object_samples=filter_invalid,
            coord_bins=coord_bins,
            train_augmentation_mode=train_augmentation_mode,
            target_order="point_object",
        )
    elif sample_mode == "reasoning_only":
        # Only records that have reasoning GT files are used for training
        _capable_records = [
            r for r in train_records
            if reasoning_index is not None and reasoning_index.get(
                f"{Path(r.image_rel).parent.name}/{Path(r.image_rel).stem}_{int(r.sample_id)}"
            ) is not None
        ]
        if not _capable_records:
            raise RuntimeError(
                "sample_mode='reasoning_only' but no train records matched the reasoning index. "
                "Check train_reasoning_dir or switch to a different sample_mode."
            )
        print(
            f"[INFO] reasoning_only: using {len(_capable_records)}/{len(train_records)} "
            f"records with reasoning GT"
        )
        train_ds = GazeDataset(
            records=_capable_records,
            prompt_template=args.prompt_template,
            prompt_text=_prompt_text_reasoning_view,
            apply_augmentation=True,
            id2label=id2label,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
            num_classes=int(num_classes),
            visual_prompting=bool(args.visual_prompting),
            image_cache_size=image_cache_size,
            filter_invalid_object_samples=filter_invalid,
            coord_bins=coord_bins,
            train_augmentation_mode=train_augmentation_mode,
            reasoning_index=reasoning_index,
            max_reasoning_words=_max_reasoning_words,
            max_reasoning_chars=_max_reasoning_chars,
            force_reasoning_format=True,
            target_order="reasoning_point_object",
        )
    else:
        # "direct&reasoning" or "direct+reasoning"
        train_ds = MultiViewGazeDataset(
            records=train_records,
            prompt_template=args.prompt_template,
            prompt_text_direct=_prompt_text_direct,
            prompt_text_reasoning=_prompt_text_reasoning_view,
            id2label=id2label,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
            num_classes=int(num_classes),
            visual_prompting=bool(args.visual_prompting),
            image_cache_size=image_cache_size,
            filter_invalid_object_samples=filter_invalid,
            coord_bins=coord_bins,
            reasoning_index=reasoning_index,
            max_reasoning_words=_max_reasoning_words,
            max_reasoning_chars=_max_reasoning_chars,
            reasoning_ratio=_reasoning_view_ratio,
            train_augmentation_mode=train_augmentation_mode,
            seed=int(getattr(args, "seed", 42)),
            sample_mode=sample_mode,
        )
    val_ds = GazeDataset(
        records=val_records,
        prompt_template=args.prompt_template,
        prompt_text=_prompt_text_eval,
        apply_augmentation=False,
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        visual_prompting=bool(args.visual_prompting),
        image_cache_size=image_cache_size,
        filter_invalid_object_samples=filter_invalid,
        coord_bins=coord_bins,
        force_reasoning_format=_force_eval,
        target_order=_eval_target_order,
    )

    # Count filtered samples
    n_train_valid = sum(
        1 for i in range(len(train_ds)) if float(train_ds[i]["target_text_valid"].item()) > 0.0
    ) if len(train_ds) <= 1000 else -1
    print(
        f"[INFO] structured pipeline: filter_invalid_object_samples={filter_invalid} "
        f"train_augmentation_mode={train_augmentation_mode} "
        f"train_valid_structured={n_train_valid if n_train_valid >= 0 else 'not_counted'}"
    )
    log_target_example("train", train_ds)
    log_target_example("val", val_ds)

    train_collator = QwenTrainCollator(
        processor=processor,
        max_text_length=int(args.max_text_length),
        scene_size=_scene_size,
    )
    test_collator = QwenTestCollator(
        processor=processor,
        max_text_length=int(args.max_text_length),
        scene_size=_scene_size,
    )

    _nw = int(args.num_workers)
    _persistent = _nw > 0
    _prefetch = 2 if _nw > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_collator,
        persistent_workers=False,
        prefetch_factor=_prefetch,
    )
    val_metric_loader = None
    if bool(getattr(args, "run_val_metrics", True)):
        _val_metric_nw = min(_nw, 2)
        val_metric_loader = DataLoader(
            val_ds,
            batch_size=max(1, int(args.test_batch_size)),
            shuffle=False,
            num_workers=_val_metric_nw,
            pin_memory=(device.type == "cuda"),
            collate_fn=test_collator,
            persistent_workers=False,
            prefetch_factor=2 if _val_metric_nw > 0 else None,
        )

    # --- init order: base model → resize embeddings → LoRA ---
    base_qwen = init_base_model(model_path=model_path, model_kwargs=model_kwargs)
    base_vocab_size = int(base_qwen.get_input_embeddings().weight.shape[0])
    base_qwen.resize_token_embeddings(new_vocab_size)
    print(f"[INFO] model embeddings resized to vocab_size={new_vocab_size}")

    if args.gradient_checkpointing and hasattr(base_qwen, "gradient_checkpointing_enable"):
        base_qwen.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(base_qwen, "enable_input_require_grads"):
        base_qwen.enable_input_require_grads()

    adapter_dir = (checkpoint_dir / "lora_adapter") if checkpoint_dir is not None else None
    if adapter_dir is not None and adapter_dir.exists():
        qwen_lora = PeftModel.from_pretrained(
            base_qwen,
            model_id=str(adapter_dir),
            is_trainable=not bool(args.eval_only),
        )
        print(f"[INFO] loaded LoRA adapter from: {adapter_dir}")
        if checkpoint_dir is not None:
            _tmp_model = QwenTextGenerationModel(qwen_model=qwen_lora)
            rows_loaded = load_token_rows(
                ckpt_dir=checkpoint_dir,
                model=_tmp_model,
                device=device,
            )
            if rows_loaded:
                print(f"[INFO] restored gaze token rows from: {checkpoint_dir / 'gaze_token_rows.pt'}")
            else:
                rows_loaded = load_added_token_rows(
                    ckpt_dir=checkpoint_dir,
                    model=_tmp_model,
                    device=device,
                )
                if rows_loaded:
                    print(f"[INFO] restored added token rows from: {checkpoint_dir / 'added_token_rows.pt'}")
    else:
        target_modules = [x.strip() for x in str(args.lora_target_modules).split(",") if x.strip()]
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias=str(args.lora_bias),
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            trainable_token_indices=gaze_token_ids,
        )
        qwen_lora = get_peft_model(base_qwen, lora_cfg)
        qwen_lora.print_trainable_parameters()

    # New runs use PEFT's trainable_token_indices, which avoids making the full
    # embedding matrix trainable. Older checkpoints may not have that adapter
    # metadata, so keep the row-mask hook as a compatibility fallback.
    if (
        not bool(getattr(args, "eval_only", False))
        and not peft_config_has_trainable_tokens(qwen_lora)
    ):
        enable_token_id_gradients(qwen_lora, gaze_token_ids)
        below_base = sum(1 for i in gaze_token_ids if i < base_vocab_size)
        above_base = sum(1 for i in gaze_token_ids if i >= base_vocab_size)
        print(
            f"[INFO] enabled fallback gradient hook for gaze token rows: "
            f"total={len(gaze_token_ids)} below_base={below_base} above_base={above_base} "
            f"base_vocab_size={base_vocab_size} new_vocab_size={new_vocab_size}"
        )
    elif not bool(getattr(args, "eval_only", False)):
        print(
            f"[INFO] using PEFT trainable_token_indices for gaze token rows: "
            f"total={len(gaze_token_ids)}"
        )

    model = QwenTextGenerationModel(qwen_model=qwen_lora).to(device)

    # ------------------------------------------------------------------
    # Stage-2 RL branch — runs instead of the SFT training loop.
    # Handles training, optional test eval, then returns.  The SFT loop
    # below is never reached when train_stage == "rl".
    # ------------------------------------------------------------------
    if train_stage == "rl" and not _inference_only:
        if checkpoint_dir is None:
            raise ValueError(
                "train_stage='rl' requires --checkpoint_dir pointing to a completed SFT "
                "checkpoint (must contain lora_adapter/).  RL from a randomly-initialised "
                "LoRA is not supported."
            )
        _rl_start = time.time()
        rl_global_step, rl_best_monitor_value = _run_rl_training(
            args=args,
            policy_model=model,
            processor=processor,
            train_ds=train_ds,
            val_metric_loader=val_metric_loader,
            device=device,
            amp_dtype=amp_dtype,
            num_classes=int(num_classes),
            out_dir=out_dir,
            base_vocab_size=base_vocab_size,
            new_vocab_size=new_vocab_size,
            model_path=model_path,
            model_kwargs=model_kwargs,
            checkpoint_dir=checkpoint_dir,
            wandb_run=wandb_run,
            scene_size=_scene_size,
            coord_bins=coord_bins,
            token_ids_to_save=gaze_token_ids,
        )

        if True:  # run_test always enabled
            _rl_best_dir = out_dir / "best"
            if _rl_best_dir.exists():
                _rl_loaded = load_checkpoint_for_eval(
                    ckpt_dir=_rl_best_dir, model=model, device=device,
                )
                if _rl_loaded:
                    print(f"[RL] loaded best checkpoint for test: {_rl_best_dir}")
                else:
                    print(f"[WARN] RL best checkpoint did not load fully: {_rl_best_dir}")
            else:
                print("[WARN] RL best checkpoint not found; testing current in-memory model.")

            test_groups = load_test_groups(
                annotation_file=test_ann,
                image_root=test_image_root,
                test_label_map=test_label_map,
                test_label_text_map=test_label_text_map,
                test_label_ids_map=test_label_ids_map,
                split_prefix=args.test_split_prefix,
                strip_split_prefix=bool(args.test_strip_split_prefix),
                bbox_round_decimals=int(args.test_bbox_round_decimals),
            )
            if not test_groups:
                print("[TEST] no valid test groups found.")
            else:
                print(f"[TEST] groups={len(test_groups)}")
                test_ds = GazeTestDataset(
                    groups=test_groups,
                    prompt_template=args.prompt_template,
                    prompt_text=_prompt_text_eval,
                    id2label=id2label,
                    vocab2id=vocab2id,
                    vocab2id_lower=vocab2id_lower,
                    num_classes=int(num_classes),
                    visual_prompting=bool(args.visual_prompting),
                    image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
                    coord_bins=coord_bins,
                    force_reasoning_format=_force_eval,
                    target_order=_eval_target_order,
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=max(1, int(args.test_batch_size)),
                    shuffle=False,
                    num_workers=_nw,
                    pin_memory=(device.type == "cuda"),
                    collate_fn=test_collator,
                    persistent_workers=_persistent,
                    prefetch_factor=_prefetch,
                )
                test_metrics = run_test_metrics(
                    model=model,
                    loader=test_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    processor=processor,
                    num_classes=int(num_classes),
                    coord_bins=coord_bins,
                    show_tqdm=True,
                    desc="Test",
                    max_new_tokens=_gen_max_tokens_eval,
                    num_beams=int(getattr(args, "generation_num_beams", 1)),
                    repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                    no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                    stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
                    constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
                    constrained_target_order=_eval_target_order,
                    constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
                    constrained_loc_decoding=str(getattr(args, "constrained_loc_decoding", "argmax")),
                    max_reasoning_tokens=int(getattr(args, "max_reasoning_tokens", 80)),
                )
                print_test_metrics_table(test_metrics)
                if wandb_run is not None:
                    wandb_run.log(test_log_payload(test_metrics), step=rl_global_step)
                (out_dir / "test_metrics.json").write_text(
                    json.dumps(test_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                maybe_save_generation_preview(
                    args=args,
                    out_dir=out_dir,
                    model=model,
                    loader=test_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    processor=processor,
                    num_classes=int(num_classes),
                    coord_bins=coord_bins,
                    max_new_tokens=_gen_max_tokens_eval,
                    constrained_target_order=_eval_target_order,
                )

        finish_wandb(wandb_run)
        print(
            f"[DONE] RL stage global_step={rl_global_step} "
            f"best_{str(getattr(args, 'checkpoint_monitor', 'val_dist')).strip() or 'val_dist'}="
            f"{rl_best_monitor_value:.6f} "
            f"elapsed_sec={time.time() - _rl_start:.1f}"
        )
        return  # ← always return; SFT loop never runs for RL stage

    # ------------------------------------------------------------------
    # SFT training loop — only reached when train_stage == "sft".
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    accum_steps = max(int(args.grad_accum_steps), 1)
    num_train_batches = len(train_loader)
    updates_per_epoch = max(1, math.ceil(num_train_batches / accum_steps))
    total_updates = max(1, updates_per_epoch * max(int(args.epochs), 1))
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    print(
        f"[INFO] structured SFT loss: point_w={loss_weights['point']:.2f} "
        f"object_w={loss_weights['object']:.2f} format_w={loss_weights['format']:.2f} "
        f"reasoning_w={loss_weights['reasoning']:.2f}"
    )
    checkpoint_monitor = str(getattr(args, "checkpoint_monitor", "val_dist")).strip() or "val_dist"
    checkpoint_monitor_mode = infer_checkpoint_monitor_mode(
        checkpoint_monitor,
        str(getattr(args, "checkpoint_monitor_mode", "auto")),
    )
    best_monitor_value = float("inf") if checkpoint_monitor_mode == "min" else -float("inf")
    print(f"[INFO] checkpoint monitor: {checkpoint_monitor} ({checkpoint_monitor_mode})")
    run_val_metrics_every_n = max(1, int(getattr(args, "run_val_metrics_every_n_epochs", 5)))
    global_step = 0
    start_time = time.time()

    effective_epochs = 0 if bool(args.eval_only) else int(args.epochs)
    if bool(args.eval_only):
        print("[INFO] eval_only=True; skipping training loop.")

    for epoch in range(1, effective_epochs + 1):
        if hasattr(train_ds, "resample_epoch_views"):
            train_ds.resample_epoch_views()
        elif hasattr(train_ds, "resample_reasoning_views"):
            train_ds.resample_reasoning_views()
        model.train()
        sum_loss = 0.0
        sample_count = 0
        step_count = 0
        updates_done_in_epoch = 0
        train_log_every = max(1, int(getattr(args, "wandb_log_every_steps", 20)))
        skipped_all_ignore = 0
        _t_fwd_sum = 0.0
        _t_bwd_sum = 0.0
        _t_data_sum = 0.0
        _t_step_sum = 0.0

        optimizer.zero_grad(set_to_none=True)
        train_iter = tqdm(
            train_loader,
            desc=f"Train {epoch}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=False,
        )

        remainder_steps = num_train_batches % accum_steps
        last_window_start = (
            (num_train_batches - int(remainder_steps) + 1)
            if int(remainder_steps) > 0
            else (num_train_batches + 1)
        )

        _t_data0 = time.perf_counter()
        for step, batch in enumerate(train_iter, start=1):
            _t_step0 = time.perf_counter()
            _t_data_sum += _t_step0 - _t_data0
            labels = batch["labels"].to(device)
            if torch.all(labels.eq(-100)):
                skipped_all_ignore += 1
                _t_data0 = time.perf_counter()
                continue

            joint_inputs = to_device(batch["joint_inputs"], device=device)
            bsz = int(labels.shape[0])
            is_last_batch = (step == num_train_batches)
            current_accum_steps = (
                int(remainder_steps)
                if (int(remainder_steps) > 0 and step >= int(last_window_start))
                else int(accum_steps)
            )

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
                    loss_mask_reasoning=batch.get("loss_mask_reasoning", None),
                    weight_point=loss_weights["point"],
                    weight_object=loss_weights["object"],
                    weight_format=loss_weights["format"],
                    weight_reasoning=loss_weights["reasoning"],
                    compute_format_rate=False,
                    loc_token_ids=loc_token_ids_for_loss,
                    gaussian_sigma=gaussian_point_sigma,
                )
                raw_loss = losses["loss"]
                loss = raw_loss / float(max(current_accum_steps, 1))
            _t_fwd_sum += time.perf_counter() - _t_fwd0

            _t_bwd0 = time.perf_counter()
            loss.backward()
            _t_bwd_sum += time.perf_counter() - _t_bwd0

            should_step = ((step % accum_steps) == 0) or is_last_batch
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
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

                        wandb_run.log(
                            {
                                "train/loss": float(raw_loss.detach().item()),
                                "train/loss_point": float(losses["loss_point"].detach().item()),
                                "train/loss_object": float(losses["loss_object"].detach().item()),
                                "train/loss_format": float(losses["loss_format"].detach().item()),
                                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "train/grad_norm": grad_norm_value,
                                "train/epoch": epoch_progress,
                            },
                            step=global_step,
                        )

            sum_loss += float(raw_loss.detach().item()) * float(bsz)
            sample_count += bsz
            step_count += 1
            _t_step_sum += time.perf_counter() - _t_step0
            _t_data0 = time.perf_counter()
            if True:
                train_iter.set_postfix(loss=f"{(sum_loss / max(sample_count, 1)):.4f}")

        if step_count == 0 or sample_count == 0:
            raise RuntimeError(
                "No effective training batches were produced. "
                "All batches may have empty target text (all labels ignored)."
            )

        train_loss = float(sum_loss / float(sample_count))
        if skipped_all_ignore > 0:
            print(f"[INFO] skipped all-ignore batches in train epoch: {skipped_all_ignore}")
        _t_other = max(0.0, _t_step_sum - _t_fwd_sum - _t_bwd_sum)
        print(
            f"[TIME train {epoch}] "
            f"data_wait={_t_data_sum:.1f}s "
            f"fwd_loss={_t_fwd_sum:.1f}s "
            f"backward={_t_bwd_sum:.1f}s "
            f"other_step={_t_other:.1f}s "
            f"steps={step_count}"
        )

        val_gen_metrics = None
        _run_val_gen = (
            val_metric_loader is not None
            and len(val_ds) > 0
            and (epoch % run_val_metrics_every_n == 0 or epoch == effective_epochs)
        )
        if _run_val_gen:
            val_gen_metrics = run_test_metrics(
                model=model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                show_tqdm=True,
                desc=f"ValMetric {epoch}/{args.epochs}",
                max_new_tokens=_gen_max_tokens_eval,
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
                constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
                constrained_target_order=_eval_target_order,
                constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
                constrained_loc_decoding=str(getattr(args, "constrained_loc_decoding", "argmax")),
                max_reasoning_tokens=int(getattr(args, "max_reasoning_tokens", 80)),
                include_l2_breakdown=False,
            )
            maybe_save_generation_preview(
                args=args,
                out_dir=out_dir,
                model=model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                preview_attr="preview_val_samples",
                filename=f"val_generation_preview_epoch_{epoch:03d}.json",
                desc=f"Val preview {epoch}/{args.epochs}",
                log_prefix="VAL",
                max_new_tokens=_gen_max_tokens_eval,
                constrained_target_order=_eval_target_order,
            )

        epoch_msg = f"[EPOCH {epoch}] train_loss={train_loss:.6f}"
        if isinstance(val_gen_metrics, dict):
            epoch_msg += (
                f" val_dist={float(val_gen_metrics.get('Dist', 0.0)):.6f}"
                f" val_object_acc={float(val_gen_metrics.get('ObjectAcc', 0.0)):.6f}"
                f" val_format_valid={float(val_gen_metrics.get('FormatValid', 0.0)):.6f}"
                f" val_l2_cov={float(val_gen_metrics.get('PointL2ValidFrac', 0.0)):.3f}"
            )
            if "DistExpected" in val_gen_metrics:
                epoch_msg += f" val_dist_expected={float(val_gen_metrics.get('DistExpected', 0.0)):.6f}"
        print(epoch_msg)

        if wandb_run is not None:
            payload: dict[str, float] = {}
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics))
            if payload:
                wandb_run.log(payload, step=global_step)

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
        if monitor_value is None:
            print(f"[WARN] checkpoint monitor {checkpoint_monitor!r} is unavailable this epoch; skipping save.")
        elif monitor_improved:
            best_monitor_value = float(monitor_value)
            best_dir = out_dir / "best"
            print(f"[INFO] new best checkpoint: {checkpoint_monitor}={best_monitor_value:.6f}")
            save_checkpoint(
                best_dir,
                epoch,
                model,
                processor,
                optimizer,
                scheduler,
                clear_dir=True,
                base_vocab_size=base_vocab_size,
                token_ids_to_save=gaze_token_ids,
            )

        save_checkpoint(
            out_dir / "last",
            epoch,
            model,
            processor,
            optimizer,
            scheduler,
            clear_dir=True,
            base_vocab_size=base_vocab_size,
            token_ids_to_save=gaze_token_ids,
        )

    if bool(args.eval_only) and len(val_ds) > 0:
        val_gen_metrics = None
        if val_metric_loader is not None:
            val_gen_metrics = run_test_metrics(
                model=model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                show_tqdm=True,
                desc="Eval metrics (checkpoint)",
                max_new_tokens=_gen_max_tokens_eval,
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
                constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
                constrained_target_order=_eval_target_order,
                constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
                constrained_loc_decoding=str(getattr(args, "constrained_loc_decoding", "argmax")),
                max_reasoning_tokens=int(getattr(args, "max_reasoning_tokens", 80)),
                include_l2_breakdown=False,
            )
        if isinstance(val_gen_metrics, dict):
            msg = (
                f"[EVAL] val_dist={float(val_gen_metrics.get('Dist', 0.0)):.6f} "
                f"val_object_acc={float(val_gen_metrics.get('ObjectAcc', 0.0)):.6f} "
                f"val_format_valid={float(val_gen_metrics.get('FormatValid', 0.0)):.6f}"
            )
            if "DistExpected" in val_gen_metrics:
                msg += f" val_dist_expected={float(val_gen_metrics.get('DistExpected', 0.0)):.6f}"
            print(msg)
        else:
            print("[EVAL] no validation generation metrics were produced.")
        if wandb_run is not None:
            payload: dict[str, float] = {}
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics))
            if payload:
                wandb_run.log(payload, step=global_step)

    elapsed = time.time() - start_time

    if True:  # run_test always enabled
        best_dir = out_dir / "best"
        if best_dir.exists():
            loaded_best = load_checkpoint_for_eval(
                ckpt_dir=best_dir,
                model=model,
                device=device,
            )
            if loaded_best:
                print(f"[INFO] loaded best checkpoint for test: {best_dir}")
            else:
                print(f"[WARN] best checkpoint exists but could not be loaded fully: {best_dir}")
        else:
            print("[WARN] best checkpoint directory not found; testing current in-memory model.")

        test_groups = load_test_groups(
            annotation_file=test_ann,
            image_root=test_image_root,
            test_label_map=test_label_map,
            test_label_text_map=test_label_text_map,
            test_label_ids_map=test_label_ids_map,
            split_prefix=args.test_split_prefix,
            strip_split_prefix=bool(args.test_strip_split_prefix),
            bbox_round_decimals=int(args.test_bbox_round_decimals),
        )
        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(f"[TEST] groups={len(test_groups)}")
            test_ds = GazeTestDataset(
                groups=test_groups,
                prompt_template=args.prompt_template,
                prompt_text=_prompt_text_eval,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                visual_prompting=bool(args.visual_prompting),
                image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
                coord_bins=coord_bins,
                force_reasoning_format=_force_eval,
                target_order=_eval_target_order,
            )
            log_target_example("test", test_ds)
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=_nw,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
                persistent_workers=_persistent,
                prefetch_factor=_prefetch,
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                show_tqdm=True,
                desc="Test",
                max_new_tokens=_gen_max_tokens_eval,
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
                constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
                constrained_target_order=_eval_target_order,
                constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
                constrained_loc_decoding=str(getattr(args, "constrained_loc_decoding", "argmax")),
                max_reasoning_tokens=int(getattr(args, "max_reasoning_tokens", 80)),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(test_log_payload(test_metrics), step=global_step)
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            maybe_save_generation_preview(
                args=args,
                out_dir=out_dir,
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                max_new_tokens=_gen_max_tokens_eval,
                constrained_target_order=_eval_target_order,
            )

    finish_wandb(wandb_run)
    print(
        f"[DONE] global_step={global_step} "
        f"best_{checkpoint_monitor}={best_monitor_value:.6f} elapsed_sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
