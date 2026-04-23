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
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .datasets import GazeDataset, GazeTestDataset
from .model import QwenTextGenerationModel
from .utils.checkpoint import load_added_token_rows, load_checkpoint_for_eval, save_checkpoint
from .utils.common import to_device
from .utils.config_parser import (
    build_parser,
    load_yaml_config,
    normalize_run_name,
    output_dir_from_run_name,
)
from .utils.data_utils import (
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
    print_generation_samples,
    print_test_metrics_table,
    run_eval,
    run_test_metrics,
)
from .utils.gaze_tokens import parse_structured_output_text, register_gaze_special_tokens
from .utils.loss_utils import compute_answer_loss
from .utils.processor_collate import (
    QwenRLCollator,
    QwenTestCollator,
    QwenTrainCollator,
    build_answer_mask,
    build_train_inputs,
)
from .utils.rl_utils import (
    AdaptiveKLController,
    FixedKLController,
    build_kl_controller,
    compute_policy_loss_per_token,
    compute_token_logprobs,
    compute_token_logprobs_sum,
    compute_total_reward,
    group_normalize_advantages,
)
from .utils.wandb_utils import finish_wandb, init_wandb, train_progress_bucket


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def resolve_model_source(path: str, *, default_namespace: str = "Qwen") -> str:
    raw = str(path).strip()
    if not raw:
        raise ValueError("Model path must not be empty.")

    local_path = resolve_path(raw)
    if local_path.exists():
        return str(local_path)

    if raw.startswith("/"):
        model_name = Path(raw).name
        inferred_repo_id = f"{default_namespace}/{model_name}"
        print(
            f"[INFO] local model path not found: {raw}. "
            f"Falling back to Hugging Face repo id: {inferred_repo_id}"
        )
        return inferred_repo_id

    path_like_prefixes = ("./", "../", "model/", "models/", "checkpoints/", "data/")
    if raw.startswith(path_like_prefixes):
        model_name = Path(raw).name
        inferred_repo_id = f"{default_namespace}/{model_name}"
        print(
            f"[INFO] local model path not found: {local_path}. "
            f"Falling back to Hugging Face repo id: {inferred_repo_id}"
        )
        return inferred_repo_id

    if raw.count("/") == 1:
        return raw

    model_name = Path(raw).name
    inferred_repo_id = f"{default_namespace}/{model_name}"
    print(
        f"[INFO] treating missing local model path as Hugging Face repo id: "
        f"{inferred_repo_id}"
    )
    return inferred_repo_id


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_dtype(dtype: str) -> torch.dtype | str:
    v = str(dtype).strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def to_autocast_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if dtype == torch.float16:
        return torch.float16
    if dtype == torch.float32:
        return torch.float32
    return torch.bfloat16


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


def env_flag(name: str) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


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


def init_processor(
    *,
    model_path: str,
    checkpoint_dir: Path | None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Any:
    processor_path: str | Path = model_path
    if checkpoint_dir is not None and (checkpoint_dir / "processor").exists():
        processor_path = checkpoint_dir / "processor"
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if min_pixels is not None:
        kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        kwargs["max_pixels"] = int(max_pixels)
    return AutoProcessor.from_pretrained(str(processor_path), **kwargs)


def init_base_model(
    *,
    model_path: str,
    model_kwargs: dict[str, Any],
) -> Any:
    return AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)


def enable_new_token_gradients(peft_model: Any, base_vocab_size: int) -> None:
    """Unfreeze only the newly added embedding rows for gradient updates.

    get_peft_model freezes all base-model parameters including embed_tokens.
    New token rows (indices base_vocab_size..) must be trainable so their
    embeddings are learned rather than staying at random initialization.

    A backward hook zeroes gradients for the original rows so only the new
    rows receive optimizer updates. Because embed_tokens and lm_head share the
    same weight tensor (tied embeddings), one hook covers both.
    """
    emb = peft_model.get_input_embeddings()
    n_base = int(base_vocab_size)
    emb.weight.requires_grad_(True)

    def _zero_base_rows(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[:n_base].zero_()
        return grad

    emb.weight.register_hook(_zero_base_rows)



def test_log_payload(test_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "test/FormatValid": float(test_metrics.get("FormatValid", 0.0)),
        "test/Avg_L2": float(test_metrics.get("Avg L2", 0.0)),
        "test/Min_L2": float(test_metrics.get("Min L2", 0.0)),
        "test/PointBinExact": float(test_metrics.get("PointBinExact", 0.0)),
        "test/ObjectAcc": float(test_metrics.get("ObjectAcc", 0.0)),
        "test/MultiAcc@1": float(test_metrics.get("MultiAcc@1", 0.0)),
        "test/JointExact": float(test_metrics.get("JointExact", 0.0)),
        "test/ExtraTextRate": float(test_metrics.get("ExtraTextRate", 0.0)),
        "test/num_samples": float(test_metrics.get("num_samples", 0.0)),
    }


def val_metric_log_payload(val_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "val/dist": float(val_metrics.get("Avg L2", 0.0)),
        "val/object_acc": float(val_metrics.get("ObjectAcc", 0.0)),
        "val/joint_exact": float(val_metrics.get("JointExact", 0.0)),
        "val/format_valid": float(val_metrics.get("FormatValid", 0.0)),
        "val/extra_text_rate": float(val_metrics.get("ExtraTextRate", 0.0)),
        "val/point_l2_valid_frac": float(val_metrics.get("PointL2ValidFrac", 0.0)),
    }


def maybe_save_generation_preview(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    num_classes: int,
) -> None:
    preview_n = max(0, int(getattr(args, "preview_test_samples", 0)))
    if preview_n <= 0:
        return

    preview_samples = collect_generation_samples(
        model=model,
        loader=loader,
        device=device,
        amp_dtype=amp_dtype,
        processor=processor,
        num_classes=int(num_classes),
        show_tqdm=bool(args.show_tqdm),
        desc="Test preview",
        max_new_tokens=int(args.generation_max_new_tokens),
        num_beams=int(getattr(args, "generation_num_beams", 1)),
        repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
        max_samples=preview_n,
        stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
    )
    preview_path = out_dir / "test_generation_preview.json"
    preview_path.write_text(
        json.dumps(preview_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[TEST] saved generation preview: {preview_path} (samples={len(preview_samples)})")
    print_generation_samples(preview_samples)


def infer_checkpoint_monitor_mode(monitor: str, mode: str) -> str:
    mode_norm = str(mode or "auto").strip().lower()
    if mode_norm in {"min", "max"}:
        return mode_norm
    if mode_norm != "auto":
        raise ValueError(f"checkpoint_monitor_mode must be one of auto|min|max, got: {mode}")
    monitor_norm = str(monitor or "").strip().lower()
    return "min" if (
        "loss" in monitor_norm or "dist" in monitor_norm
        or "l2" in monitor_norm or "extra" in monitor_norm
    ) else "max"


def checkpoint_monitor_value(
    monitor: str,
    *,
    val_loss: float,
    val_gen_metrics: dict[str, float] | None,
) -> float | None:
    monitor_norm = str(monitor or "val_loss").strip().lower().replace("/", "_").replace("-", "_")
    if monitor_norm in {"val_loss", "loss"}:
        return float(val_loss)
    if val_gen_metrics is None:
        return None
    key_map: dict[str, str] = {
        "val_dist": "Avg L2",
        "dist": "Avg L2",
        "val_avg_l2": "Avg L2",
        "avg_l2": "Avg L2",
        "val_point_l2": "Avg L2",
        "point_l2": "Avg L2",
        "val_min_l2": "Min L2",
        "min_l2": "Min L2",
        "val_object_acc": "ObjectAcc",
        "object_acc": "ObjectAcc",
        "val_joint_exact": "JointExact",
        "joint_exact": "JointExact",
        "val_format_valid": "FormatValid",
        "format_valid": "FormatValid",
        "val_extra_text_rate": "ExtraTextRate",
        "extra_text_rate": "ExtraTextRate",
    }
    metric_key = key_map.get(monitor_norm)
    if metric_key is None:
        raise ValueError(
            "Unsupported checkpoint_monitor. Use one of: "
            "val_loss, val_dist, val_object_acc, val_joint_exact, val_format_valid."
        )
    return float(val_gen_metrics.get(metric_key, 0.0))


def _infer_logprobs_chunked(
    model: torch.nn.Module,
    lp_joint: dict[str, Any],
    input_ids: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    micro_bsz: int,
) -> torch.Tensor:
    """Inference-mode per-token log-probs, processed in micro-batches.

    Splits the B*G batch into chunks of `micro_bsz` so that only one chunk's
    activations + logits live on GPU at a time.  Returns [B*G, L-1] on CPU.

    pixel_values in Qwen-VL is a flat [total_patches, C, h, w] tensor — not
    [B, ...].  We use image_grid_thw (num_images, 3) to slice it per sample.
    When image_grid_thw is unavailable we fall back to equal-size splitting.
    """
    total = int(input_ids.shape[0])
    if micro_bsz <= 0 or micro_bsz >= total:
        # No chunking: process full batch at once
        joint_dev = to_device(lp_joint, device=device)
        ids_dev = input_ids.to(device=device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
                out = model(joint_inputs=joint_dev, use_cache=False)
        lp = compute_token_logprobs(out["logits"].detach(), ids_dev).cpu()
        del out
        return lp

    # --- Build per-sample patch counts from image_grid_thw ---
    grid_thw = lp_joint.get("image_grid_thw", None)
    if torch.is_tensor(grid_thw) and int(grid_thw.shape[0]) == total:
        # Each row is (T, H, W); patches_i = T * H * W
        patches_per_sample = [
            int(grid_thw[i, 0]) * int(grid_thw[i, 1]) * int(grid_thw[i, 2])
            for i in range(total)
        ]
    else:
        # Fallback: assume equal patch counts
        pv = lp_joint.get("pixel_values", None)
        total_patches = int(pv.shape[0]) if torch.is_tensor(pv) else 0
        base = total_patches // total if total > 0 else 0
        patches_per_sample = [base] * total

    # --- Slice helper: extract keys for sample indices [start, end) ---
    def _slice_joint(start: int, end: int) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        patch_start = sum(patches_per_sample[:start])
        patch_end   = sum(patches_per_sample[:end])
        for k, v in lp_joint.items():
            if not torch.is_tensor(v):
                sliced[k] = v
            elif k == "pixel_values":
                sliced[k] = v[patch_start:patch_end]
            elif k == "image_grid_thw":
                sliced[k] = v[start:end]
            elif v.shape[0] == total:
                sliced[k] = v[start:end]
            else:
                sliced[k] = v
        return sliced

    # --- Chunked forward ---
    chunks: list[torch.Tensor] = []
    for start in range(0, total, micro_bsz):
        end = min(start + micro_bsz, total)
        chunk_joint = to_device(_slice_joint(start, end), device=device)
        chunk_ids   = input_ids[start:end].to(device=device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
                out = model(joint_inputs=chunk_joint, use_cache=False)
        chunks.append(compute_token_logprobs(out["logits"].detach(), chunk_ids).cpu())
        del out, chunk_joint
    return torch.cat(chunks, dim=0)   # [B*G, L-1]


def _run_rl_training(
    *,
    args: argparse.Namespace,
    policy_model: torch.nn.Module,
    processor: Any,
    train_ds: Any,
    val_loader: DataLoader,
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
    loss_weights: dict[str, float],
    wandb_run: Any,
    scene_size: tuple[int, int] | None,
) -> tuple[int, float]:
    """GRPO-based RL post-training (Stage 2).

    Assumes policy_model is already loaded from an SFT checkpoint with trainable
    LoRA + new token embeddings.  A frozen copy of the same checkpoint is loaded
    as the reference model for the KL penalty term.

    Returns (global_step, best_val_loss).
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
    best_val_loss = float("inf")
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
            disable=not bool(getattr(args, "show_tqdm", True)),
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
                parsed = parse_structured_output_text(preds[k], int(num_classes))
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
            ref_log_probs = _infer_logprobs_chunked(
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
            old_log_probs = _infer_logprobs_chunked(
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

            if bool(getattr(args, "show_tqdm", True)):
                n_r = max(rollout_count, 1)
                it.set_postfix(
                    rwd=f"{(sum_reward / n_r):.3f}",
                    kl=f"{(sum_kl / max(update_count, 1)):.4f}",
                    kl_c=f"{kl_ctrl.kl_coef:.4f}",
                )

        # -------------------------------------------------------
        # End of RL epoch: validation
        # -------------------------------------------------------
        # Restore train() mode before validation (run_eval uses train/eval internally).
        policy_model.train()
        n_roll = max(rollout_count, 1)
        n_upd = max(update_count, 1)
        epoch_msg = (
            f"[RL EPOCH {epoch}] reward={sum_reward / n_roll:.4f} "
            f"fmt_invalid={sum_invalid_fmt / n_roll:.3f} "
            f"kl={sum_kl / n_upd:.4f} pg_loss={sum_pg_loss / n_upd:.4f}"
        )

        val_metrics = (
            run_eval(
                policy_model, val_loader, device, amp_dtype,
                loss_weights=loss_weights,
                show_tqdm=bool(getattr(args, "show_tqdm", True)),
                desc=f"RL Val {epoch}",
            )
            if len(val_loader) > 0
            else {"loss": 0.0}
        )
        val_loss = float(val_metrics.get("loss", 0.0))

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
                show_tqdm=bool(getattr(args, "show_tqdm", True)),
                desc=f"RL ValMetric {epoch}",
                max_new_tokens=rl_max_new_tokens,
                num_beams=1,
                repetition_penalty=1.0,
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
            )
            epoch_msg += (
                f" val_dist={float(val_gen_metrics.get('Avg L2', 0.0)):.6f}"
                f" val_fmt={float(val_gen_metrics.get('FormatValid', 0.0)):.4f}"
                f" val_obj={float(val_gen_metrics.get('ObjectAcc', 0.0)):.4f}"
            )
        print(epoch_msg)

        if wandb_run is not None:
            _wlog: dict[str, float] = {
                "val/loss": float(val_loss),
                "rl/epoch_reward_mean": sum_reward / n_roll,
                "rl/epoch_invalid_fmt": sum_invalid_fmt / n_roll,
            }
            if isinstance(val_gen_metrics, dict):
                _wlog.update(val_metric_log_payload(val_gen_metrics))
            wandb_run.log(_wlog, step=global_step)

        # Monitor-based checkpoint save
        monitor_value = checkpoint_monitor_value(
            checkpoint_monitor,
            val_loss=val_loss,
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
            best_val_loss = val_loss
            save_checkpoint(
                out_dir / "best", epoch, policy_model, processor,
                optimizer, scheduler, clear_dir=True,
                base_vocab_size=base_vocab_size,
            )
            print(f"[RL] new best checkpoint: {checkpoint_monitor}={best_monitor_value:.6f}")

        save_checkpoint(
            out_dir / "last", epoch, policy_model, processor,
            optimizer, scheduler, clear_dir=True,
            base_vocab_size=base_vocab_size,
        )

    elapsed = time.time() - start_time
    print(
        f"[RL DONE] global_step={global_step} best_val_loss={best_val_loss:.6f} "
        f"elapsed_sec={elapsed:.1f}"
    )
    return global_step, best_val_loss


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config.yaml")
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

    prompt_text_for_run = str(args.prompt_text or "")
    filter_invalid = bool(getattr(args, "filter_invalid_object_samples", True))
    loss_weights = {
        "point": float(getattr(args, "loss_point_weight", 1.0)),
        "object": float(getattr(args, "loss_object_weight", 1.0)),
        "format": float(getattr(args, "loss_format_weight", 0.25)),
    }

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
    )
    new_vocab_size = len(processor.tokenizer)
    print(
        f"[INFO] tokenizer extended: vocab_size={new_vocab_size} "
        f"(added loc tokens: 1000, obj tokens: {num_classes}, fmt tokens: 0)"
    )

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
                rows_loaded = load_added_token_rows(
                    ckpt_dir=checkpoint_dir,
                    model=QwenTextGenerationModel(qwen_model=qwen_model),
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
                prompt_text=prompt_text_for_run,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                visual_prompting=bool(args.visual_prompting),
                image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
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
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
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
    train_ds = GazeDataset(
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
    )
    val_ds = GazeDataset(
        records=val_records,
        prompt_template=args.prompt_template,
        prompt_text=prompt_text_for_run,
        apply_augmentation=False,
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        visual_prompting=bool(args.visual_prompting),
        image_cache_size=image_cache_size,
        filter_invalid_object_samples=filter_invalid,
    )

    # Count filtered samples
    n_train_valid = sum(
        1 for i in range(len(train_ds)) if float(train_ds[i]["target_text_valid"].item()) > 0.0
    ) if len(train_ds) <= 1000 else -1
    print(
        f"[INFO] structured pipeline: filter_invalid_object_samples={filter_invalid} "
        f"train_valid_structured={n_train_valid if n_train_valid >= 0 else 'not_counted'}"
    )
    log_target_example("train", train_ds)
    log_target_example("val", val_ds)

    train_collator = QwenTrainCollator(
        processor=processor,
        max_text_length=int(args.max_text_length),
        scene_size=_scene_size,
    )
    val_collator = QwenTrainCollator(
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
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=val_collator,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    val_metric_loader = None
    if bool(getattr(args, "run_val_metrics", True)):
        val_metric_loader = DataLoader(
            val_ds,
            batch_size=max(1, int(args.test_batch_size)),
            shuffle=False,
            num_workers=_nw,
            pin_memory=(device.type == "cuda"),
            collate_fn=test_collator,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
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
        )
        qwen_lora = get_peft_model(base_qwen, lora_cfg)
        qwen_lora.print_trainable_parameters()

    # Unfreeze only the new token embedding rows added by resize_token_embeddings.
    # get_peft_model freezes all base-model params; without this the gaze special
    # token embeddings (<loc_XXX>, <obj_XXX>, etc.) would remain at their
    # random-initialisation values throughout training and never be updated.
    if not bool(getattr(args, "eval_only", False)):
        enable_new_token_gradients(qwen_lora, base_vocab_size)
        print(
            f"[INFO] enabled gradient for new token embedding rows "
            f"[{base_vocab_size}:{new_vocab_size}] ({new_vocab_size - base_vocab_size} rows)"
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
        rl_global_step, rl_best_val_loss = _run_rl_training(
            args=args,
            policy_model=model,
            processor=processor,
            train_ds=train_ds,
            val_loader=val_loader,
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
            loss_weights=loss_weights,
            wandb_run=wandb_run,
            scene_size=_scene_size,
        )

        if args.run_test:
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
                    prompt_text=prompt_text_for_run,
                    id2label=id2label,
                    vocab2id=vocab2id,
                    vocab2id_lower=vocab2id_lower,
                    num_classes=int(num_classes),
                    visual_prompting=bool(args.visual_prompting),
                    image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
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
                    show_tqdm=bool(args.show_tqdm),
                    desc="Test",
                    max_new_tokens=int(args.generation_max_new_tokens),
                    num_beams=int(getattr(args, "generation_num_beams", 1)),
                    repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                    no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                    stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
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
                )

        finish_wandb(wandb_run)
        print(
            f"[DONE] RL stage global_step={rl_global_step} "
            f"best_val_loss={rl_best_val_loss:.6f} "
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
        f"object_w={loss_weights['object']:.2f} format_w={loss_weights['format']:.2f}"
    )
    checkpoint_monitor = str(getattr(args, "checkpoint_monitor", "val_dist")).strip() or "val_dist"
    checkpoint_monitor_mode = infer_checkpoint_monitor_mode(
        checkpoint_monitor,
        str(getattr(args, "checkpoint_monitor_mode", "auto")),
    )
    best_monitor_value = float("inf") if checkpoint_monitor_mode == "min" else -float("inf")
    best_val_loss = float("inf")
    print(f"[INFO] checkpoint monitor: {checkpoint_monitor} ({checkpoint_monitor_mode})")
    run_val_metrics_every_n = max(1, int(getattr(args, "run_val_metrics_every_n_epochs", 5)))
    global_step = 0
    start_time = time.time()

    effective_epochs = 0 if bool(args.eval_only) else int(args.epochs)
    if bool(args.eval_only):
        print("[INFO] eval_only=True; skipping training loop.")

    for epoch in range(1, effective_epochs + 1):
        model.train()
        sum_loss = 0.0
        sample_count = 0
        step_count = 0
        updates_done_in_epoch = 0
        last_logged_train_bucket = 0
        skipped_all_ignore = 0

        optimizer.zero_grad(set_to_none=True)
        train_iter = tqdm(
            train_loader,
            desc=f"Train {epoch}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=not args.show_tqdm,
        )

        remainder_steps = num_train_batches % accum_steps
        last_window_start = (
            (num_train_batches - int(remainder_steps) + 1)
            if int(remainder_steps) > 0
            else (num_train_batches + 1)
        )

        for step, batch in enumerate(train_iter, start=1):
            labels = batch["labels"].to(device)
            if torch.all(labels.eq(-100)):
                skipped_all_ignore += 1
                continue

            joint_inputs = to_device(batch["joint_inputs"], device=device)
            bsz = int(labels.shape[0])
            is_last_batch = (step == num_train_batches)
            current_accum_steps = (
                int(remainder_steps)
                if (int(remainder_steps) > 0 and step >= int(last_window_start))
                else int(accum_steps)
            )

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
                )
                raw_loss = losses["loss"]
                loss = raw_loss / float(max(current_accum_steps, 1))

            loss.backward()

            should_step = ((step % accum_steps) == 0) or is_last_batch
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                updates_done_in_epoch += 1

                if wandb_run is not None:
                    current_train_bucket = train_progress_bucket(
                        updates_done_in_epoch,
                        updates_per_epoch,
                    )
                    should_log_train = current_train_bucket > last_logged_train_bucket
                    if should_log_train:
                        last_logged_train_bucket = current_train_bucket
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
                                "train/FormatValidRate": float(losses["format_valid_rate"].detach().item()),
                                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "train/grad_norm": grad_norm_value,
                                "train/epoch": epoch_progress,
                            },
                            step=global_step,
                        )

            sum_loss += float(raw_loss.detach().item()) * float(bsz)
            sample_count += bsz
            step_count += 1
            if args.show_tqdm:
                train_iter.set_postfix(loss=f"{(sum_loss / max(sample_count, 1)):.4f}")

        if step_count == 0 or sample_count == 0:
            raise RuntimeError(
                "No effective training batches were produced. "
                "All batches may have empty target text (all labels ignored)."
            )

        train_loss = float(sum_loss / float(sample_count))
        if skipped_all_ignore > 0:
            print(f"[INFO] skipped all-ignore batches in train epoch: {skipped_all_ignore}")

        val_metrics = (
            run_eval(
                model,
                val_loader,
                device,
                amp_dtype,
                loss_weights=loss_weights,
                show_tqdm=bool(args.show_tqdm),
                desc=f"Eval {epoch}/{args.epochs}",
            )
            if len(val_ds) > 0
            else {"loss": train_loss}
        )
        val_loss = float(val_metrics.get("loss", train_loss))

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
                show_tqdm=bool(args.show_tqdm),
                desc=f"ValMetric {epoch}/{args.epochs}",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
            )

        epoch_msg = (
            f"[EPOCH {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if isinstance(val_gen_metrics, dict):
            epoch_msg += (
                f" val_dist={float(val_gen_metrics.get('Avg L2', 0.0)):.6f}"
                f" val_object_acc={float(val_gen_metrics.get('ObjectAcc', 0.0)):.6f}"
                f" val_format_valid={float(val_gen_metrics.get('FormatValid', 0.0)):.6f}"
                f" val_l2_cov={float(val_gen_metrics.get('PointL2ValidFrac', 0.0)):.3f}"
            )
        print(epoch_msg)

        if wandb_run is not None:
            payload: dict[str, float] = {
                "val/loss": float(val_loss),
                "val/loss_point": float(val_metrics.get("loss_point", 0.0)),
                "val/loss_object": float(val_metrics.get("loss_object", 0.0)),
                "val/loss_format": float(val_metrics.get("loss_format", 0.0)),
            }
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics))
            wandb_run.log(payload, step=global_step)

        monitor_value = checkpoint_monitor_value(
            checkpoint_monitor,
            val_loss=val_loss,
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
            best_val_loss = val_loss
            best_dir = out_dir / "best"
            print(
                f"[INFO] new best checkpoint: {checkpoint_monitor}={best_monitor_value:.6f} "
                f"val_loss={val_loss:.6f}"
            )
            save_checkpoint(
                best_dir,
                epoch,
                model,
                processor,
                optimizer,
                scheduler,
                clear_dir=True,
                base_vocab_size=base_vocab_size,
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
        )

    if bool(args.eval_only) and len(val_ds) > 0:
        val_metrics = run_eval(
            model,
            val_loader,
            device,
            amp_dtype,
            loss_weights=loss_weights,
            show_tqdm=bool(args.show_tqdm),
            desc="Eval (checkpoint)",
        )
        best_val_loss = float(val_metrics.get("loss", best_val_loss))
        val_gen_metrics = None
        if val_metric_loader is not None:
            val_gen_metrics = run_test_metrics(
                model=model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                num_classes=int(num_classes),
                show_tqdm=bool(args.show_tqdm),
                desc="Eval metrics (checkpoint)",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
            )
        print(f"[EVAL] val_loss={best_val_loss:.6f}")
        if wandb_run is not None:
            payload = {"val/loss": float(best_val_loss)}
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics))
            wandb_run.log(payload, step=global_step)

    elapsed = time.time() - start_time

    if args.run_test:
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
                prompt_text=prompt_text_for_run,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                visual_prompting=bool(args.visual_prompting),
                image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
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
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
                stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
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
            )

    finish_wandb(wandb_run)
    print(
        f"[DONE] global_step={global_step} best_val_loss={best_val_loss:.6f} "
        f"best_{checkpoint_monitor}={best_monitor_value:.6f} elapsed_sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
