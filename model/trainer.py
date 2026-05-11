from __future__ import annotations

import argparse
import json
import math
import os
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
from transformers import get_cosine_schedule_with_warmup

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .datasets import GazeDataset, GazeTestDataset
from .model import QwenTextGenerationModel
from .utils.checkpoint import (
    checkpoint_monitor_value,
    infer_checkpoint_monitor_mode,
    load_added_token_rows,
    load_checkpoint_for_eval,
    load_token_rows,
    save_checkpoint,
)
from .utils.common import env_flag, parse_dtype, resolve_path, set_seed, to_autocast_dtype
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
    maybe_save_generation_preview,
    print_test_metrics_table,
    run_test_metrics,
)
from .utils.special_tokens import (
    GAZE_SCHEMA_MARKERS,
    _obj_token_width,
    format_loc_token,
    format_obj_token,
    register_gaze_special_tokens,
)
from .utils.processor_collate import (
    QwenTestCollator,
    QwenTrainCollator,
)
from .utils.wandb_utils import finish_wandb, init_wandb, test_log_payload, val_metric_log_payload
from .rl_trainer import run_rl_training
from .sdft_trainer import EMATeacher, run_sdft_epoch
from .sft_trainer import run_sft_epoch


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
    if train_stage not in {"sft", "sdft", "rl"}:
        raise ValueError(f"train_stage must be 'sft', 'sdft', or 'rl', got: {train_stage!r}")
    if train_stage == "sdft" and (not _inference_only) and checkpoint_dir is None:
        raise ValueError(
            "train_stage='sdft' requires --checkpoint_dir pointing to the Stage1 "
            "best-val_dist checkpoint (must contain lora_adapter/)."
        )

    # prompt_text_direct falls back to prompt_text when absent.
    _prompt_fallback = str(getattr(args, "prompt_text", "") or "")
    _prompt_text_direct = str(getattr(args, "prompt_text_direct", "") or "") or _prompt_fallback
    _prompt_text_teacher = str(getattr(args, "prompt_text_teacher", "") or "")
    prompt_text_for_run = _prompt_text_direct
    _prompt_text_eval = _prompt_text_direct
    _eval_target_order = "point_object"
    _gen_max_tokens = int(getattr(args, "generation_max_new_tokens", 8))
    _gen_max_tokens_eval = _gen_max_tokens

    filter_invalid = bool(getattr(args, "filter_invalid_object_samples", True))
    loss_weights = {
        "point": float(getattr(args, "loss_point_weight", 1.0)),
        "object": float(getattr(args, "loss_object_weight", 1.0)),
        "format": float(getattr(args, "loss_format_weight", 0.25)),
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
    _ow = _obj_token_width(num_classes)
    _obj_id_list = [int(token_id_map.get(format_obj_token(i, _ow), -1)) for i in range(num_classes)]
    object_token_ids_tensor: torch.Tensor | None = (
        torch.tensor(_obj_id_list, dtype=torch.long)
        if all(i >= 0 for i in _obj_id_list) else None
    )
    object_token_ids_for_loss: torch.Tensor | None = (
        object_token_ids_tensor.to(device=device)
        if object_token_ids_tensor is not None else None
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
    _aug_modes = {"full", "crop_flip_color", "default", "color", "color_only", "photometric", "safe", "no_crop", "flip_color", "hflip_color", "crop_only", "safe_crop", "none", "no_aug", "off", "false"}
    train_augmentation_mode_direct = str(getattr(args, "train_augmentation_mode_direct", "full") or "full").strip().lower()
    if train_augmentation_mode_direct not in _aug_modes:
        raise ValueError(
            f"unsupported train_augmentation_mode_direct={train_augmentation_mode_direct!r}; "
            "expected one of: full, color_only, no_crop, crop_only, no_aug"
        )

    distil_kl_weight = float(getattr(args, "distil_kl_weight", 0.0))
    reasoning_index = None
    if train_stage == "sdft" and (not _inference_only) and distil_kl_weight > 0.0:
        reasoning_dir_raw = str(getattr(args, "train_reasoning_dir", "") or "").strip()
        if reasoning_dir_raw:
            reasoning_dir = resolve_path(reasoning_dir_raw)
            if reasoning_dir.exists():
                reasoning_index = build_reasoning_index(reasoning_dir)
                print(f"[INFO] SDFT teacher demonstrations: indexed {len(reasoning_index)} files from {reasoning_dir}")
            else:
                print(f"[WARN] train_reasoning_dir not found; SDFT teacher demos disabled: {reasoning_dir}")
        else:
            print("[WARN] train_reasoning_dir is empty; SDFT teacher demos disabled.")

    train_ds = GazeDataset(
        records=train_records,
        prompt_template=args.prompt_template,
        prompt_text=prompt_text_for_run,
        prompt_text_teacher=_prompt_text_teacher,
        apply_augmentation=True,
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        visual_prompting=bool(args.visual_prompting),
        image_cache_size=image_cache_size,
        filter_invalid_object_samples=filter_invalid,
        coord_bins=coord_bins,
        train_augmentation_mode=train_augmentation_mode_direct,
        target_order="point_object",
        reasoning_index=reasoning_index,
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
        target_order=_eval_target_order,
    )

    # Count filtered samples
    n_train_valid = sum(
        1 for i in range(len(train_ds)) if float(train_ds[i]["target_text_valid"].item()) > 0.0
    ) if len(train_ds) <= 1000 else -1
    print(
        f"[INFO] structured pipeline: filter_invalid_object_samples={filter_invalid} "
        f"train_augmentation_mode_direct={train_augmentation_mode_direct} "
        f"train_valid_structured={n_train_valid if n_train_valid >= 0 else 'not_counted'}"
    )
    log_target_example("train", train_ds)
    log_target_example("val", val_ds)

    distil_temperature = float(getattr(args, "distil_temperature", 1.0))
    distil_teacher_eval_mode = bool(getattr(args, "distil_teacher_eval_mode", False))
    distil_teacher_suffix = str(getattr(args, "teacher_suffix", "\n\nUse the following reasoning to guide your prediction:\n{reasoning_text}\n\nNow apply the same reasoning process to predict the gaze point and target."))
    # Pre-resolve struct-token range variables so rollout mode can safely call
    # .format(reasoning_text=..., object_text=...) without KeyError.
    _coord_n = int(coord_bins)
    _loc_w = max(3, len(str(_coord_n - 1)))
    _obj_max = max(0, int(num_classes) - 1)
    _obj_w = max(3, len(str(_obj_max)))
    distil_teacher_suffix = distil_teacher_suffix.format(
        loc_tok_min=format_loc_token(0, _loc_w),
        loc_tok_max=format_loc_token(_coord_n - 1, _loc_w),
        obj_tok_min=format_obj_token(0, _obj_w),
        obj_tok_max=format_obj_token(_obj_max, _obj_w),
        reasoning_text="{reasoning_text}",
        object_text="{object_text}",
    )

    # sdft mode config (only relevant when train_stage == "sdft")
    sdft_mode = str(getattr(args, "sdft_mode", "teacher_forcing")).strip().lower()
    sdft_ce_weight = float(getattr(args, "sdft_ce_weight", 0.0))
    rollout_max_new_tokens = int(getattr(args, "rollout_max_new_tokens", 16))
    rollout_do_sample = bool(getattr(args, "rollout_do_sample", False))
    rollout_temperature = float(getattr(args, "rollout_temperature", 1.0))
    rollout_top_p = float(getattr(args, "rollout_top_p", 1.0))
    rollout_constrained_decoding = bool(getattr(args, "rollout_constrained_decoding", False))
    rollout_constrained_loc_decoding = str(
        getattr(args, "rollout_constrained_loc_decoding", "argmax")
    )
    skip_invalid_rollouts = bool(getattr(args, "skip_invalid_rollouts", True))
    skip_truncated_rollouts = bool(getattr(args, "skip_truncated_rollouts", True))
    kl_on_point = bool(getattr(args, "kl_on_point", True))
    kl_on_object = bool(getattr(args, "kl_on_object", True))
    kl_on_format = bool(getattr(args, "kl_on_format", False))

    if train_stage == "sdft" and sdft_mode not in {"teacher_forcing", "rollout"}:
        raise ValueError(
            f"sdft.sdft_mode must be 'teacher_forcing' or 'rollout', got: {sdft_mode!r}"
        )
    if train_stage == "sdft" and sdft_mode == "rollout":
        if kl_on_format:
            raise ValueError("kl_on_format=True is not supported in rollout mode.")
    if train_stage == "sdft":
        print(f"[INFO] SDFT mode: {sdft_mode}")

    train_collator = QwenTrainCollator(
        processor=processor,
        max_text_length=int(args.max_text_length),
        scene_size=_scene_size,
        distil_kl_weight=distil_kl_weight,
        distil_teacher_suffix=distil_teacher_suffix,
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

    teacher_update = str(getattr(args, "teacher_update", "fixed")).strip().lower()
    teacher_ema_decay = float(getattr(args, "teacher_ema_decay", 0.999))
    ema_teacher: EMATeacher | None = None
    if train_stage == "sdft" and teacher_update == "ema" and not _inference_only:
        ema_teacher = EMATeacher(model=model, decay=teacher_ema_decay)
        n_ema_params = len(ema_teacher.ema_params)
        print(f"[INFO] EMA teacher initialised: decay={teacher_ema_decay} tracking {n_ema_params} trainable param tensors")

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
        rl_global_step, rl_best_monitor_value = run_rl_training(
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
    # SFT/SDFT training loop — only reached when train_stage is "sft" or "sdft".
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
    print(f"[INFO] checkpoint monitor: {checkpoint_monitor} ({checkpoint_monitor_mode})")
    run_val_metrics_every_n = max(1, int(getattr(args, "run_val_metrics_every_n_epochs", 5)))
    global_step = 0
    start_time = time.time()

    effective_epochs = 0 if bool(args.eval_only) else int(args.epochs)
    if bool(args.eval_only):
        print("[INFO] eval_only=True; skipping training loop.")

    for epoch in range(1, effective_epochs + 1):
        if train_stage == "sft":
            _sft = run_sft_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                trainable_params=trainable_params,
                device=device,
                amp_dtype=amp_dtype,
                loss_weights=loss_weights,
                loc_token_ids_for_loss=loc_token_ids_for_loss,
                gaussian_point_sigma=gaussian_point_sigma,
                accum_steps=accum_steps,
                max_grad_norm=float(args.max_grad_norm),
                wandb_run=wandb_run,
                epoch=epoch,
                num_epochs=int(args.epochs),
                num_train_batches=num_train_batches,
                updates_per_epoch=updates_per_epoch,
                global_step=global_step,
                train_log_every=max(1, int(getattr(args, "wandb_log_every_steps", 20))),
            )
            global_step = _sft.global_step
            train_loss = _sft.train_loss
            if _sft.skipped_all_ignore > 0:
                print(f"[INFO] skipped all-ignore batches in train epoch: {_sft.skipped_all_ignore}")
            _t_other = max(0.0, _sft.t_step - _sft.t_fwd - _sft.t_bwd)
            print(
                f"[TIME train {epoch}] "
                f"data_wait={_sft.t_data:.1f}s "
                f"fwd_loss={_sft.t_fwd:.1f}s "
                f"backward={_sft.t_bwd:.1f}s "
                f"other_step={_t_other:.1f}s "
                f"steps={_sft.step_count}"
            )
        else:
            _sdft = run_sdft_epoch(
                model=model,
                train_loader=train_loader,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                trainable_params=trainable_params,
                device=device,
                amp_dtype=amp_dtype,
                sdft_mode=sdft_mode,
                loss_weights=loss_weights,
                loc_token_ids_for_loss=loc_token_ids_for_loss,
                object_token_ids_for_loss=object_token_ids_for_loss,
                gaussian_point_sigma=gaussian_point_sigma,
                distil_kl_weight=distil_kl_weight,
                distil_temperature=distil_temperature,
                distil_teacher_eval_mode=distil_teacher_eval_mode,
                distil_teacher_suffix=distil_teacher_suffix,
                ema_teacher=ema_teacher,
                sdft_ce_weight=sdft_ce_weight,
                rollout_max_new_tokens=rollout_max_new_tokens,
                rollout_do_sample=rollout_do_sample,
                rollout_temperature=rollout_temperature,
                rollout_top_p=rollout_top_p,
                rollout_constrained_decoding=rollout_constrained_decoding,
                rollout_constrained_loc_decoding=rollout_constrained_loc_decoding,
                skip_invalid_rollouts=skip_invalid_rollouts,
                skip_truncated_rollouts=skip_truncated_rollouts,
                kl_on_point=kl_on_point,
                kl_on_object=kl_on_object,
                num_classes=int(num_classes),
                coord_bins=coord_bins,
                max_text_length=int(args.max_text_length),
                accum_steps=accum_steps,
                max_grad_norm=float(args.max_grad_norm),
                wandb_run=wandb_run,
                epoch=epoch,
                num_epochs=int(args.epochs),
                num_train_batches=num_train_batches,
                updates_per_epoch=updates_per_epoch,
                global_step=global_step,
                train_log_every=max(1, int(getattr(args, "wandb_log_every_steps", 20))),
            )
            global_step = _sdft.global_step
            train_loss = _sdft.train_loss
            if _sdft.skipped_all_ignore > 0:
                print(f"[INFO] skipped all-ignore batches in train epoch: {_sdft.skipped_all_ignore}")
            _t_other = max(0.0, _sdft.t_step - _sdft.t_fwd - _sdft.t_bwd)
            print(
                f"[TIME train {epoch}] "
                f"data_wait={_sdft.t_data:.1f}s "
                f"fwd_loss={_sdft.t_fwd:.1f}s "
                f"backward={_sdft.t_bwd:.1f}s "
                f"other_step={_t_other:.1f}s "
                f"steps={_sdft.step_count}"
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
                f" val_acc@1={float(val_gen_metrics.get('Acc@1', val_gen_metrics.get('ObjectAcc', 0.0))):.6f}"
                f" val_acc@3={float(val_gen_metrics.get('Acc@3', val_gen_metrics.get('Acc@1', val_gen_metrics.get('ObjectAcc', 0.0)))):.6f}"
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
                ema_teacher=ema_teacher,
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
            ema_teacher=ema_teacher,
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
                include_l2_breakdown=False,
            )
        if isinstance(val_gen_metrics, dict):
            msg = (
                f"[EVAL] val_dist={float(val_gen_metrics.get('Dist', 0.0)):.6f} "
                f"val_acc@1={float(val_gen_metrics.get('Acc@1', val_gen_metrics.get('ObjectAcc', 0.0))):.6f} "
                f"val_acc@3={float(val_gen_metrics.get('Acc@3', val_gen_metrics.get('Acc@1', val_gen_metrics.get('ObjectAcc', 0.0)))):.6f} "
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
