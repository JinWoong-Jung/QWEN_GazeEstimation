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
from .utils.checkpoint import load_checkpoint_for_eval, save_checkpoint
from .utils.config_parser import build_parser, load_yaml_config
from .utils.data_utils import (
    load_label_map,
    load_label_text_map,
    load_records,
    load_test_groups,
    load_test_label_map,
    load_vocab2id,
)
from .utils.eval_utils import print_test_metrics_table, run_eval, run_test_metrics
from .utils.loss_utils import compute_structured_losses
from .utils.processor_collate import QwenTestCollator, QwenTrainCollator
from .utils.wandb_utils import finish_wandb, init_wandb


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


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


def _move_joint_inputs_to_device(joint_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in joint_inputs.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def _build_id2label(vocab2id: dict[str, int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for label_text, idx in vocab2id.items():
        idx_i = int(idx)
        if idx_i not in out:
            out[idx_i] = str(label_text)
    return out


def _count_valid_targets(records: list[Any], id2label: dict[int, str]) -> int:
    n = 0
    for r in records:
        label_id = int(getattr(r, "label_id", -100))
        txt = str(getattr(r, "label_text", "") or "").strip()
        if (label_id >= 0) or bool(txt):
            n += 1
    return n


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

    model_path = resolve_path(args.model_path)
    checkpoint_dir = resolve_path(args.checkpoint_dir) if str(args.checkpoint_dir).strip() else None
    train_ann = resolve_path(args.train_ann)
    val_ann = resolve_path(args.val_ann)
    test_ann = resolve_path(args.test_ann)
    image_root = resolve_path(args.image_root)
    test_image_root = resolve_path(args.test_image_root)
    train_labels = resolve_path(args.train_labels)
    val_labels = resolve_path(args.val_labels)
    test_labels = resolve_path(args.test_labels)
    labels_rgs = resolve_path(args.labels_rgs) if str(getattr(args, "labels_rgs", "")).strip() else None
    labels_ssa = resolve_path(args.labels_ssa) if str(getattr(args, "labels_ssa", "")).strip() else None
    vocab2id_path = resolve_path(args.vocab2id)

    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    id2label = _build_id2label(vocab2id)
    num_classes = (max((int(x) for x in vocab2id.values()), default=-1) + 1) if vocab2id else 0
    if vocab2id:
        print(f"[INFO] loaded vocab2id classes: {len(vocab2id)} (id_range=0..{max(num_classes - 1, 0)})")
    else:
        print("[WARN] vocab2id is missing/empty. target text will rely on csv label text only.")

    prompt_text_for_run = str(args.prompt_text or "")
    if int(num_classes) > 0 and ("ObjectID must be an integer in the closed-set class range" not in prompt_text_for_run):
        if prompt_text_for_run and (not prompt_text_for_run.endswith("\n")):
            prompt_text_for_run += "\n"
        prompt_text_for_run += (
            f"ObjectID must be an integer in the closed-set class range [0, {int(num_classes) - 1}]."
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

        processor_path = model_path
        if checkpoint_dir is not None and (checkpoint_dir / "processor").exists():
            processor_path = checkpoint_dir / "processor"
        processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)

        collator_include_raw_inputs = bool(getattr(args, "collator_include_raw_inputs", False))
        if collator_include_raw_inputs:
            print("[INFO] collator raw input passthrough: enabled (scene_images/text_inputs).")

        test_collator = QwenTestCollator(
            processor=processor,
            scene_size=(int(args.scene_h), int(args.scene_w)),
            max_text_length=int(args.max_text_length),
            include_raw_inputs=collator_include_raw_inputs,
        )

        base_qwen = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)
        adapter_dir = (checkpoint_dir / "lora_adapter") if checkpoint_dir is not None else None
        if adapter_dir is not None and adapter_dir.exists():
            qwen_model = PeftModel.from_pretrained(
                base_qwen,
                model_id=str(adapter_dir),
                is_trainable=False,
            )
            print(f"[INFO] loaded LoRA adapter from: {adapter_dir}")
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
            max_groups=int(args.max_test_samples),
        )
        start_index = max(0, int(getattr(args, "test_start_index", 0)))
        if start_index > 0:
            test_groups = test_groups[start_index:]
        test_num_samples = int(getattr(args, "test_num_samples", 0))
        if test_num_samples > 0:
            test_groups = test_groups[: max(1, test_num_samples)]

        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(
                f"[TEST] groups={len(test_groups)} "
                f"(start={start_index}, limit={(test_num_samples if test_num_samples > 0 else 'all')})"
            )
            test_ds = GazeTestDataset(
                groups=test_groups,
                prompt_template=args.prompt_template,
                prompt_text=prompt_text_for_run,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                answer_template=args.answer_template,
                fallback_target_text=args.fallback_target_text,
                fallback_object_id=int(args.fallback_object_id),
                point_decimals=int(args.point_decimals),
                visual_prompting=bool(args.visual_prompting),
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "test/epoch": 0.0,
                        "test/ExactMatch": float(test_metrics["ExactMatch"]),
                        "test/Contains": float(test_metrics["Contains"]),
                        "test/AvgL2": float(test_metrics.get("Avg L2", test_metrics.get("PointL2", 0.0))),
                        "test/MinL2": float(test_metrics.get("Min L2", 0.0)),
                        "test/PointL2": float(test_metrics.get("PointL2", 0.0)),
                        "test/acc@1": float(test_metrics.get("acc@1", 0.0)),
                        "test/acc@3": float(test_metrics.get("acc@3", 0.0)),
                        "test/multiacc@1": float(test_metrics.get("multiacc@1", 0.0)),
                        "test/ObjectIDValidRate": float(test_metrics.get("ObjectIDValidRate", 0.0)),
                        "test/num_samples": float(test_metrics["num_samples"]),
                        "test/num_valid_targets": float(test_metrics["num_valid_targets"]),
                    },
                    step=0,
                )
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
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
        fallback_csvs=[labels_rgs, labels_ssa],
        fallback_text_key="annotation",
        split_name="train",
    )
    val_label_map, val_label_stats = load_label_map(
        val_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        fallback_csvs=[labels_rgs, labels_ssa],
        fallback_text_key="annotation",
        split_name="validation",
    )

    train_label_text_map, train_label_text_stats = load_label_text_map(
        train_labels,
        text_key="gaze_pseudo_label",
    )
    val_label_text_map, val_label_text_stats = load_label_text_map(
        val_labels,
        text_key="gaze_pseudo_label",
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
        f"fallback_mapped={train_label_stats.get('fallback_mapped', 0)}"
    )
    print(
        "[INFO] val label id coverage: "
        f"rows={val_label_stats['rows']} mapped={val_label_stats['mapped']} "
        f"missing_text={val_label_stats['missing_text']} unknown_text={val_label_stats['unknown_text']} "
        f"fallback_mapped={val_label_stats.get('fallback_mapped', 0)}"
    )
    print(
        "[INFO] train label text coverage: "
        f"rows={train_label_text_stats['rows']} with_text={train_label_text_stats['with_text']} "
        f"missing_text={train_label_text_stats['missing_text']}"
    )
    print(
        "[INFO] val label text coverage: "
        f"rows={val_label_text_stats['rows']} with_text={val_label_text_stats['with_text']} "
        f"missing_text={val_label_text_stats['missing_text']}"
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
        max_samples=int(args.max_train_samples),
    )
    val_records = load_records(
        annotation_file=val_ann,
        image_root=image_root,
        label_map=val_label_map,
        label_text_map=val_label_text_map,
        split_prefix=args.split_prefix,
        strip_split_prefix=bool(args.strip_split_prefix),
        max_samples=int(args.max_val_samples),
    )
    if not train_records:
        raise RuntimeError("No train samples were loaded.")

    train_valid_targets = _count_valid_targets(train_records, id2label)
    val_valid_targets = _count_valid_targets(val_records, id2label)
    print(
        f"[INFO] train_records={len(train_records)} val_records={len(val_records)} "
        f"train_valid_targets={train_valid_targets} val_valid_targets={val_valid_targets}"
    )

    train_ds = GazeDataset(
        records=train_records,
        prompt_template=args.prompt_template,
        prompt_text=prompt_text_for_run,
        apply_augmentation=True,
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        answer_template=args.answer_template,
        fallback_target_text=args.fallback_target_text,
        fallback_object_id=int(args.fallback_object_id),
        point_decimals=int(args.point_decimals),
        visual_prompting=bool(args.visual_prompting),
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
        answer_template=args.answer_template,
        fallback_target_text=args.fallback_target_text,
        fallback_object_id=int(args.fallback_object_id),
        point_decimals=int(args.point_decimals),
        visual_prompting=bool(args.visual_prompting),
    )

    load_dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
        print("[WARN] non-CUDA device detected; forcing model dtype to float32.")
        load_dtype = torch.float32

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if load_dtype != "auto":
        model_kwargs["dtype"] = load_dtype

    processor_path = model_path
    if checkpoint_dir is not None and (checkpoint_dir / "processor").exists():
        processor_path = checkpoint_dir / "processor"
    processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)

    collator_include_raw_inputs = bool(getattr(args, "collator_include_raw_inputs", False))
    if collator_include_raw_inputs:
        print("[INFO] collator raw input passthrough: enabled (scene_images/text_inputs).")

    train_collator = QwenTrainCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
        include_raw_inputs=collator_include_raw_inputs,
    )
    val_collator = QwenTrainCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
        include_raw_inputs=collator_include_raw_inputs,
    )
    test_collator = QwenTestCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
        include_raw_inputs=collator_include_raw_inputs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=val_collator,
    )

    base_qwen = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)
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

    model = QwenTextGenerationModel(qwen_model=qwen_lora).to(device)

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

    amp_dtype = to_autocast_dtype(load_dtype)
    loss_answer_weight = float(getattr(args, "loss_answer_weight", 0.0))
    loss_localization_weight = float(getattr(args, "loss_localization_weight", 1.0))
    loss_recognition_weight = float(getattr(args, "loss_recognition_weight", 1.0))
    loss_use_lm_fallback = bool(getattr(args, "loss_use_lm_fallback", False))
    print(
        "[INFO] structured loss weights: "
        f"answer={loss_answer_weight:.4f} "
        f"localization={loss_localization_weight:.4f} "
        f"recognition={loss_recognition_weight:.4f} "
        f"lm_fallback={str(loss_use_lm_fallback).lower()}"
    )
    best_val_loss = float("inf")
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

            joint_inputs = _move_joint_inputs_to_device(batch["joint_inputs"], device=device)
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
                out = model(
                    joint_inputs=joint_inputs,
                    labels=labels,
                    use_cache=False,
                )
                lm_loss = out.get("loss", None)
                if lm_loss is None:
                    raise RuntimeError("Model forward must return loss during training.")
                structured = compute_structured_losses(
                    logits=out.get("logits", None),
                    labels=labels,
                    loss_mask_answer=batch.get("loss_mask_answer", None),
                    loss_mask_point=batch.get("loss_mask_point", None),
                    loss_mask_objectid=batch.get("loss_mask_objectid", None),
                    weight_answer=loss_answer_weight,
                    weight_point=loss_localization_weight,
                    weight_objectid=loss_recognition_weight,
                    fallback_loss=(lm_loss if loss_use_lm_fallback else None),
                )
                raw_loss = structured["loss"]
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

                if (
                    wandb_run is not None
                    and int(args.wandb_log_every_steps) > 0
                    and (global_step % int(args.wandb_log_every_steps) == 0)
                ):
                    grad_norm_value = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    epoch_progress = (float(epoch) - 1.0) + (
                        float(updates_done_in_epoch) / max(float(updates_per_epoch), 1.0)
                    )
                    wandb_run.log(
                        {
                            "train/loss": float(raw_loss.detach().item()),
                            "train/loss_answer": float(structured["loss_answer"].detach().item()),
                            "train/loss_localization": float(structured["loss_localization"].detach().item()),
                            "train/loss_recognition": float(structured["loss_recognition"].detach().item()),
                            "train/loss_lm_fallback": float(lm_loss.detach().item()),
                            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "train/grad_norm": grad_norm_value,
                            "train/global_step": float(global_step),
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
                loss_answer_weight=loss_answer_weight,
                loss_localization_weight=loss_localization_weight,
                loss_recognition_weight=loss_recognition_weight,
                loss_use_lm_fallback=loss_use_lm_fallback,
                show_tqdm=bool(args.show_tqdm),
                desc=f"Eval {epoch}/{args.epochs}",
            )
            if len(val_ds) > 0
            else {"loss": train_loss}
        )
        val_loss = float(val_metrics.get("loss", train_loss))

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_loc={float(val_metrics.get('loss_localization', 0.0)):.6f} "
            f"val_rec={float(val_metrics.get('loss_recognition', 0.0)):.6f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/index": float(epoch),
                    "epoch/global_step": float(global_step),
                    "epoch/train_loss": float(train_loss),
                    "val/epoch": float(epoch),
                    "val/loss": float(val_loss),
                    "val/loss_answer": float(val_metrics.get("loss_answer", 0.0)),
                    "val/loss_localization": float(val_metrics.get("loss_localization", 0.0)),
                    "val/loss_recognition": float(val_metrics.get("loss_recognition", 0.0)),
                    "metric/val/loss": float(val_loss),
                },
                step=global_step,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = out_dir / "best"
            save_checkpoint(
                best_dir,
                epoch,
                model,
                processor,
                optimizer,
                scheduler,
                clear_dir=True,
            )

    if bool(args.eval_only) and len(val_ds) > 0:
        val_metrics = run_eval(
            model,
            val_loader,
            device,
            amp_dtype,
            loss_answer_weight=loss_answer_weight,
            loss_localization_weight=loss_localization_weight,
            loss_recognition_weight=loss_recognition_weight,
            loss_use_lm_fallback=loss_use_lm_fallback,
            show_tqdm=bool(args.show_tqdm),
            desc="Eval (checkpoint)",
        )
        best_val_loss = float(val_metrics.get("loss", best_val_loss))
        print(
            "[EVAL] "
            f"val_loss={best_val_loss:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/epoch": 0.0,
                    "val/loss": float(best_val_loss),
                    "val/loss_answer": float(val_metrics.get("loss_answer", 0.0)),
                    "val/loss_localization": float(val_metrics.get("loss_localization", 0.0)),
                    "val/loss_recognition": float(val_metrics.get("loss_recognition", 0.0)),
                    "metric/val/loss": float(best_val_loss),
                },
                step=global_step,
            )

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
            max_groups=int(args.max_test_samples),
        )
        start_index = max(0, int(getattr(args, "test_start_index", 0)))
        if start_index > 0:
            test_groups = test_groups[start_index:]
        test_num_samples = int(getattr(args, "test_num_samples", 0))
        if test_num_samples > 0:
            test_groups = test_groups[: max(1, test_num_samples)]
        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(
                f"[TEST] groups={len(test_groups)} "
                f"(start={start_index}, limit={(test_num_samples if test_num_samples > 0 else 'all')})"
            )
            test_ds = GazeTestDataset(
                groups=test_groups,
                prompt_template=args.prompt_template,
                prompt_text=prompt_text_for_run,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                answer_template=args.answer_template,
                fallback_target_text=args.fallback_target_text,
                fallback_object_id=int(args.fallback_object_id),
                point_decimals=int(args.point_decimals),
                visual_prompting=bool(args.visual_prompting),
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "test/epoch": float(effective_epochs),
                        "test/ExactMatch": float(test_metrics["ExactMatch"]),
                        "test/Contains": float(test_metrics["Contains"]),
                        "test/AvgL2": float(test_metrics.get("Avg L2", test_metrics.get("PointL2", 0.0))),
                        "test/MinL2": float(test_metrics.get("Min L2", 0.0)),
                        "test/PointL2": float(test_metrics.get("PointL2", 0.0)),
                        "test/acc@1": float(test_metrics.get("acc@1", 0.0)),
                        "test/acc@3": float(test_metrics.get("acc@3", 0.0)),
                        "test/multiacc@1": float(test_metrics.get("multiacc@1", 0.0)),
                        "test/ObjectIDValidRate": float(test_metrics.get("ObjectIDValidRate", 0.0)),
                        "test/num_samples": float(test_metrics["num_samples"]),
                        "test/num_valid_targets": float(test_metrics["num_valid_targets"]),
                    },
                    step=global_step,
                )
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    finish_wandb(wandb_run)
    print(f"[DONE] global_step={global_step} best_val_loss={best_val_loss:.6f} elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
