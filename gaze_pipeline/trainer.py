from __future__ import annotations

import csv
import inspect
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

from .core import (
    CHECKPOINT_ROOT,
    build_generation_kwargs,
    build_gazefollow_prompt,
    build_model_kwargs,
    build_processor_kwargs,
    crop_head_from_bbox,
    enforce_numeric_output_prompt,
    ensure_dir,
    generate_one,
    is_normalized_point,
    normalize_bbox_pixels,
    parse_lora_target_modules,
    parse_prediction_point_from_text,
    prepare_model,
    resolve_path,
    sanitize_bbox_pixels,
)


def freeze_params_by_name_patterns(
    model: Any,
    name_patterns: list[str],
) -> tuple[int, int]:
    patterns = [p.strip().lower() for p in name_patterns if p and p.strip()]
    if not patterns:
        return 0, 0

    frozen_tensors = 0
    frozen_params = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(pat in lname for pat in patterns):
            if param.requires_grad:
                frozen_tensors += 1
                frozen_params += int(param.numel())
            param.requires_grad = False
    return frozen_tensors, frozen_params


def load_gazefollow_records(
    annotation_file: Path,
    images_root: Path,
    split_prefix: str,
    max_samples: int,
    skip_outside_frame: bool,
    show_progress: bool,
    strip_split_prefix: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not annotation_file.exists():
        raise FileNotFoundError(f"annotation file not found: {annotation_file}")
    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")

    with annotation_file.open("r", encoding="utf-8", newline="") as f:
        all_rows = [row for row in csv.reader(f) if row]

    records: list[dict[str, Any]] = []
    stats = {
        "total_rows": 0,
        "kept": 0,
        "skip_split": 0,
        "skip_missing_image": 0,
        "skip_bad_gaze": 0,
        "skip_bad_bbox": 0,
        "skip_bad_image": 0,
    }

    size_cache: dict[Path, tuple[int, int] | None] = {}
    iterator = tqdm(
        all_rows,
        desc=f"Load {annotation_file.name}",
        unit="row",
        disable=not show_progress,
    )
    for idx, row in enumerate(iterator):
        stats["total_rows"] += 1
        image_rel = str(row[0]).strip()
        if split_prefix and not image_rel.startswith(split_prefix):
            stats["skip_split"] += 1
            continue

        image_rel_for_join = image_rel
        if split_prefix and strip_split_prefix:
            image_rel_for_join = image_rel[len(split_prefix) :]
        image_path = images_root / image_rel_for_join
        if not image_path.exists():
            stats["skip_missing_image"] += 1
            continue

        try:
            gaze_x = float(row[8])
            gaze_y = float(row[9])
        except (IndexError, ValueError):
            stats["skip_bad_gaze"] += 1
            continue

        if skip_outside_frame and (gaze_x < 0.0 or gaze_y < 0.0):
            stats["skip_bad_gaze"] += 1
            continue
        if not is_normalized_point((gaze_x, gaze_y)):
            stats["skip_bad_gaze"] += 1
            continue

        try:
            bbox_px = (float(row[10]), float(row[11]), float(row[12]), float(row[13]))
        except (IndexError, ValueError):
            stats["skip_bad_bbox"] += 1
            continue

        if image_path not in size_cache:
            try:
                with Image.open(image_path) as img:
                    size_cache[image_path] = img.size
            except Exception:
                size_cache[image_path] = None

        size = size_cache[image_path]
        if size is None:
            stats["skip_bad_image"] += 1
            continue
        width, height = size
        if width <= 0 or height <= 0:
            stats["skip_bad_image"] += 1
            continue

        sanitized_bbox = sanitize_bbox_pixels(bbox_px, width, height)
        if sanitized_bbox is None:
            stats["skip_bad_bbox"] += 1
            continue
        bbox_norm = normalize_bbox_pixels(sanitized_bbox, width, height)

        records.append(
            {
                "sample_id": f"{Path(image_rel).stem}_{idx}",
                "image_rel": image_rel,
                "image_path": image_path,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "bbox_norm": bbox_norm,
            }
        )
        stats["kept"] += 1
        if max_samples > 0 and len(records) >= max_samples:
            break
    return records, stats


class GazeFollowTrainDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        input_cfg: dict[str, Any],
        prompt_cfg: dict[str, Any],
        target_decimals: int,
    ) -> None:
        self.records = records
        self.input_cfg = input_cfg
        self.prompt_cfg = prompt_cfg
        self.target_decimals = target_decimals
        self.use_head_crop = bool(input_cfg.get("use_head_crop", False))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image_path = Path(record["image_path"])
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        bbox_norm = tuple(record["bbox_norm"])
        prompt_text = build_gazefollow_prompt(bbox_norm, self.prompt_cfg)
        prompt_text = enforce_numeric_output_prompt(prompt_text, self.prompt_cfg)
        target_text = (
            f"{float(record['gaze_x']):.{self.target_decimals}f} "
            f"{float(record['gaze_y']):.{self.target_decimals}f}"
        )

        head_crop_image = None
        if self.use_head_crop:
            head_crop_image = crop_head_from_bbox(image, bbox_norm)

        return {
            "sample_id": record["sample_id"],
            "image": image,
            "head_crop_image": head_crop_image,
            "prompt_text": prompt_text,
            "target_text": target_text,
        }


class QwenVLTrainCollator:
    def __init__(
        self,
        processor: Any,
        input_cfg: dict[str, Any],
        prompt_cfg: dict[str, Any],
        max_length: int,
    ) -> None:
        self.processor = processor
        self.input_cfg = input_cfg
        self.prompt_cfg = prompt_cfg
        self.max_length = max_length
        self.use_image = bool(input_cfg.get("use_image", True))
        self.use_head_crop = self.use_image and bool(input_cfg.get("use_head_crop", False))
        self.strict_head_bbox = bool(input_cfg.get("strict_head_bbox", True))
        self.use_chat_template = bool(prompt_cfg.get("use_chat_template", True))
        self.head_crop_context_text = str(
            prompt_cfg.get(
                "head_crop_context_text",
                "The first image is the full scene and the second image is the cropped head region.",
            )
        ).strip()

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # For multimodal batches, tokenizer truncation can desync image tokens
        # and text tokens. Keep full sequence when images are present.
        allow_truncation = (self.max_length > 0) and (not self.use_image)

        full_texts: list[str] = []
        prompt_only_texts: list[str] = []
        images_payload: list[Any] = []

        for feat in features:
            content: list[dict[str, Any]] = []
            if self.use_image:
                content.append({"type": "image", "image": feat["image"]})
                if self.use_head_crop:
                    head_crop_image = feat.get("head_crop_image")
                    if head_crop_image is None:
                        if self.strict_head_bbox:
                            raise RuntimeError(
                                f"Missing head crop image for sample: {feat['sample_id']}"
                            )
                    else:
                        content.append({"type": "image", "image": head_crop_image})
                        if self.head_crop_context_text:
                            content.append(
                                {"type": "text", "text": self.head_crop_context_text}
                            )
            content.append({"type": "text", "text": feat["prompt_text"]})

            if self.use_chat_template and hasattr(self.processor, "apply_chat_template"):
                user_messages = [{"role": "user", "content": content}]
                full_messages = user_messages + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": feat["target_text"]}],
                    }
                ]
                prompt_only_texts.append(
                    self.processor.apply_chat_template(
                        user_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                full_texts.append(
                    self.processor.apply_chat_template(
                        full_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            else:
                prompt_only = feat["prompt_text"].rstrip()
                full_text = f"{prompt_only}\n{feat['target_text']}"
                prompt_only_texts.append(prompt_only)
                full_texts.append(full_text)

            if self.use_image:
                if self.use_head_crop and feat.get("head_crop_image") is not None:
                    images_payload.append([feat["image"], feat["head_crop_image"]])
                else:
                    images_payload.append(feat["image"])

        processor_inputs: dict[str, Any] = {
            "text": full_texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": allow_truncation,
        }
        if allow_truncation:
            processor_inputs["max_length"] = self.max_length
        if self.use_image:
            processor_inputs["images"] = images_payload

        batch = self.processor(**processor_inputs)

        tok_kwargs: dict[str, Any] = {
            "text": prompt_only_texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": allow_truncation,
        }
        if allow_truncation:
            tok_kwargs["max_length"] = self.max_length
        prompt_tokens = self.processor.tokenizer(**tok_kwargs)
        prompt_lens = prompt_tokens["attention_mask"].sum(dim=1).tolist()

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, plen in enumerate(prompt_lens):
            keep = min(int(plen), labels.shape[1])
            labels[i, :keep] = -100
        batch["labels"] = labels
        return batch


def load_grouped_rows_for_split(
    annotation_file: Path,
    split_prefix: str,
) -> dict[str, list[list[str]]]:
    rows_by_image: dict[str, list[list[str]]] = {}
    with annotation_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            image_rel = str(row[0]).strip()
            if split_prefix and not image_rel.startswith(split_prefix):
                continue
            rows_by_image.setdefault(image_rel, []).append(row)
    return rows_by_image


def parse_gt_points_from_rows(rows: list[list[str]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        try:
            x = float(row[8])
            y = float(row[9])
        except (IndexError, ValueError):
            continue
        if x == -1.0 or y == -1.0:
            continue
        if not is_normalized_point((x, y)):
            continue
        points.append((x, y))
    return points


def parse_head_bbox_from_row(row: list[str]) -> tuple[float, float, float, float] | None:
    try:
        x1 = float(row[10])
        y1 = float(row[11])
        x2 = float(row[12])
        y2 = float(row[13])
    except (IndexError, ValueError):
        return None
    return x1, y1, x2, y2


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def run_final_test_eval(
    model: Any,
    processor: Any,
    device: str,
    input_cfg: dict[str, Any],
    prompt_cfg: dict[str, Any],
    generation_kwargs: dict[str, Any],
    config_dir: Path,
    test_eval_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(test_eval_cfg.get("enabled", True)):
        print("[INFO][test] final test evaluation disabled by config.")
        return None

    annotation_file = resolve_path(
        config_dir,
        str(
            test_eval_cfg.get(
                "annotation_file",
                "data/gazefollow_extended/test_annotations_release.txt",
            )
        ),
    )
    images_root = resolve_path(
        config_dir,
        str(
            test_eval_cfg.get(
                "images_dir",
                "/home/elicer/QWEN_GazeEstimation/data/gazefollow_extended/test2",
            )
        ),
    )
    split_prefix = str(test_eval_cfg.get("split_prefix", "test2/"))
    strip_split_prefix = bool(test_eval_cfg.get("strip_split_prefix", True))
    max_samples = int(test_eval_cfg.get("max_samples", 0))
    show_tqdm = bool(test_eval_cfg.get("show_tqdm", True))

    if not annotation_file.exists():
        raise FileNotFoundError(f"[test] annotation file not found: {annotation_file}")
    if not images_root.exists():
        raise FileNotFoundError(f"[test] images root not found: {images_root}")

    rows_by_image = load_grouped_rows_for_split(annotation_file, split_prefix)
    image_rels = sorted(rows_by_image.keys())
    if max_samples > 0:
        image_rels = image_rels[:max_samples]
    if not image_rels:
        raise RuntimeError("[test] no test samples found with current split_prefix.")

    print(f"[INFO][test] annotation_file={annotation_file}")
    print(f"[INFO][test] images_root={images_root}")
    print(f"[INFO][test] split_prefix={split_prefix}")
    print(f"[INFO][test] total_samples={len(image_rels)}")

    sum_dist_to_avg = 0.0
    sum_avg_dist = 0.0
    sum_min_dist = 0.0
    num_obs = 0
    missing_image = 0
    invalid_gt = 0
    invalid_bbox = 0
    invalid_pred = 0
    failed_infer = 0

    eval_input_cfg = dict(input_cfg)
    eval_input_cfg["use_image"] = True

    with torch.inference_mode():
        iterator = tqdm(image_rels, desc="Final test eval", unit="sample", disable=not show_tqdm)
        for image_rel in iterator:
            rows = rows_by_image[image_rel]
            image_rel_for_join = image_rel
            if split_prefix and strip_split_prefix and image_rel.startswith(split_prefix):
                image_rel_for_join = image_rel[len(split_prefix) :]
            image_path = images_root / image_rel_for_join
            if not image_path.exists():
                missing_image += 1
                continue

            gt_points = parse_gt_points_from_rows(rows)
            if not gt_points:
                invalid_gt += 1
                continue

            bbox_px = parse_head_bbox_from_row(rows[0])
            if bbox_px is None:
                invalid_bbox += 1
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                missing_image += 1
                continue
            if width <= 0 or height <= 0:
                missing_image += 1
                continue

            sanitized_bbox = sanitize_bbox_pixels(bbox_px, width, height)
            if sanitized_bbox is None:
                invalid_bbox += 1
                continue
            bbox_norm = normalize_bbox_pixels(sanitized_bbox, width, height)
            prompt_text = build_gazefollow_prompt(bbox_norm, prompt_cfg)

            try:
                raw_pred = generate_one(
                    processor=processor,
                    model=model,
                    device=device,
                    image_path=image_path,
                    prompt_text=prompt_text,
                    input_cfg=eval_input_cfg,
                    prompt_cfg=prompt_cfg,
                    generation_kwargs=generation_kwargs,
                )
            except Exception:
                failed_infer += 1
                continue

            pred_point, _ = parse_prediction_point_from_text(raw_pred)
            if pred_point is None:
                invalid_pred += 1
                continue

            gt_avg = (
                sum(p[0] for p in gt_points) / len(gt_points),
                sum(p[1] for p in gt_points) / len(gt_points),
            )
            dists = [euclidean(pred_point, gt_pt) for gt_pt in gt_points]
            sum_dist_to_avg += euclidean(pred_point, gt_avg)
            sum_avg_dist += sum(dists) / len(dists)
            sum_min_dist += min(dists)
            num_obs += 1

    if num_obs == 0:
        raise RuntimeError("[test] no valid predictions to evaluate.")

    dist = sum_dist_to_avg / num_obs
    avg_l2 = sum_avg_dist / num_obs
    min_l2 = sum_min_dist / num_obs

    print("[TEST] done")
    print(f"[TEST] dist={dist:.6f}")
    print(f"[TEST] Avg L2={avg_l2:.6f}")
    print(f"[TEST] Min L2={min_l2:.6f}")
    print(
        "[TEST] counts: "
        f"missing_image={missing_image}, "
        f"invalid_gt={invalid_gt}, "
        f"invalid_bbox={invalid_bbox}, "
        f"invalid_pred={invalid_pred}, "
        f"failed_infer={failed_infer}"
    )
    return {
        "dist": dist,
        "avg_l2": avg_l2,
        "min_l2": min_l2,
        "num_obs": num_obs,
        "missing_image": missing_image,
        "invalid_gt": invalid_gt,
        "invalid_bbox": invalid_bbox,
        "invalid_pred": invalid_pred,
        "failed_infer": failed_infer,
    }


def run_trainer(
    cfg: dict[str, Any],
    config_dir: Path,
) -> None:
    model_cfg = dict(cfg.get("model", {}) or {})
    data_paths_cfg = dict(cfg.get("data_paths", {}) or {})
    input_cfg = dict(cfg.get("input", {}) or {})
    prompt_cfg = dict(cfg.get("prompt", {}) or {})
    generation_cfg = dict(cfg.get("generation", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})

    dataset_cfg = dict(train_cfg.get("dataset", {}) or {})
    optimizer_cfg = dict(train_cfg.get("optimizer", {}) or {})
    trainer_cfg = dict(train_cfg.get("trainer", {}) or {})
    lora_cfg = dict(train_cfg.get("lora", {}) or {})
    output_cfg = dict(train_cfg.get("output", {}) or {})
    test_eval_cfg = dict(train_cfg.get("test_eval", {}) or {})
    wandb_cfg = dict(train_cfg.get("wandb", {}) or {})

    default_train_images_dir = str(
        data_paths_cfg.get(
            "train_images_dir",
            "/home/elicer/QWEN_GazeEstimation/data/gazefollow_extended/train",
        )
    )
    default_val_images_dir = str(
        data_paths_cfg.get(
            "val_images_dir",
            "/home/elicer/QWEN_GazeEstimation/data/gazefollow_extended/train",
        )
    )
    default_test_images_dir = str(
        data_paths_cfg.get(
            "test_images_dir",
            "/home/elicer/QWEN_GazeEstimation/data/gazefollow_extended/test2",
        )
    )
    default_test_annotation_file = str(
        data_paths_cfg.get(
            "test_annotation_file",
            "/home/elicer/QWEN_GazeEstimation/data/gazefollow_extended/test_annotations_release.txt",
        )
    )

    annotation_file = resolve_path(
        config_dir,
        str(dataset_cfg.get("train_annotation_file", "data/gazefollow/train_annotations_new.txt")),
    )
    train_images_root = resolve_path(
        config_dir,
        str(dataset_cfg.get("train_images_dir", default_train_images_dir)),
    )
    val_images_root = resolve_path(
        config_dir,
        str(dataset_cfg.get("val_images_dir", default_val_images_dir)),
    )
    test_images_root = resolve_path(
        config_dir,
        str(dataset_cfg.get("test_images_dir", default_test_images_dir)),
    )
    train_split_prefix = str(dataset_cfg.get("train_split_prefix", "train/"))
    val_split_prefix = str(dataset_cfg.get("val_split_prefix", "train/"))
    strip_split_prefix = bool(dataset_cfg.get("strip_split_prefix", True))
    max_train_samples = int(dataset_cfg.get("max_train_samples", 0))
    max_val_samples = int(dataset_cfg.get("max_val_samples", 0))
    skip_outside = bool(dataset_cfg.get("skip_outside_frame", True))
    show_data_tqdm = bool(dataset_cfg.get("show_loading_tqdm", True))

    val_annotation_raw = str(dataset_cfg.get("val_annotation_file", "")).strip()
    val_annotation_file = (
        resolve_path(config_dir, val_annotation_raw) if val_annotation_raw else None
    )
    do_eval = val_annotation_file is not None

    model_dir, model_name, device, torch_dtype, trust_remote_code = prepare_model(
        model_cfg,
        config_dir,
    )
    processor_kwargs = build_processor_kwargs(model_cfg, trust_remote_code)
    model_kwargs = build_model_kwargs(model_cfg, trust_remote_code, torch_dtype)
    generation_kwargs = build_generation_kwargs(generation_cfg)

    print(f"[INFO][train] model_dir={model_dir}")
    print(f"[INFO][train] model_name={model_name}")
    print(f"[INFO][train] device={device}, torch_dtype={torch_dtype}")
    print(f"[INFO][train] train_images_root={train_images_root}")
    print(f"[INFO][train] val_images_root={val_images_root}")
    print(f"[INFO][train] test_images_root={test_images_root}")

    processor = AutoProcessor.from_pretrained(str(model_dir), **processor_kwargs)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(device)

    gradient_checkpointing = bool(trainer_cfg.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if bool(model_kwargs.get("load_in_4bit", False)) or bool(model_kwargs.get("load_in_8bit", False)):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    task_type_raw = str(lora_cfg.get("task_type", "CAUSAL_LM")).upper()
    task_type = getattr(TaskType, task_type_raw, TaskType.CAUSAL_LM)
    lora_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=task_type,
        target_modules=parse_lora_target_modules(lora_cfg.get("target_modules", [])),
    )
    model = get_peft_model(model, lora_config)

    if bool(lora_cfg.get("freeze_vision", False)):
        vision_patterns_raw = lora_cfg.get(
            "vision_freeze_patterns",
            ["visual", "vision", "vision_tower", "image_tower", "visual_encoder", "vit"],
        )
        if isinstance(vision_patterns_raw, str):
            vision_patterns = [x.strip() for x in vision_patterns_raw.split(",") if x.strip()]
        elif isinstance(vision_patterns_raw, list):
            vision_patterns = [str(x).strip() for x in vision_patterns_raw if str(x).strip()]
        else:
            vision_patterns = []
        frozen_tensors, frozen_params = freeze_params_by_name_patterns(model, vision_patterns)
        print(
            "[INFO][train] vision_freeze=True "
            f"(patterns={vision_patterns}, frozen_tensors={frozen_tensors}, frozen_params={frozen_params})"
        )

    model.print_trainable_parameters()

    train_records, train_stats = load_gazefollow_records(
        annotation_file=annotation_file,
        images_root=train_images_root,
        split_prefix=train_split_prefix,
        max_samples=max_train_samples,
        skip_outside_frame=skip_outside,
        show_progress=show_data_tqdm,
        strip_split_prefix=strip_split_prefix,
    )
    if not train_records:
        raise RuntimeError("No valid train records found from annotation file.")

    print(f"[INFO][train] train_records={len(train_records)}")
    print(f"[INFO][train] train_stats={train_stats}")

    val_records: list[dict[str, Any]] = []
    if do_eval and val_annotation_file is not None:
        val_records, val_stats = load_gazefollow_records(
            annotation_file=val_annotation_file,
            images_root=val_images_root,
            split_prefix=val_split_prefix,
            max_samples=max_val_samples,
            skip_outside_frame=skip_outside,
            show_progress=show_data_tqdm,
            strip_split_prefix=strip_split_prefix,
        )
        if val_records:
            print(f"[INFO][train] val_records={len(val_records)}")
            print(f"[INFO][train] val_stats={val_stats}")
        else:
            print("[WARN][train] val annotation is set but no valid val records found. Eval disabled.")
            do_eval = False

    target_decimals = int(train_cfg.get("target_decimals", 6))
    train_dataset = GazeFollowTrainDataset(
        records=train_records,
        input_cfg=input_cfg,
        prompt_cfg=prompt_cfg,
        target_decimals=target_decimals,
    )
    eval_dataset = (
        GazeFollowTrainDataset(
            records=val_records,
            input_cfg=input_cfg,
            prompt_cfg=prompt_cfg,
            target_decimals=target_decimals,
        )
        if do_eval and val_records
        else None
    )

    collator = QwenVLTrainCollator(
        processor=processor,
        input_cfg=input_cfg,
        prompt_cfg=prompt_cfg,
        max_length=int(train_cfg.get("max_seq_length", 2048)),
    )

    checkpoints_dir = CHECKPOINT_ROOT
    checkpoints_subdir = str(output_cfg.get("checkpoints_subdir", "")).strip()
    run_name = str(output_cfg.get("run_name", "")).strip() or f"{model_name}-gazefollow-lora"
    run_parent_dir = checkpoints_dir / checkpoints_subdir if checkpoints_subdir else checkpoints_dir
    run_output_dir = run_parent_dir / run_name
    ensure_dir(run_output_dir)
    ensure_dir(CHECKPOINT_ROOT)

    report_to_raw = trainer_cfg.get("report_to", "none")
    if isinstance(report_to_raw, str):
        report_to = [] if report_to_raw.lower() == "none" else [report_to_raw]
    elif isinstance(report_to_raw, list):
        report_to = report_to_raw
    else:
        report_to = []
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        if "wandb" not in report_to:
            report_to.append("wandb")
        wandb_project = str(wandb_cfg.get("project", "")).strip()
        wandb_entity = str(wandb_cfg.get("entity", "")).strip()
        wandb_name = str(wandb_cfg.get("name", "")).strip()
        wandb_notes = str(wandb_cfg.get("notes", "")).strip()
        wandb_tags_raw = wandb_cfg.get("tags", [])
        if wandb_project:
            os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
        if wandb_name:
            os.environ["WANDB_NAME"] = wandb_name
        if wandb_notes:
            os.environ["WANDB_NOTES"] = wandb_notes
        if isinstance(wandb_tags_raw, list) and wandb_tags_raw:
            os.environ["WANDB_TAGS"] = ",".join(str(x) for x in wandb_tags_raw)
    else:
        report_to = [x for x in report_to if x != "wandb"]

    evaluation_strategy = "no"
    if do_eval and eval_dataset is not None:
        evaluation_strategy = str(trainer_cfg.get("evaluation_strategy", "steps"))

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(run_output_dir),
        "run_name": run_name,
        "remove_unused_columns": False,
        "per_device_train_batch_size": int(trainer_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(trainer_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(trainer_cfg.get("num_train_epochs", 1)),
        "max_steps": int(trainer_cfg.get("max_steps", -1)),
        "learning_rate": float(optimizer_cfg.get("learning_rate", 2e-4)),
        "weight_decay": float(optimizer_cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(optimizer_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": str(optimizer_cfg.get("lr_scheduler_type", "cosine")),
        "max_grad_norm": float(optimizer_cfg.get("max_grad_norm", 1.0)),
        "logging_steps": int(trainer_cfg.get("logging_steps", 10)),
        "logging_first_step": True,
        "save_strategy": str(trainer_cfg.get("save_strategy", "steps")),
        "save_steps": int(trainer_cfg.get("save_steps", 500)),
        "save_total_limit": int(trainer_cfg.get("save_total_limit", 3)),
        "eval_steps": int(trainer_cfg.get("eval_steps", 500)),
        "bf16": bool(trainer_cfg.get("bf16", False)),
        "fp16": bool(trainer_cfg.get("fp16", False)),
        "gradient_checkpointing": gradient_checkpointing,
        "dataloader_num_workers": int(trainer_cfg.get("dataloader_num_workers", 2)),
        "report_to": report_to,
        "disable_tqdm": bool(trainer_cfg.get("disable_tqdm", False)),
        "save_safetensors": bool(trainer_cfg.get("save_safetensors", True)),
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        training_args_kwargs["eval_strategy"] = evaluation_strategy
    else:
        training_args_kwargs["evaluation_strategy"] = evaluation_strategy
    training_args = TrainingArguments(**training_args_kwargs)

    print(f"[INFO][train] run_output_dir={run_output_dir}")
    print(f"[INFO][train] tqdm_enabled={not training_args.disable_tqdm}")
    print(f"[INFO][train] wandb_enabled={wandb_enabled}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=getattr(processor, "tokenizer", None),
    )

    resume_from_checkpoint_raw = str(train_cfg.get("resume_from_checkpoint", "")).strip()
    resume_from_checkpoint = None
    if resume_from_checkpoint_raw:
        resume_from_checkpoint = str(resolve_path(config_dir, resume_from_checkpoint_raw))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    final_adapter_dir = run_output_dir / "final_adapter"
    ensure_dir(final_adapter_dir)
    trainer.model.save_pretrained(str(final_adapter_dir))
    processor.save_pretrained(str(final_adapter_dir))

    print(f"[INFO][train] final_adapter_saved={final_adapter_dir}")
    merged_test_eval_cfg = dict(test_eval_cfg)
    merged_test_eval_cfg.setdefault("annotation_file", default_test_annotation_file)
    merged_test_eval_cfg.setdefault("images_dir", default_test_images_dir)
    merged_test_eval_cfg.setdefault("split_prefix", "test2/")
    merged_test_eval_cfg.setdefault("strip_split_prefix", True)
    merged_test_eval_cfg.setdefault("max_samples", 0)
    merged_test_eval_cfg.setdefault("show_tqdm", True)
    run_final_test_eval(
        model=trainer.model,
        processor=processor,
        device=device,
        input_cfg=input_cfg,
        prompt_cfg=prompt_cfg,
        generation_kwargs=generation_kwargs,
        config_dir=config_dir,
        test_eval_cfg=merged_test_eval_cfg,
    )
