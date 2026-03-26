#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from main import (
    DEFAULT_CONFIG,
    build_generation_kwargs,
    build_model_kwargs,
    build_processor_kwargs,
    crop_head_from_bbox,
    enforce_numeric_output_prompt,
    finalize_prediction,
    generate_one,
    load_yaml,
    parse_head_bbox_from_prompt,
    prepare_model,
)
from tools.prompt_generator import build_prompt, normalize_bbox, sanitize_bbox

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_FILE = ROOT_DIR / "data" / "gazefollow_extended" / "test_annotations_release.txt"
DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "gazefollow_extended"


def is_normalized_point(point: tuple[float, float]) -> bool:
    return 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0


def parse_gt_points(rows: list[list[str]]) -> list[tuple[float, float]]:
    gt_points: list[tuple[float, float]] = []
    for row in rows:
        try:
            x = float(row[8])
            y = float(row[9])
        except (IndexError, ValueError):
            continue

        # GazeFollow sometimes uses -1 as invalid marker.
        if x == -1.0 or y == -1.0:
            continue
        if not is_normalized_point((x, y)):
            continue
        gt_points.append((x, y))
    return gt_points


def parse_head_bbox(row: list[str]) -> tuple[float, float, float, float] | None:
    try:
        x1 = float(row[10])
        y1 = float(row[11])
        x2 = float(row[12])
        y2 = float(row[13])
    except (IndexError, ValueError):
        return None
    return x1, y1, x2, y2


def load_grouped_rows(
    annotation_file: Path,
    split_prefix: str,
) -> dict[str, list[list[str]]]:
    rows_by_image: dict[str, list[list[str]]] = defaultdict(list)

    with annotation_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            image_rel = row[0].strip()
            if not image_rel.startswith(split_prefix):
                continue
            rows_by_image[image_rel].append(row)

    return rows_by_image


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def generate_batch(
    processor: Any,
    model: Any,
    device: str,
    batch_items: list[dict[str, Any]],
    input_cfg: dict[str, Any],
    prompt_cfg: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> list[str]:
    use_image = bool(input_cfg.get("use_image", True))
    use_head_crop = use_image and bool(input_cfg.get("use_head_crop", True))
    strict_head_bbox = bool(input_cfg.get("strict_head_bbox", True))

    use_chat_template = bool(prompt_cfg.get("use_chat_template", True))
    add_generation_prompt = bool(prompt_cfg.get("add_generation_prompt", True))
    head_crop_context_text = str(
        prompt_cfg.get(
            "head_crop_context_text",
            "The first image is the full scene and the second image is the cropped head region.",
        )
    ).strip()

    prompts: list[str] = []
    images_payload: list[Any] = []

    for item in batch_items:
        image_path = item["image_path"]
        prompt_text = enforce_numeric_output_prompt(item["prompt_text"], prompt_cfg)

        image = None
        head_crop_image = None
        if use_image:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            if use_head_crop:
                bbox_norm = parse_head_bbox_from_prompt(prompt_text)
                if bbox_norm is None:
                    if strict_head_bbox:
                        raise RuntimeError(f"Head bbox not found in prompt for sample: {item['image_rel']}")
                else:
                    head_crop_image = crop_head_from_bbox(image, bbox_norm)
                    if head_crop_image is None and strict_head_bbox:
                        raise RuntimeError(
                            f"Failed to crop head image from bbox for sample: {item['image_rel']}"
                        )

        if use_chat_template and hasattr(processor, "apply_chat_template"):
            content: list[dict[str, Any]] = []
            if use_image:
                content.append({"type": "image", "image": image})
                if head_crop_image is not None:
                    content.append({"type": "image", "image": head_crop_image})
                    if head_crop_context_text:
                        content.append({"type": "text", "text": head_crop_context_text})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            prompt = prompt_text

        prompts.append(prompt)
        if use_image:
            if head_crop_image is not None:
                images_payload.append([image, head_crop_image])
            else:
                images_payload.append(image)

    processor_inputs: dict[str, Any] = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": True,
    }
    if use_image:
        processor_inputs["images"] = images_payload

    try:
        inputs = processor(**processor_inputs)
    except Exception as e:
        if use_image:
            raise RuntimeError(
                "Failed to build batched image+text inputs. "
                "Try lowering --batch-size or checking model/processor compatibility."
            ) from e
        raise

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(**inputs, **generation_kwargs)

    if "input_ids" in inputs and generated.shape[1] >= inputs["input_ids"].shape[1]:
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    else:
        new_tokens = generated

    texts = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [t.strip() for t in texts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot inference on GazeFollow test set directly from annotation file "
            "and report Avg L2 / Min L2 without saving prediction files."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-prefix", type=str, default="test2/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all samples.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")

    config_path = args.config.resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)
    model_cfg = dict(cfg.get("model", {}) or {})
    input_cfg = dict(cfg.get("input", {}) or {})
    prompt_cfg = dict(cfg.get("prompt", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})
    generation_cfg = dict(cfg.get("generation", {}) or {})

    if not args.annotation_file.exists():
        raise FileNotFoundError(f"annotation file not found: {args.annotation_file}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    grouped_rows = load_grouped_rows(args.annotation_file, args.split_prefix)
    image_rels = sorted(grouped_rows.keys())
    if args.max_samples > 0:
        image_rels = image_rels[: args.max_samples]
    if not image_rels:
        raise RuntimeError("No samples found from annotation file with current split prefix.")

    model_dir, model_name, device, torch_dtype, trust_remote_code = prepare_model(model_cfg, config_dir)
    processor_kwargs = build_processor_kwargs(model_cfg, trust_remote_code)
    model_kwargs = build_model_kwargs(model_cfg, trust_remote_code, torch_dtype)
    generation_kwargs = build_generation_kwargs(generation_cfg)

    print(f"[INFO] model_dir={model_dir}")
    print(f"[INFO] model_name={model_name}")
    print(f"[INFO] device={device}, torch_dtype={torch_dtype}")
    print(f"[INFO] annotation_file={args.annotation_file}")
    print(f"[INFO] dataset_root={args.dataset_root}")
    print(f"[INFO] split_prefix={args.split_prefix}")
    print(f"[INFO] batch_size={args.batch_size}")
    print(f"[INFO] total_samples={len(image_rels)}")

    processor = AutoProcessor.from_pretrained(str(model_dir), **processor_kwargs)
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "padding_side", None) != "left":
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    # Enforce image-based inference for this benchmark flow.
    run_input_cfg: dict[str, Any] = dict(input_cfg)
    run_input_cfg["use_image"] = True

    sum_avg_dist = 0.0
    sum_min_dist = 0.0
    num_obs = 0

    missing_image = 0
    invalid_gt = 0
    invalid_bbox = 0
    invalid_pred = 0
    failed_infer = 0
    pred_parse_source_counts: Counter[str] = Counter()

    def consume_batch(batch_items: list[dict[str, Any]]) -> None:
        nonlocal sum_avg_dist
        nonlocal sum_min_dist
        nonlocal num_obs
        nonlocal invalid_pred
        nonlocal failed_infer
        nonlocal pred_parse_source_counts

        if not batch_items:
            return

        try:
            raw_preds = generate_batch(
                processor=processor,
                model=model,
                device=device,
                batch_items=batch_items,
                input_cfg=run_input_cfg,
                prompt_cfg=prompt_cfg,
                generation_kwargs=generation_kwargs,
            )
            if len(raw_preds) != len(batch_items):
                raise RuntimeError(
                    f"Batched output size mismatch: got {len(raw_preds)} for {len(batch_items)} items."
                )
        except Exception:
            raw_preds = []
            for item in batch_items:
                try:
                    raw_pred = generate_one(
                        processor=processor,
                        model=model,
                        device=device,
                        image_path=item["image_path"],
                        prompt_text=item["prompt_text"],
                        input_cfg=run_input_cfg,
                        prompt_cfg=prompt_cfg,
                        generation_kwargs=generation_kwargs,
                    )
                except Exception:
                    failed_infer += 1
                    continue
                raw_preds.append((item, raw_pred))

            for item, raw_pred in raw_preds:
                _, parsed_point, parse_source = finalize_prediction(
                    raw_prediction=raw_pred,
                    output_cfg=output_cfg,
                )
                pred_parse_source_counts[parse_source] += 1
                if parsed_point is None:
                    invalid_pred += 1
                    continue

                gt_points = item["gt_points"]
                dists = [euclidean(parsed_point, gt_pt) for gt_pt in gt_points]
                sum_avg_dist += sum(dists) / len(dists)
                sum_min_dist += min(dists)
                num_obs += 1
            return

        for item, raw_pred in zip(batch_items, raw_preds):
            _, parsed_point, parse_source = finalize_prediction(
                raw_prediction=raw_pred,
                output_cfg=output_cfg,
            )
            pred_parse_source_counts[parse_source] += 1
            if parsed_point is None:
                invalid_pred += 1
                continue

            gt_points = item["gt_points"]
            dists = [euclidean(parsed_point, gt_pt) for gt_pt in gt_points]
            sum_avg_dist += sum(dists) / len(dists)
            sum_min_dist += min(dists)
            num_obs += 1

    with torch.inference_mode():
        progress_bar = tqdm(image_rels, desc="Zero-shot eval", unit="sample")
        pending_batch: list[dict[str, Any]] = []
        for idx, image_rel in enumerate(progress_bar, start=1):
            rows = grouped_rows[image_rel]
            image_path = args.dataset_root / image_rel
            if not image_path.exists():
                missing_image += 1
                continue

            gt_points = parse_gt_points(rows)
            if not gt_points:
                invalid_gt += 1
                continue

            bbox_px = parse_head_bbox(rows[0])
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

            sanitized_bbox = sanitize_bbox(bbox=bbox_px, width=width, height=height)
            if sanitized_bbox is None:
                invalid_bbox += 1
                continue

            bbox_norm = normalize_bbox(bbox=sanitized_bbox, width=width, height=height)
            prompt_text = build_prompt(bbox_norm=bbox_norm)

            pending_batch.append(
                {
                    "image_rel": image_rel,
                    "image_path": image_path,
                    "prompt_text": prompt_text,
                    "gt_points": gt_points,
                }
            )
            if len(pending_batch) >= args.batch_size:
                consume_batch(pending_batch)
                pending_batch = []

            if args.progress_every > 0 and idx % args.progress_every == 0:
                progress_bar.set_postfix(
                    valid=num_obs,
                    invalid_pred=invalid_pred,
                    failed_infer=failed_infer,
                )

        if pending_batch:
            consume_batch(pending_batch)

    if num_obs == 0:
        raise RuntimeError("No valid predictions to evaluate.")

    avg_l2 = sum_avg_dist / num_obs
    min_l2 = sum_min_dist / num_obs

    print("[EVAL] done")
    print(f"[EVAL] num_obs={num_obs}")
    print(f"[EVAL] avg_l2={avg_l2:.6f}")
    print(f"[EVAL] min_l2={min_l2:.6f}")
    print(
        "[EVAL] counts: "
        f"missing_image={missing_image}, "
        f"invalid_gt={invalid_gt}, "
        f"invalid_bbox={invalid_bbox}, "
        f"invalid_pred={invalid_pred}, "
        f"failed_infer={failed_infer}"
    )
    if pred_parse_source_counts:
        print(f"[EVAL] pred_parse_sources={dict(pred_parse_source_counts)}")


if __name__ == "__main__":
    main()
