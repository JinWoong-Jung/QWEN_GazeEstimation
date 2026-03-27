#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForImageTextToText, AutoProcessor

from gaze_pipeline.core import (
    DEFAULT_CONFIG,
    build_generation_kwargs,
    build_model_kwargs,
    build_processor_kwargs,
    load_yaml,
    prepare_model,
)
from gaze_pipeline.trainer import run_final_test_eval

ROOT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot (base model only) evaluation on GazeFollow test split "
            "without saving per-sample prediction files."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--split-prefix", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="0 means all samples")
    parser.add_argument("--group-mode", type=str, default=None, choices=["auto", "image", "image_bbox"])
    parser.add_argument("--bbox-round-decimals", type=int, default=None)
    parser.add_argument("--hide-tqdm", action="store_true")
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)

    model_cfg = dict(cfg.get("model", {}) or {})
    input_cfg = dict(cfg.get("input", {}) or {})
    prompt_cfg = dict(cfg.get("prompt", {}) or {})
    generation_cfg = dict(cfg.get("generation", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})
    test_eval_cfg = dict(train_cfg.get("test_eval", {}) or {})

    # Zero-shot script intentionally evaluates the base model only.
    model_dir, model_name, device, torch_dtype, trust_remote_code = prepare_model(
        model_cfg,
        config_dir,
    )
    processor_kwargs = build_processor_kwargs(model_cfg, trust_remote_code)
    model_kwargs = build_model_kwargs(model_cfg, trust_remote_code, torch_dtype)
    generation_kwargs = build_generation_kwargs(generation_cfg)

    processor = AutoProcessor.from_pretrained(str(model_dir), **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    run_test_cfg: dict[str, Any] = dict(test_eval_cfg)
    run_test_cfg["enabled"] = True
    if args.annotation_file is not None:
        run_test_cfg["annotation_file"] = str(args.annotation_file)
    if args.images_dir is not None:
        run_test_cfg["images_dir"] = str(args.images_dir)
    if args.split_prefix is not None:
        run_test_cfg["split_prefix"] = args.split_prefix
    if args.max_samples is not None:
        run_test_cfg["max_samples"] = int(args.max_samples)
    if args.hide_tqdm:
        run_test_cfg["show_tqdm"] = False
    if args.group_mode is not None:
        run_test_cfg["group_mode"] = args.group_mode
    if args.bbox_round_decimals is not None:
        run_test_cfg["bbox_round_decimals"] = int(args.bbox_round_decimals)
    run_test_cfg.setdefault("coord_bins", int(train_cfg.get("coord_bins", 1000)))

    print(f"[INFO][zeroshot] config_path={config_path}")
    print(f"[INFO][zeroshot] base_model_dir={model_dir}")
    print(f"[INFO][zeroshot] model_name={model_name}")
    print(f"[INFO][zeroshot] device={device}, torch_dtype={torch_dtype}")

    result = run_final_test_eval(
        model=model,
        processor=processor,
        device=device,
        input_cfg=input_cfg,
        prompt_cfg=prompt_cfg,
        generation_kwargs=generation_kwargs,
        config_dir=config_dir,
        test_eval_cfg=run_test_cfg,
    )
    if result is None:
        raise RuntimeError("Zero-shot evaluation returned no result (disabled by config).")

    result["model_name"] = model_name
    result["model_source"] = "base_model_only"
    result["base_model_dir"] = str(model_dir)

    if args.save_json is not None:
        save_path = args.save_json
        if not save_path.is_absolute():
            save_path = ROOT_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO][zeroshot] saved_json={save_path}")


if __name__ == "__main__":
    main()
