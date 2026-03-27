#!/usr/bin/env python3
from __future__ import annotations

from gaze_pipeline.core import (
    CHECKPOINT_ROOT,
    DEFAULT_CONFIG,
    MODEL_STORAGE_ROOT,
    ROOT_DIR,
    build_generation_kwargs,
    build_gazefollow_prompt,
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
from gaze_pipeline.inference import run_inference
from gaze_pipeline.trainer import run_trainer


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    config_dir = config_path.parent
    cfg = load_yaml(config_path)
    run_cfg = dict(cfg.get("run", {}) or {})
    run_mode = str(run_cfg.get("mode", "train")).strip().lower()

    print(f"[INFO] config_path={config_path}")
    print(f"[INFO] run_mode={run_mode}")

    if run_mode == "train":
        run_trainer(cfg=cfg, config_dir=config_dir)
        return
    if run_mode in {"infer", "inference"}:
        run_inference(cfg=cfg, config_dir=config_dir)
        return
    raise ValueError(f"Unsupported run.mode: {run_mode} (expected: train|infer)")


if __name__ == "__main__":
    main()
