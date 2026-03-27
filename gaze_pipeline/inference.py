from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForImageTextToText, AutoProcessor

from .core import (
    CHECKPOINT_ROOT,
    build_generation_kwargs,
    build_model_kwargs,
    build_processor_kwargs,
    finalize_prediction,
    generate_one,
    list_batch_samples,
    list_single_sample,
    load_prompt,
    prepare_model,
    resolve_path,
    validate_input_config,
)


def maybe_run_evaluation(
    cfg: dict[str, Any],
    config_dir: Path,
    pred_dir: Path,
    model_name: str,
) -> None:
    eval_cfg = dict(cfg.get("evaluation", {}) or {})
    if not bool(eval_cfg.get("run_after_generation", False)):
        return

    from evaluate import evaluate as evaluate_predictions

    gt_dir = resolve_path(config_dir, str(eval_cfg.get("gt_dir", "100_imgs_target")))
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found for evaluation: {gt_dir}")

    result = evaluate_predictions(pred_dir=pred_dir, gt_dir=gt_dir)
    result["pred_dir"] = str(pred_dir)
    result["model_name"] = model_name

    print("[EVAL] done")
    print(f"[EVAL] num_obs={result['num_obs']}")
    print(f"[EVAL] avg_l2={result['avg_l2']:.6f}")
    print(f"[EVAL] min_l2={result['min_l2']:.6f}")
    print(f"[EVAL] dist_to_avg={result['dist_to_avg']:.6f}")

    save_json_raw = str(eval_cfg.get("save_json", "")).strip()
    if save_json_raw:
        save_json_path = resolve_path(config_dir, save_json_raw)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[EVAL] saved_json={save_json_path}")


def _has_model_files(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    return (
        any(model_dir.glob("*.safetensors"))
        or any(model_dir.glob("pytorch_model*.bin"))
        or (model_dir / "model.safetensors.index.json").exists()
    )


def _has_processor_files(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    candidates = (
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    return any((model_dir / name).exists() for name in candidates)


def _resolve_optional_path(config_dir: Path, raw: str) -> Path | None:
    raw = str(raw).strip()
    if not raw:
        return None
    return resolve_path(config_dir, raw)


def _resolve_default_train_artifact_dirs(cfg: dict[str, Any]) -> tuple[Path | None, Path | None]:
    train_cfg = dict(cfg.get("train", {}) or {})
    train_output_cfg = dict(train_cfg.get("output", {}) or {})
    checkpoints_subdir = str(train_output_cfg.get("checkpoints_subdir", "")).strip()
    run_name = str(train_output_cfg.get("run_name", "")).strip()
    if not run_name:
        return None, None
    run_parent = CHECKPOINT_ROOT / checkpoints_subdir if checkpoints_subdir else CHECKPOINT_ROOT
    run_dir = run_parent / run_name
    merged_subdir = str(train_output_cfg.get("merged_model_subdir", "merged_model")).strip()
    merged_dir = run_dir / merged_subdir if merged_subdir else None
    adapter_dir = run_dir / "final_adapter"
    return merged_dir, adapter_dir


def load_model_for_inference(
    cfg: dict[str, Any],
    config_dir: Path,
    base_model_dir: Path,
    base_model_name: str,
    processor_kwargs: dict[str, Any],
    model_kwargs: dict[str, Any],
) -> tuple[Any, Any, str, str]:
    infer_cfg = dict(cfg.get("inference", {}) or {})

    explicit_model_dir = _resolve_optional_path(config_dir, infer_cfg.get("model_dir", ""))
    explicit_merged_dir = _resolve_optional_path(config_dir, infer_cfg.get("merged_model_dir", ""))
    explicit_adapter_dir = _resolve_optional_path(config_dir, infer_cfg.get("adapter_dir", ""))
    explicit_base_model_dir = _resolve_optional_path(config_dir, infer_cfg.get("base_model_dir", ""))

    use_train_artifacts = bool(infer_cfg.get("use_train_artifacts", True))
    prefer_merged_model = bool(infer_cfg.get("prefer_merged_model", True))
    allow_adapter_fallback = bool(infer_cfg.get("allow_adapter_fallback", True))

    default_merged_dir = None
    default_adapter_dir = None
    if use_train_artifacts:
        default_merged_dir, default_adapter_dir = _resolve_default_train_artifact_dirs(cfg)

    model_source = "base_model"
    model_name = base_model_name

    if explicit_model_dir is not None and _has_model_files(explicit_model_dir):
        processor = AutoProcessor.from_pretrained(str(explicit_model_dir), **processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(str(explicit_model_dir), **model_kwargs)
        return processor, model, explicit_model_dir.name, "explicit_model_dir"

    merged_dir = None
    if explicit_merged_dir is not None and _has_model_files(explicit_merged_dir):
        merged_dir = explicit_merged_dir
    elif prefer_merged_model and default_merged_dir is not None and _has_model_files(default_merged_dir):
        merged_dir = default_merged_dir
    if merged_dir is not None:
        processor = AutoProcessor.from_pretrained(str(merged_dir), **processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(str(merged_dir), **model_kwargs)
        return processor, model, merged_dir.name, "merged_model"

    adapter_dir = None
    if explicit_adapter_dir is not None and explicit_adapter_dir.exists():
        adapter_dir = explicit_adapter_dir
    elif allow_adapter_fallback and default_adapter_dir is not None and default_adapter_dir.exists():
        adapter_dir = default_adapter_dir

    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "Adapter inference requested but peft is not available. Install peft or use merged model."
            ) from e

        base_dir = explicit_base_model_dir if explicit_base_model_dir is not None else base_model_dir
        base_model = AutoModelForImageTextToText.from_pretrained(str(base_dir), **model_kwargs)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        processor_dir = adapter_dir if _has_processor_files(adapter_dir) else base_dir
        processor = AutoProcessor.from_pretrained(str(processor_dir), **processor_kwargs)
        model_source = f"base_plus_adapter({adapter_dir})"
        model_name = f"{base_model_name}+{adapter_dir.name}"
        return processor, model, model_name, model_source

    processor = AutoProcessor.from_pretrained(str(base_model_dir), **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(str(base_model_dir), **model_kwargs)
    return processor, model, model_name, model_source


def run_inference(
    cfg: dict[str, Any],
    config_dir: Path,
) -> None:
    input_cfg = dict(cfg.get("input", {}) or {})
    prompt_cfg = dict(cfg.get("prompt", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})
    generation_cfg = dict(cfg.get("generation", {}) or {})

    validate_input_config(input_cfg, config_dir)

    model_dir, base_model_name, device, torch_dtype, trust_remote_code = prepare_model(
        dict(cfg.get("model", {}) or {}),
        config_dir,
    )

    model_cfg = dict(cfg.get("model", {}) or {})
    processor_kwargs = build_processor_kwargs(model_cfg, trust_remote_code)
    model_kwargs = build_model_kwargs(model_cfg, trust_remote_code, torch_dtype)
    generation_kwargs = build_generation_kwargs(generation_cfg)

    processor, model, model_name, model_source = load_model_for_inference(
        cfg=cfg,
        config_dir=config_dir,
        base_model_dir=model_dir,
        base_model_name=base_model_name,
        processor_kwargs=processor_kwargs,
        model_kwargs=model_kwargs,
    )

    output_root = resolve_path(config_dir, str(output_cfg.get("root_dir", "100_imgs_output")))
    output_dir = output_root / model_name
    overwrite = bool(output_cfg.get("overwrite", True))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] base_model_dir={model_dir}")
    print(f"[INFO] model_name={model_name}")
    print(f"[INFO] model_source={model_source}")
    print(f"[INFO] device={device}, torch_dtype={torch_dtype}")
    print(f"[INFO] output_dir={output_dir}")

    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    mode = str(input_cfg["mode"])
    if mode == "batch":
        samples = list_batch_samples(input_cfg, config_dir)
    elif mode == "single":
        samples = list_single_sample(input_cfg, config_dir)
    else:
        raise ValueError(f"Unsupported input.mode: {mode}")

    if not samples:
        print("[INFO] no samples to process.")
        return

    total = len(samples)
    print(f"[INFO] total_samples={total}")

    jsonl_path = output_dir / "results.jsonl"
    jsonl_mode = "w" if overwrite else "a"
    with jsonl_path.open(jsonl_mode, encoding="utf-8") as jf:
        for idx, (image_path, prompt_path, stem) in enumerate(samples, start=1):
            print(f"[RUN] ({idx}/{total}) {stem}")
            out_path = output_dir / f"{stem}.txt"
            if out_path.exists() and not overwrite:
                print(f"[SKIP] exists: {out_path}")
                continue

            prompt_text = load_prompt(prompt_path, str(input_cfg.get("prompt_text", "")))
            raw_pred = generate_one(
                processor=processor,
                model=model,
                device=device,
                image_path=image_path,
                prompt_text=prompt_text,
                input_cfg=input_cfg,
                prompt_cfg=prompt_cfg,
                generation_kwargs=generation_kwargs,
            )
            prediction_text, parsed_point, parse_source = finalize_prediction(
                raw_prediction=raw_pred,
                output_cfg=output_cfg,
            )

            out_path.write_text(prediction_text + "\n", encoding="utf-8")
            row = {
                "id": stem,
                "image_path": str(image_path),
                "prompt_path": str(prompt_path) if prompt_path else None,
                "used_head_crop": bool(input_cfg.get("use_head_crop", True)),
                "prediction_saved": prediction_text,
                "raw_prediction": raw_pred,
                "parsed_point": list(parsed_point) if parsed_point is not None else None,
                "parse_source": parse_source,
            }
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            print(f"[DONE] ({idx}/{total}) {stem}")

    print(f"[INFO] saved results to: {output_dir}")
    maybe_run_evaluation(cfg=cfg, config_dir=config_dir, pred_dir=output_dir, model_name=model_name)
