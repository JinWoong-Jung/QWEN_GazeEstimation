#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any

from gaze_pipeline.core import (
    CHECKPOINT_ROOT,
    DEFAULT_CONFIG,
    build_generation_kwargs,
    build_model_kwargs,
    build_processor_kwargs,
    is_normalized_point,
    load_yaml,
    parse_prediction_point_from_text,
    prepare_model,
)
from gaze_pipeline.trainer import run_final_test_eval

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PRED_ROOT = ROOT_DIR / "100_imgs_output"
DEFAULT_GT_DIR = ROOT_DIR / "100_imgs_target"

FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
XY_LINE_RE = re.compile(
    rf"^\s*({FLOAT_RE.pattern})\s*(?:,|\s)\s*({FLOAT_RE.pattern})\s*$"
)


def parse_gt_points_from_txt(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = XY_LINE_RE.match(line)
        if not m:
            continue
        x = float(m.group(1))
        y = float(m.group(2))
        points.append((x, y))
    return points


def parse_prediction_point_from_txt(
    path: Path,
    coord_bins: int,
) -> tuple[tuple[float, float] | None, str]:
    text = path.read_text(encoding="utf-8")
    return parse_prediction_point_from_text(text=text, coord_bins=coord_bins)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def normalize_model_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def list_model_dirs(pred_root: Path) -> list[Path]:
    if not pred_root.exists():
        return []
    return sorted([p for p in pred_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def resolve_pred_dir(pred_root: Path, model: str | None, pred_dir: Path | None) -> tuple[Path, str]:
    if pred_dir is not None:
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
        return pred_dir, pred_dir.name

    model_dirs = list_model_dirs(pred_root)
    model_names = [d.name for d in model_dirs]

    if model is not None:
        aliases = {model.strip(), model.strip().split("/")[-1]}
        alias_keys = {normalize_model_key(a) for a in aliases}

        matched: list[Path] = []
        for d in model_dirs:
            if d.name in aliases or normalize_model_key(d.name) in alias_keys:
                matched.append(d)

        if not matched:
            available = ", ".join(model_names) if model_names else "(none)"
            raise ValueError(f"Model '{model}' not found under {pred_root}. Available: {available}")
        if len(matched) > 1:
            raise ValueError(
                "Ambiguous model match for "
                f"'{model}': {', '.join(m.name for m in matched)}. "
                "Please pass --pred-dir explicitly."
            )
        return matched[0], matched[0].name

    if len(model_dirs) == 1:
        return model_dirs[0], model_dirs[0].name

    if len(model_dirs) > 1:
        raise ValueError(
            "Multiple model directories found under "
            f"{pred_root}: {', '.join(model_names)}. "
            "Please specify --model (or --pred-dir)."
        )

    has_txt = any(pred_root.glob("*.txt"))
    if has_txt:
        return pred_root, pred_root.name

    raise FileNotFoundError(f"No prediction files found under {pred_root}")


def evaluate(
    pred_dir: Path,
    gt_dir: Path,
    coord_bins: int = 1000,
) -> dict[str, object]:
    pred_files = {p.stem: p for p in pred_dir.glob("*.txt") if p.is_file() and p.name != "example.txt"}
    gt_files = {p.stem: p for p in gt_dir.glob("*.txt") if p.is_file()}

    missing_pred = sorted(set(gt_files) - set(pred_files))
    extra_pred = sorted(set(pred_files) - set(gt_files))
    common_ids = sorted(set(pred_files) & set(gt_files))

    sum_dist_to_avg = 0.0
    sum_avg_dist = 0.0
    sum_min_dist = 0.0
    num_obs = 0
    invalid_pred: list[str] = []
    invalid_gt: list[str] = []
    gt_point_counts: list[int] = []
    pred_parse_source_counts: Counter[str] = Counter()

    for sample_id in common_ids:
        gp_pred, pred_parse_source = parse_prediction_point_from_txt(
            pred_files[sample_id],
            coord_bins=coord_bins,
        )
        pred_parse_source_counts[pred_parse_source] += 1
        if gp_pred is None:
            invalid_pred.append(sample_id)
            continue

        gp_gt_all = parse_gt_points_from_txt(gt_files[sample_id])
        gp_gt = [pt for pt in gp_gt_all if pt[0] != -1 and is_normalized_point(pt)]
        if not gp_gt:
            invalid_gt.append(sample_id)
            continue
        gt_point_counts.append(len(gp_gt))

        gp_gt_avg = (
            sum(p[0] for p in gp_gt) / len(gp_gt),
            sum(p[1] for p in gp_gt) / len(gp_gt),
        )
        dists = [euclidean(gt_pt, gp_pred) for gt_pt in gp_gt]

        sum_dist_to_avg += euclidean(gp_gt_avg, gp_pred)
        sum_avg_dist += sum(dists) / len(dists)
        sum_min_dist += min(dists)
        num_obs += 1

    if num_obs == 0:
        raise RuntimeError("No valid matched samples to evaluate.")

    dist_to_avg = sum_dist_to_avg / num_obs
    avg_dist = sum_avg_dist / num_obs
    min_dist = sum_min_dist / num_obs
    single_gt_mode = all(c == 1 for c in gt_point_counts)
    gt_points_min = min(gt_point_counts)
    gt_points_max = max(gt_point_counts)
    gt_points_mean = sum(gt_point_counts) / len(gt_point_counts)

    return {
        "num_obs": num_obs,
        "dist_to_avg": dist_to_avg,
        "avg_dist": avg_dist,
        "min_dist": min_dist,
        "avg_l2": avg_dist,
        "min_l2": min_dist,
        "single_gt_mode": single_gt_mode,
        "point_l2": avg_dist if single_gt_mode else None,
        "gt_points_min": gt_points_min,
        "gt_points_max": gt_points_max,
        "gt_points_mean": gt_points_mean,
        "missing_pred_count": len(missing_pred),
        "extra_pred_count": len(extra_pred),
        "invalid_pred_count": len(invalid_pred),
        "invalid_gt_count": len(invalid_gt),
        "missing_pred_ids": missing_pred,
        "extra_pred_ids": extra_pred,
        "invalid_pred_ids": invalid_pred,
        "invalid_gt_ids": invalid_gt,
        "pred_parse_source_counts": dict(pred_parse_source_counts),
    }


def _resolve_run_output_dir(cfg: dict[str, Any]) -> Path:
    train_cfg = dict(cfg.get("train", {}) or {})
    output_cfg = dict(train_cfg.get("output", {}) or {})
    checkpoints_subdir = str(output_cfg.get("checkpoints_subdir", "")).strip()
    run_name = str(output_cfg.get("run_name", "")).strip()
    if not run_name:
        raise ValueError("train.output.run_name is required to resolve checkpoint directory.")
    run_parent = CHECKPOINT_ROOT / checkpoints_subdir if checkpoints_subdir else CHECKPOINT_ROOT
    return run_parent / run_name


def _find_latest_step_checkpoint(run_output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for p in run_output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


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


def _load_model_for_eval(
    *,
    cfg: dict[str, Any],
    base_model_dir: Path,
    base_model_name: str,
    processor_kwargs: dict[str, Any],
    model_kwargs: dict[str, Any],
    artifact_policy: str,
    explicit_checkpoint_dir: Path | None,
) -> tuple[Any, Any, str, str]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if explicit_checkpoint_dir is not None:
        artifact_dir = explicit_checkpoint_dir.resolve()
        artifact_source = "explicit_checkpoint_dir"
        if not artifact_dir.exists():
            raise FileNotFoundError(f"checkpoint_dir not found: {artifact_dir}")
    else:
        run_output_dir = _resolve_run_output_dir(cfg)
        final_adapter_dir = run_output_dir / "final_adapter"
        latest_ckpt_dir = _find_latest_step_checkpoint(run_output_dir)
        train_output_cfg = dict(dict(cfg.get("train", {}) or {}).get("output", {}) or {})
        merged_subdir = str(train_output_cfg.get("merged_model_subdir", "merged_model")).strip()
        merged_dir = run_output_dir / merged_subdir if merged_subdir else None

        artifact_dir = None
        artifact_source = "base_model_only"
        if artifact_policy == "final_adapter":
            if final_adapter_dir.exists():
                artifact_dir = final_adapter_dir
                artifact_source = "final_adapter"
            elif latest_ckpt_dir is not None:
                artifact_dir = latest_ckpt_dir
                artifact_source = f"latest_checkpoint({latest_ckpt_dir.name})"
        elif artifact_policy == "latest_checkpoint":
            if latest_ckpt_dir is not None:
                artifact_dir = latest_ckpt_dir
                artifact_source = f"latest_checkpoint({latest_ckpt_dir.name})"
            elif final_adapter_dir.exists():
                artifact_dir = final_adapter_dir
                artifact_source = "final_adapter_fallback"
        elif artifact_policy == "merged":
            if merged_dir is not None and merged_dir.exists():
                artifact_dir = merged_dir
                artifact_source = "merged_model"
        elif artifact_policy == "auto":
            if final_adapter_dir.exists():
                artifact_dir = final_adapter_dir
                artifact_source = "final_adapter"
            elif latest_ckpt_dir is not None:
                artifact_dir = latest_ckpt_dir
                artifact_source = f"latest_checkpoint({latest_ckpt_dir.name})"
            elif merged_dir is not None and merged_dir.exists():
                artifact_dir = merged_dir
                artifact_source = "merged_model"
        else:
            raise ValueError(
                f"Unsupported artifact policy: {artifact_policy}. "
                "Expected one of: final_adapter|latest_checkpoint|merged|auto"
            )

        if artifact_dir is None and artifact_policy != "auto":
            raise FileNotFoundError(
                f"No artifact found for policy='{artifact_policy}' under run dir: {run_output_dir}"
            )

    # merged/full model path
    if artifact_dir is not None and (artifact_dir / "config.json").exists():
        processor_dir = artifact_dir if _has_processor_files(artifact_dir) else base_model_dir
        processor = AutoProcessor.from_pretrained(str(processor_dir), **processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(str(artifact_dir), **model_kwargs)
        model_name = artifact_dir.name
        return processor, model, model_name, artifact_source

    # adapter path (final_adapter or checkpoint-xxxx)
    if artifact_dir is not None and (artifact_dir / "adapter_config.json").exists():
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "Adapter checkpoint evaluation requires `peft`. Please install peft first."
            ) from e

        base_model = AutoModelForImageTextToText.from_pretrained(str(base_model_dir), **model_kwargs)
        model = PeftModel.from_pretrained(base_model, str(artifact_dir), is_trainable=False)
        processor_dir = artifact_dir if _has_processor_files(artifact_dir) else base_model_dir
        processor = AutoProcessor.from_pretrained(str(processor_dir), **processor_kwargs)
        model_name = f"{base_model_name}+{artifact_dir.name}"
        return processor, model, model_name, artifact_source

    if artifact_dir is not None:
        raise RuntimeError(
            f"Unsupported artifact format at {artifact_dir}. "
            "Expected either adapter_config.json (LoRA adapter) or config.json (merged/full model)."
        )

    # fallback to base model only
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(base_model_dir), **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(str(base_model_dir), **model_kwargs)
    return processor, model, base_model_name, "base_model_only"


def run_model_eval(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)

    model_cfg = dict(cfg.get("model", {}) or {})
    prompt_cfg = dict(cfg.get("prompt", {}) or {})
    generation_cfg = dict(cfg.get("generation", {}) or {})
    input_cfg = dict(cfg.get("input", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})
    test_eval_cfg = dict(train_cfg.get("test_eval", {}) or {})

    model_dir, base_model_name, device, torch_dtype, trust_remote_code = prepare_model(
        model_cfg,
        config_dir,
    )
    processor_kwargs = build_processor_kwargs(model_cfg, trust_remote_code)
    model_kwargs = build_model_kwargs(model_cfg, trust_remote_code, torch_dtype)
    generation_kwargs = build_generation_kwargs(generation_cfg)

    explicit_checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else None
    processor, model, model_name, model_source = _load_model_for_eval(
        cfg=cfg,
        base_model_dir=model_dir,
        base_model_name=base_model_name,
        processor_kwargs=processor_kwargs,
        model_kwargs=model_kwargs,
        artifact_policy=args.artifact,
        explicit_checkpoint_dir=explicit_checkpoint_dir,
    )

    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    run_test_cfg = dict(test_eval_cfg)
    run_test_cfg["enabled"] = True
    data_paths_cfg = dict(cfg.get("data_paths", {}) or {})
    default_ann = data_paths_cfg.get(
        "test_annotation_file",
        str(ROOT_DIR / "data" / "gazefollow_extended" / "test_annotations_release.txt"),
    )
    default_images = data_paths_cfg.get(
        "test_images_dir",
        str(ROOT_DIR / "data" / "gazefollow_extended" / "test2"),
    )
    run_test_cfg.setdefault("annotation_file", default_ann)
    run_test_cfg.setdefault("images_dir", default_images)
    run_test_cfg.setdefault("split_prefix", "test2/")
    if args.annotation_file is not None:
        run_test_cfg["annotation_file"] = str(args.annotation_file)
    if args.images_dir is not None:
        run_test_cfg["images_dir"] = str(args.images_dir)
    if args.split_prefix is not None:
        run_test_cfg["split_prefix"] = args.split_prefix
    if args.max_samples is not None:
        run_test_cfg["max_samples"] = int(args.max_samples)
    if args.batch_size is not None:
        run_test_cfg["batch_size"] = int(args.batch_size)
    if args.hide_tqdm:
        run_test_cfg["show_tqdm"] = False
    if args.group_mode is not None:
        run_test_cfg["group_mode"] = args.group_mode
    if args.bbox_round_decimals is not None:
        run_test_cfg["bbox_round_decimals"] = int(args.bbox_round_decimals)
    run_test_cfg.setdefault("coord_bins", int(train_cfg.get("coord_bins", 1000)))

    print(f"[INFO][eval] config_path={config_path}")
    print(f"[INFO][eval] base_model_dir={model_dir}")
    print(f"[INFO][eval] model_name={model_name}")
    print(f"[INFO][eval] model_source={model_source}")
    print(f"[INFO][eval] artifact_policy={args.artifact}")
    print(f"[INFO][eval] device={device}, torch_dtype={torch_dtype}")

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
        raise RuntimeError("Model evaluation is disabled by config/test settings.")

    result["model_name"] = model_name
    result["model_source"] = model_source
    result["base_model_dir"] = str(model_dir)
    return result


def run_prediction_file_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.list_models:
        model_dirs = list_model_dirs(args.pred_root)
        if not model_dirs:
            print(f"No model directories found under: {args.pred_root}")
            return {}
        print("available_models:")
        for d in model_dirs:
            print(f"- {d.name}")
        return {}

    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")

    pred_dir, model_name = resolve_pred_dir(
        pred_root=args.pred_root,
        model=args.model,
        pred_dir=args.pred_dir,
    )

    result = evaluate(pred_dir=pred_dir, gt_dir=args.gt_dir, coord_bins=args.coord_bins)
    result["pred_dir"] = str(pred_dir)
    result["model_name"] = model_name

    print(f"pred_dir={pred_dir}")
    print(f"model_name={model_name}")
    print(f"num_obs={result['num_obs']}")
    print(f"dist_to_avg={result['dist_to_avg']:.6f}")
    print(f"avg_dist={result['avg_dist']:.6f}")
    print(f"min_dist={result['min_dist']:.6f}")
    print(f"avg_l2={result['avg_l2']:.6f}")
    print(f"min_l2={result['min_l2']:.6f}")
    print(
        "gt_points_per_sample: "
        f"min={result['gt_points_min']}, "
        f"mean={result['gt_points_mean']:.3f}, "
        f"max={result['gt_points_max']}"
    )
    if result["single_gt_mode"]:
        print("[INFO] single_gt_mode=true: dist_to_avg, avg_dist, min_dist are identical by definition.")
        print(f"point_l2={result['point_l2']:.6f}")
    print(
        "counts: "
        f"missing_pred={result['missing_pred_count']}, "
        f"extra_pred={result['extra_pred_count']}, "
        f"invalid_pred={result['invalid_pred_count']}, "
        f"invalid_gt={result['invalid_gt_count']}"
    )
    if result["pred_parse_source_counts"]:
        print(f"pred_parse_sources={json.dumps(result['pred_parse_source_counts'], ensure_ascii=False)}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate gaze predictions. Default mode evaluates the trained model directly on "
            "GazeFollow test annotations."
        )
    )
    parser.add_argument("--mode", choices=["model", "pred"], default="model")
    parser.add_argument("--save-json", type=Path, default=None)

    # model eval mode
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact",
        type=str,
        default="final_adapter",
        choices=["final_adapter", "latest_checkpoint", "merged", "auto"],
        help="Model artifact source for evaluation. Default is final_adapter.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Explicit checkpoint/adapter directory (overrides --artifact policy).",
    )
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--split-prefix", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="0 means all samples")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for test inference")
    parser.add_argument("--group-mode", type=str, default=None, choices=["auto", "image", "image_bbox"])
    parser.add_argument("--bbox-round-decimals", type=int, default=None)
    parser.add_argument("--hide-tqdm", action="store_true")

    # prediction-file eval mode
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--model", type=str, default=None, help="Model name to evaluate under --pred-root")
    parser.add_argument("--pred-dir", type=Path, default=None, help="Direct prediction directory override")
    parser.add_argument("--list-models", action="store_true", help="List discovered models under --pred-root")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--coord-bins", type=int, default=1000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "model":
        result = run_model_eval(args)
    else:
        result = run_prediction_file_eval(args)

    if args.save_json is not None and result:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={args.save_json}")


if __name__ == "__main__":
    main()
