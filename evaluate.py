#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PRED_ROOT = ROOT_DIR / "100_imgs_output"
DEFAULT_GT_DIR = ROOT_DIR / "100_imgs_target"
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
XY_LINE_RE = re.compile(
    rf"^\s*({FLOAT_RE.pattern})\s*(?:,|\s)\s*({FLOAT_RE.pattern})\s*$"
)


def parse_points_from_txt(path: Path) -> list[tuple[float, float]]:
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

    # Backward compatibility: allow predictions directly under pred_root.
    has_txt = any(pred_root.glob("*.txt"))
    if has_txt:
        return pred_root, pred_root.name

    raise FileNotFoundError(f"No prediction files found under {pred_root}")


def evaluate(pred_dir: Path, gt_dir: Path) -> dict[str, object]:
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

    for sample_id in common_ids:
        pred_points = parse_points_from_txt(pred_files[sample_id])
        if not pred_points:
            invalid_pred.append(sample_id)
            continue
        gp_pred = pred_points[0]

        gp_gt_all = parse_points_from_txt(gt_files[sample_id])
        gp_gt = [pt for pt in gp_gt_all if pt[0] != -1]
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
        # semgaze GFTestDistance naming
        "dist_to_avg": dist_to_avg,
        "avg_dist": avg_dist,
        "min_dist": min_dist,
        # common naming aliases
        "avg_l2": avg_dist,
        "min_l2": min_dist,
        # when there is only one GT point per sample, these metrics collapse to one value
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gaze point predictions with semgaze GFTestDistance logic.")
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--model", type=str, default=None, help="Model name to evaluate (e.g., Qwen3.5-9B).")
    parser.add_argument("--pred-dir", type=Path, default=None, help="Direct prediction directory override.")
    parser.add_argument("--list-models", action="store_true", help="List discovered models under --pred-root and exit.")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save full results JSON.")
    args = parser.parse_args()

    if args.list_models:
        model_dirs = list_model_dirs(args.pred_root)
        if not model_dirs:
            print(f"No model directories found under: {args.pred_root}")
            return
        print("available_models:")
        for d in model_dirs:
            print(f"- {d.name}")
        return

    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")

    pred_dir, model_name = resolve_pred_dir(
        pred_root=args.pred_root,
        model=args.model,
        pred_dir=args.pred_dir,
    )

    result = evaluate(pred_dir=pred_dir, gt_dir=args.gt_dir)
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
        print(
            "[INFO] single_gt_mode=true: dist_to_avg, avg_dist, min_dist are identical by definition."
        )
        print(f"point_l2={result['point_l2']:.6f}")
    print(
        "counts: "
        f"missing_pred={result['missing_pred_count']}, "
        f"extra_pred={result['extra_pred_count']}, "
        f"invalid_pred={result['invalid_pred_count']}, "
        f"invalid_gt={result['invalid_gt_count']}"
    )

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={args.save_json}")


if __name__ == "__main__":
    main()
