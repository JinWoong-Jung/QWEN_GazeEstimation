#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen3-VL-4B-Instruct"
DEFAULT_PRED_DIR = ROOT_DIR / "100_imgs_output" / DEFAULT_MODEL_NAME
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate import parse_gt_points_from_txt, parse_prediction_point_from_txt

HEAD_BBOX_RE = re.compile(
    r"Head bbox\s*\[xmin,ymin,xmax,ymax\]\s*\(normalized\):\s*\[([^\]]+)\]"
)


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x = int(round(clamp01(point[0]) * (width - 1)))
    y = int(round(clamp01(point[1]) * (height - 1)))
    return x, y


def draw_point(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=1)


def parse_head_bbox_from_prompt(prompt_path: Path) -> tuple[float, float, float, float] | None:
    text = prompt_path.read_text(encoding="utf-8")
    m = HEAD_BBOX_RE.search(text)
    if not m:
        return None
    try:
        vals = [float(x.strip()) for x in m.group(1).split(",")]
    except ValueError:
        return None
    if len(vals) != 4:
        return None
    x1, y1, x2, y2 = vals
    x1, x2 = sorted((clamp01(x1), clamp01(x2)))
    y1, y2 = sorted((clamp01(y1), clamp01(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def collect_images(image_dir: Path) -> dict[str, Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: dict[str, Path] = {}
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in valid_exts:
            images[p.stem] = p
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize GT (white) and prediction (red) points on images."
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Prediction directory (default: 100_imgs_output/Qwen3-VL-4B-Instruct).",
    )
    parser.add_argument("--image-dir", type=Path, default=ROOT_DIR / "100_imgs", help="Source image directory.")
    parser.add_argument("--gt-dir", type=Path, default=ROOT_DIR / "100_imgs_target", help="Ground-truth txt directory.")
    parser.add_argument("--prompt-dir", type=Path, default=ROOT_DIR / "100_imgs_prompt", help="Prompt txt directory for head bbox.")
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "100_imgs_visualization", help="Visualization output root directory.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Output subfolder name.")
    parser.add_argument("--point-radius", type=int, default=5, help="Point radius in pixels.")
    parser.add_argument("--bbox-width", type=int, default=2, help="Head bbox line width.")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True, help="Overwrite existing visualization files (default: true).")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Do not overwrite existing visualization files.")
    args = parser.parse_args()

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"pred-dir not found: {args.pred_dir}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"image-dir not found: {args.image_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"gt-dir not found: {args.gt_dir}")
    if not args.prompt_dir.exists():
        raise FileNotFoundError(f"prompt-dir not found: {args.prompt_dir}")

    model_name = args.model_name.strip() or args.pred_dir.name
    output_dir = args.output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_map = collect_images(args.image_dir)
    gt_map = {p.stem: p for p in args.gt_dir.glob("*.txt") if p.is_file()}
    prompt_map = {p.stem: p for p in args.prompt_dir.glob("*.txt") if p.is_file()}
    pred_map = {p.stem: p for p in args.pred_dir.glob("*.txt") if p.is_file() and p.name != "results.jsonl"}

    sample_ids = sorted(set(image_map) & set(gt_map) & set(prompt_map) & set(pred_map))

    saved = 0
    skipped_exists = 0
    invalid_pred = 0
    invalid_gt = 0

    for sample_id in sample_ids:
        img_path = image_map[sample_id]
        gt_path = gt_map[sample_id]
        prompt_path = prompt_map[sample_id]
        pred_path = pred_map[sample_id]
        out_path = output_dir / img_path.name

        if out_path.exists() and not args.overwrite:
            skipped_exists += 1
            continue

        pred_point, _ = parse_prediction_point_from_txt(pred_path)
        if pred_point is None:
            invalid_pred += 1
            continue

        gt_points_all = parse_gt_points_from_txt(gt_path)
        gt_points = [pt for pt in gt_points_all if pt[0] != -1]
        if not gt_points:
            invalid_gt += 1
            continue

        with Image.open(img_path) as im:
            image = im.convert("RGB")

        w, h = image.size
        draw = ImageDraw.Draw(image)

        bbox = parse_head_bbox_from_prompt(prompt_path)
        if bbox is not None:
            x1n, y1n, x2n, y2n = bbox
            x1, y1 = to_pixel((x1n, y1n), w, h)
            x2, y2 = to_pixel((x2n, y2n), w, h)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=max(1, args.bbox_width))

        # Draw GT points first (white), then prediction (red) on top.
        for gt in gt_points:
            gx, gy = to_pixel(gt, w, h)
            draw_point(draw, gx, gy, args.point_radius, fill=(255, 255, 255), outline=(0, 0, 0))

        px, py = to_pixel(pred_point, w, h)
        draw_point(draw, px, py, args.point_radius, fill=(255, 0, 0), outline=(255, 255, 255))

        image.save(out_path)
        saved += 1

    print(f"output_dir={output_dir}")
    print(f"model_name={model_name}")
    print(f"total_candidates={len(sample_ids)}")
    print(f"saved={saved}")
    print(f"skipped_exists={skipped_exists}")
    print(f"invalid_pred={invalid_pred}")
    print(f"invalid_gt={invalid_gt}")


if __name__ == "__main__":
    main()
