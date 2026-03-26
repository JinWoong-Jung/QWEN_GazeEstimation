#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_annotations(annotation_path: Path) -> dict[str, list[list[str]]]:
    rows_by_name: dict[str, list[list[str]]] = defaultdict(list)

    with annotation_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            image_name = Path(row[0]).name
            rows_by_name[image_name].append(row)

    return rows_by_name


def get_head_bbox_from_row(row: list[str]) -> tuple[float, float, float, float]:
    """
    Parse head bbox from annotation row.
    Columns: [10, 11, 12, 13] -> xmin, ymin, xmax, ymax
    """
    xmin = float(row[10])
    ymin = float(row[11])
    xmax = float(row[12])
    ymax = float(row[13])
    return xmin, ymin, xmax, ymax


def get_head_bbox_for_image(
    image_name: str,
    rows_by_name: dict[str, list[list[str]]],
) -> tuple[tuple[float, float, float, float] | None, bool]:
    """
    Return:
    - bbox (xmin, ymin, xmax, ymax) or None if missing/invalid
    - duplicated flag (True if more than one annotation row exists)

    Rule for duplicates: use first row only.
    """
    matches = rows_by_name.get(image_name, [])
    if not matches:
        return None, False

    duplicated = len(matches) > 1
    row = matches[0]
    try:
        bbox = get_head_bbox_from_row(row)
    except (IndexError, ValueError):
        return None, duplicated

    return bbox, duplicated


def ensure_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path is a file, not directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    return xmin / width, ymin / height, xmax / width, ymax / height


def sanitize_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = bbox

    # Some rows have reversed order or slight out-of-bound values.
    x1, x2 = sorted((xmin, xmax))
    y1, y2 = sorted((ymin, ymax))

    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_prompt(bbox_norm: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = bbox_norm
    return (
        "Given the highlighted subject, estimate where the subject is looking(normalized gaze point).\n"
        f"Head bbox [xmin,ymin,xmax,ymax] (normalized): [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]\n"
        "Important rules:\n"
        "1) Predict where the person is looking AT in the scene (target object/location),\n"
        "NOT the eye position, face center, or head center.\n"
        "2) Coordinates must be in FULL SCENE coordinates, not head-crop coordinates.\n"
        "3) Prefer a point on the likely attended object along the gaze direction.\n"
        "4) Avoid points inside the head bbox unless no other plausible target is visible.\n"
        "\n"
        "Output format (one line only): x y"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "100_imgs")
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=ROOT_DIR / "train_annotations_new.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "100_imgs_prompt")
    parser.add_argument(
        "--overwrite",
        default=True,
        action="store_true",
        help="Overwrite existing txt files.",
    )
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    rows_by_name = load_annotations(args.annotation_file)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    created = 0
    skipped = 0
    missing = 0
    duplicated = 0

    for image_path in sorted(args.input_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in valid_exts:
            continue

        image_name = image_path.name
        out_path = args.output_dir / f"{image_path.stem}.txt"

        bbox, is_dup = get_head_bbox_for_image(image_name, rows_by_name)
        if bbox is None:
            print(f"[MISSING/INVALID] annotation not usable: {image_name}")
            missing += 1
            continue
        if is_dup:
            print(f"[DUPLICATE] {image_name}: using first row only.")
            duplicated += 1

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            print(f"[INVALID] failed to read image size: {image_name}")
            missing += 1
            continue

        if width <= 0 or height <= 0:
            print(f"[INVALID] invalid image size: {image_name}")
            missing += 1
            continue

        sanitized_bbox = sanitize_bbox(bbox=bbox, width=width, height=height)
        if sanitized_bbox is None:
            print(f"[INVALID] empty bbox after sanitize: {image_name}")
            missing += 1
            continue

        bbox_norm = normalize_bbox(bbox=sanitized_bbox, width=width, height=height)
        if not all(0.0 <= v <= 1.0 for v in bbox_norm):
            print(f"[INVALID] normalized bbox out of range [0,1]: {image_name}")
            missing += 1
            continue

        prompt_text = build_prompt(bbox_norm=bbox_norm)
        out_path.write_text(prompt_text, encoding="utf-8")
        created += 1

    print(
        f"Done. created={created}, skipped={skipped}, "
        f"missing_or_invalid={missing}, duplicates={duplicated}"
    )


if __name__ == "__main__":
    main()
