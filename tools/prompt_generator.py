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


def build_prompt(image_name: str, bbox_norm: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = bbox_norm
    return (
        "Task:\n"
        "You are given a person's head bounding box in the image. "
        "Predict the normalized gaze point (x, y) in [0, 1] where this person is looking.\n\n"
        f"- Image file: {image_name}\n"
        f"- Person Head Bbox [xmin, ymin, xmax, ymax] (normalized) = [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]\n\n"
        "Output Format:\n"
        "x, y  # normalized in [0,1]\n"
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

        bbox_norm = normalize_bbox(bbox=bbox, width=width, height=height)
        if not all(0.0 <= v <= 1.0 for v in bbox_norm):
            print(f"[INVALID] normalized bbox out of range [0,1]: {image_name}")
            missing += 1
            continue

        prompt_text = build_prompt(image_name=image_name, bbox_norm=bbox_norm)
        out_path.write_text(prompt_text, encoding="utf-8")
        created += 1

    print(
        f"Done. created={created}, skipped={skipped}, "
        f"missing_or_invalid={missing}, duplicates={duplicated}"
    )


if __name__ == "__main__":
    main()
