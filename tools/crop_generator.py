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


def ensure_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(
            f"Output path exists as file, not directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def normalize_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    return xmin / width, ymin / height, xmax / width, ymax / height


def denormalize_bbox(
    xmin_n: float,
    ymin_n: float,
    xmax_n: float,
    ymax_n: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        int(xmin_n * width),
        int(ymin_n * height),
        int(xmax_n * width),
        int(ymax_n * height),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_DIR / "100_imgs",
        help="Directory with source images.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=ROOT_DIR / "train_annotations_new.txt",
        help="Path to annotation txt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "100_imgs_crop",
        help="Directory to write head crop images.",
    )
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    rows_by_name = load_annotations(args.annotation_file)

    saved = 0
    missing = 0
    duplicates = 0
    invalid = 0

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for image_path in sorted(args.input_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in valid_exts:
            continue

        name = image_path.name
        matches = rows_by_name.get(name, [])

        if not matches:
            print(f"[MISSING] annotation not found: {name}")
            missing += 1
            continue

        if len(matches) > 1:
            print(f"[DUPLICATE] {name}: {len(matches)} rows found, using first row.")
            duplicates += 1

        row = matches[0]
        try:
            xmin = float(row[10])
            ymin = float(row[11])
            xmax = float(row[12])
            ymax = float(row[13])
        except (IndexError, ValueError):
            print(f"[INVALID] failed to parse head bbox: {name}")
            invalid += 1
            continue

        with Image.open(image_path) as img:
            width, height = img.size

            if width <= 0 or height <= 0:
                print(f"[INVALID] invalid image size: {name}")
                invalid += 1
                continue

            xmin_n, ymin_n, xmax_n, ymax_n = normalize_bbox(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                width=width,
                height=height,
            )
            if not all(0.0 <= v <= 1.0 for v in (xmin_n, ymin_n, xmax_n, ymax_n)):
                print(f"[INVALID] normalized bbox out of range [0,1]: {name}")
                invalid += 1
                continue

            x1, y1, x2, y2 = denormalize_bbox(
                xmin_n=xmin_n,
                ymin_n=ymin_n,
                xmax_n=xmax_n,
                ymax_n=ymax_n,
                width=width,
                height=height,
            )

            # Clamp bbox to image bounds for safe cropping.
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                print(f"[INVALID] empty bbox after clamp: {name} ({x1},{y1},{x2},{y2})")
                invalid += 1
                continue

            head_crop = img.crop((x1, y1, x2, y2))
            out_path = args.output_dir / name
            head_crop.save(out_path)
            saved += 1

    print(
        f"Done. saved={saved}, missing={missing}, duplicates={duplicates}, invalid={invalid}"
    )


if __name__ == "__main__":
    main()
