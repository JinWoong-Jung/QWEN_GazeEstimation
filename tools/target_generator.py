#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

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
        help="Path to train_annotations_new.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "100_imgs_target",
        help="Directory to write txt targets.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_name = load_annotations(args.annotation_file)

    saved = 0
    missing = 0
    duplicates = 0

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for image_path in sorted(args.input_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in valid_exts:
            continue

        name = image_path.name
        stem = image_path.stem
        matches = rows_by_name.get(name, [])

        if not matches:
            print(f"[MISSING] annotation not found: {name}")
            missing += 1
            continue

        if len(matches) > 1:
            # Keep deterministic behavior when duplicated basenames exist.
            print(f"[DUPLICATE] {name}: {len(matches)} rows found, using first row.")
            duplicates += 1

        row = matches[0]
        try:
            gaze_x_norm = float(row[8])
            gaze_y_norm = float(row[9])
        except (IndexError, ValueError):
            print(f"[INVALID] failed to parse gaze coords: {name}")
            missing += 1
            continue

        if not (0.0 <= gaze_x_norm <= 1.0 and 0.0 <= gaze_y_norm <= 1.0):
            print(f"[INVALID] gaze coords out of range [0,1]: {name}")
            missing += 1
            continue

        out_path = args.output_dir / f"{stem}.txt"
        out_path.write_text(f"{gaze_x_norm:.6f} {gaze_y_norm:.6f}\n", encoding="utf-8")
        saved += 1

    print(f"Done. saved={saved}, missing={missing}, duplicates={duplicates}")


if __name__ == "__main__":
    main()
