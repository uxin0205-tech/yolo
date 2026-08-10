#!/usr/bin/env python3
"""Prepare a YOLO pose export as a YOLO detect dataset.

The source images are linked instead of copied. Source pose labels are read
as ``class x y w h kpt...`` and written as ``class x y w h``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parents[1] / "pose_dataset" / "bbt5.v1i.yolov8"
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Original YOLO pose dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for the detect dataset.",
    )
    return parser.parse_args()


def iter_images(directory: Path) -> list[Path]:
    """Return real image files, excluding Windows Zone.Identifier sidecars."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def convert_label(source: Path, destination: Path) -> int:
    """Convert one pose label file and return its object count."""
    output_rows: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 11:
            raise ValueError(
                f"{source}:{line_number}: expected 11 YOLO pose fields, got {len(fields)}"
            )

        try:
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:5]]
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: invalid class or box") from error

        if class_id < 0 or any(value < 0.0 or value > 1.0 for value in box):
            raise ValueError(f"{source}:{line_number}: class or box is out of range")

        output_rows.append(" ".join([str(class_id), *fields[1:5]]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")
    return len(output_rows)


def link_image(source: Path, destination: Path) -> None:
    """Create an idempotent relative symlink without replacing user files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve(strict=False) == source:
            return
        raise FileExistsError(f"existing symlink points elsewhere: {destination}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing file: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def prepare_split(source: Path, output: Path, split: str) -> tuple[int, int]:
    source_images = source / split / "images"
    source_labels = source / split / "labels"
    output_images = output / split / "images"
    output_labels = output / split / "labels"
    images = iter_images(source_images)
    object_count = 0

    for image in images:
        label = source_labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing label for image: {image}")
        link_image(image, output_images / image.name)
        object_count += convert_label(label, output_labels / label.name)

    return len(images), object_count


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "train" / "images").is_dir():
        raise FileNotFoundError(f"source dataset not found: {source}")

    summary: dict[str, object] = {"source": str(source), "output": str(output), "splits": {}}
    for split in ("train", "valid"):
        image_count, object_count = prepare_split(source, output, split)
        summary["splits"][split] = {"images": image_count, "objects": object_count}  # type: ignore[index]

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
