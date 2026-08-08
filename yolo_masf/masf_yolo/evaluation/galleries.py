"""Verifiable false-positive records and letterbox contact sheets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from masf_yolo.artifacts.io import atomic_write_json


def _iou(left: list[float], right: list[float]) -> float:
    left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
    right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
    width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = width * height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def false_positive_records(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    image_names = {int(image["id"]): image["file_name"] for image in ground_truth["images"]}
    annotations: dict[tuple[int, int], list[list[float]]] = {}
    for annotation in ground_truth["annotations"]:
        key = (int(annotation["image_id"]), int(annotation["category_id"]))
        annotations.setdefault(key, []).append(annotation["bbox"])
    records: list[dict[str, Any]] = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        targets = annotations.get((image_id, category_id), [])
        if any(_iou(prediction["bbox"], target) >= iou_threshold for target in targets):
            continue
        records.append(
            {
                "image_id": image_id,
                "file_name": image_names[image_id],
                "category_id": category_id,
                "bbox": prediction["bbox"],
                "score": float(prediction["score"]),
            }
        )
    return sorted(records, key=lambda record: record["score"], reverse=True)


def _letterbox(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source = source.convert("RGB")
        scale = min(640 / source.width, 640 / source.height)
        resized = source.resize(
            (round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS
        )
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(resized, ((640 - resized.width) // 2, (640 - resized.height) // 2))
    return canvas


def write_false_positive_gallery(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    output_dir: Path,
    *,
    maximum: int = 100,
) -> list[dict[str, Any]]:
    records = false_positive_records(ground_truth, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "false_positives.json", records)
    selected = records[:maximum]
    if not selected:
        sheet = Image.new("RGB", (512, 96), "white")
        ImageDraw.Draw(sheet).text((20, 36), "No false positives at the evaluation threshold", fill="black")
        sheet.save(output_dir / "false_positives.png")
        return records
    tile_size = 256
    columns = 4
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, record in enumerate(selected):
        image = _letterbox(Path(record["file_name"]))
        draw = ImageDraw.Draw(image)
        x, y, width, height = record["bbox"]
        color = "red" if record["category_id"] == 0 else "orange"
        draw.rectangle((x, y, x + width, y + height), outline=color, width=5)
        draw.text((x, max(0, y - 14)), f"{record['category_id']} {record['score']:.3f}", fill=color)
        tile = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    sheet.save(output_dir / "false_positives.png")
    return records
