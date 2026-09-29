"""Strict YOLO detection-label parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float


def parse_yolo_label(path: Path) -> tuple[Box, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing label: {path}")
    boxes: list[Box] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from error
        if class_id not in (0, 1):
            raise ValueError(f"{path}:{line_number}: unsupported class {class_id}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: values must be finite")
        x, y, width, height = values
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{path}:{line_number}: normalized value out of range")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number}: box dimensions must be positive")
        boxes.append(Box(class_id, x, y, width, height))
    return tuple(boxes)
