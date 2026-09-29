"""Coordinate conversion for exact 640-letterbox COCO annotations."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import Box


@dataclass(frozen=True, slots=True)
class LetterboxCocoBox:
    class_id: int
    bbox: tuple[float, float, float, float]
    area: float
    short_side: float
    aspect_ratio: float
    size_bin: str
    blur_proxy: bool


def box_to_letterbox_coco(
    box: Box,
    image_width: int,
    image_height: int,
    imgsz: int = 640,
) -> LetterboxCocoBox:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(imgsz / image_width, imgsz / image_height)
    resized_width = image_width * scale
    resized_height = image_height * scale
    pad_x = (imgsz - resized_width) / 2
    pad_y = (imgsz - resized_height) / 2
    width = box.width * image_width * scale
    height = box.height * image_height * scale
    center_x = box.x * image_width * scale + pad_x
    center_y = box.y * image_height * scale + pad_y
    x = center_x - width / 2
    y = center_y - height / 2
    short_side = min(width, height)
    if short_side < 8:
        size_bin = "tiny"
    elif short_side <= 16:
        size_bin = "small"
    else:
        size_bin = "large"
    aspect_ratio = max(width, height) / min(width, height)
    return LetterboxCocoBox(
        class_id=box.class_id,
        bbox=(x, y, width, height),
        area=width * height,
        short_side=short_side,
        aspect_ratio=aspect_ratio,
        size_bin=size_bin,
        blur_proxy=aspect_ratio > 2,
    )
