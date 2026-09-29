from __future__ import annotations

import pytest

from masf_yolo.data.export import box_to_letterbox_coco
from masf_yolo.data.labels import Box


def test_letterbox_coco_coordinates_include_vertical_padding() -> None:
    box = Box(class_id=0, x=0.5, y=0.5, width=0.1, height=0.2)

    converted = box_to_letterbox_coco(box, image_width=1280, image_height=720, imgsz=640)

    assert converted.bbox == pytest.approx((288.0, 284.0, 64.0, 72.0))
    assert converted.area == pytest.approx(4608.0)
    assert converted.short_side == pytest.approx(64.0)
    assert converted.aspect_ratio == pytest.approx(72.0 / 64.0)


def test_ball_size_bins_use_short_side_boundaries() -> None:
    assert box_to_letterbox_coco(Box(0, 0.5, 0.5, 0.01, 0.01), 640, 640).size_bin == "tiny"
    assert box_to_letterbox_coco(Box(0, 0.5, 0.5, 0.0125, 0.0125), 640, 640).size_bin == "small"
    assert box_to_letterbox_coco(Box(0, 0.5, 0.5, 0.025, 0.025), 640, 640).size_bin == "small"
    assert box_to_letterbox_coco(Box(0, 0.5, 0.5, 0.03, 0.03), 640, 640).size_bin == "large"
