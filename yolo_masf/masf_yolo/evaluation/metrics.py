"""Baseball-specific metric subset aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ObjectObservation:
    short_side: float
    aspect_ratio: float
    matched: bool


BallObservation = ObjectObservation


def translate_predictions_to_letterbox(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    *,
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for image in ground_truth["images"]:
        name = Path(image["file_name"]).name
        if name in by_name:
            raise ValueError(f"COCO image basenames must be unique: {name}")
        by_name[name] = image
    translated: list[dict[str, Any]] = []
    for prediction in predictions:
        name = prediction.get("file_name")
        if name not in by_name:
            raise ValueError(f"prediction image is absent from ground truth: {name}")
        image = by_name[name]
        source_width = image["source_width"]
        source_height = image["source_height"]
        scale = min(imgsz / source_width, imgsz / source_height)
        pad_x = (imgsz - source_width * scale) / 2
        pad_y = (imgsz - source_height * scale) / 2
        x, y, width, height = prediction["bbox"]
        raw_category_id = int(prediction["category_id"])
        if raw_category_id < 1:
            raise ValueError(
                f"Ultralytics JSON category IDs must be 1-based, got {raw_category_id}"
            )
        translated.append(
            {
                "image_id": image["id"],
                "category_id": raw_category_id - 1,
                "bbox": [x * scale + pad_x, y * scale + pad_y, width * scale, height * scale],
                "score": float(prediction["score"]),
            }
        )
    return translated


def _iou(left: list[float], right: list[float]) -> float:
    left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
    right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def _match_category(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    category_id: int,
    iou_threshold: float = 0.5,
) -> tuple[list[ObjectObservation], int, int]:
    annotations = [
        annotation for annotation in ground_truth["annotations"]
        if int(annotation["category_id"]) == category_id
    ]
    by_image: dict[int, list[int]] = {}
    for index, annotation in enumerate(annotations):
        by_image.setdefault(int(annotation["image_id"]), []).append(index)
    matched: set[int] = set()
    category_predictions = sorted(
        (
            prediction
            for prediction in predictions
            if int(prediction["category_id"]) == category_id
        ),
        key=lambda prediction: float(prediction["score"]),
        reverse=True,
    )
    matched_predictions = 0
    for prediction in category_predictions:
        candidates = [
            index for index in by_image.get(int(prediction["image_id"]), []) if index not in matched
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda index: _iou(prediction["bbox"], annotations[index]["bbox"]))
        if _iou(prediction["bbox"], annotations[best]["bbox"]) >= iou_threshold:
            matched.add(best)
            matched_predictions += 1
    observations: list[ObjectObservation] = []
    for index, annotation in enumerate(annotations):
        width, height = annotation["bbox"][2:]
        observations.append(
            ObjectObservation(
                short_side=float(annotation.get("short_side", min(width, height))),
                aspect_ratio=max(width, height) / min(width, height),
                matched=index in matched,
            )
        )
    return observations, len(category_predictions), matched_predictions


def ball_observations_from_coco(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> list[BallObservation]:
    observations, _, _ = _match_category(
        ground_truth,
        predictions,
        category_id=0,
        iou_threshold=iou_threshold,
    )
    return observations


def _result(observations: list[BallObservation]) -> dict[str, int | float | None]:
    count = len(observations)
    return {
        "gt_count": count,
        "recall": sum(observation.matched for observation in observations) / count if count else None,
    }


def summarize_ball_subsets(
    observations: list[BallObservation],
) -> dict[str, dict[str, int | float | None]]:
    return {
        "tiny": _result([observation for observation in observations if observation.short_side < 8]),
        "small": _result(
            [observation for observation in observations if 8 <= observation.short_side <= 16]
        ),
        "large": _result([observation for observation in observations if observation.short_side > 16]),
        "blur_proxy": _result(
            [observation for observation in observations if observation.aspect_ratio > 2]
        ),
    }


def summarize_class_diagnostics(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    category_id: int,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    observations, prediction_count, true_positive_count = _match_category(
        ground_truth,
        predictions,
        category_id=category_id,
        iou_threshold=iou_threshold,
    )
    gt_count = len(observations)
    return {
        "gt_count": gt_count,
        "prediction_count": prediction_count,
        "true_positive_count": true_positive_count,
        "missed_count": gt_count - true_positive_count,
        "false_positive_count": prediction_count - true_positive_count,
        "precision": (
            true_positive_count / prediction_count if prediction_count else None
        ),
        "recall": true_positive_count / gt_count if gt_count else None,
        "subsets": summarize_ball_subsets(observations),
    }
