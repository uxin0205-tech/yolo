from __future__ import annotations

import pytest

from masf_yolo.evaluation.metrics import (
    BallObservation,
    ball_observations_from_coco,
    summarize_ball_subsets,
    translate_predictions_to_letterbox,
)
from masf_yolo.evaluation.runner import evaluate_coco


def test_ball_size_and_blur_recall_keep_exact_ground_truth_counts() -> None:
    observations = [
        BallObservation(short_side=7.9, aspect_ratio=1.0, matched=True),
        BallObservation(short_side=4.0, aspect_ratio=3.0, matched=False),
        BallObservation(short_side=8.0, aspect_ratio=1.5, matched=True),
        BallObservation(short_side=16.0, aspect_ratio=2.1, matched=True),
        BallObservation(short_side=16.1, aspect_ratio=1.0, matched=False),
    ]

    metrics = summarize_ball_subsets(observations)

    assert metrics["tiny"] == {"gt_count": 2, "recall": 0.5}
    assert metrics["small"] == {"gt_count": 2, "recall": 1.0}
    assert metrics["large"] == {"gt_count": 1, "recall": 0.0}
    assert metrics["blur_proxy"] == {"gt_count": 2, "recall": 0.5}


def test_unsupported_subset_uses_null_metric_and_retains_zero_count() -> None:
    metrics = summarize_ball_subsets([])

    assert metrics == {
        "tiny": {"gt_count": 0, "recall": None},
        "small": {"gt_count": 0, "recall": None},
        "large": {"gt_count": 0, "recall": None},
        "blur_proxy": {"gt_count": 0, "recall": None},
    }


def test_ultralytics_original_coordinates_translate_to_locked_letterbox_space() -> None:
    ground_truth = {
        "images": [
            {
                "id": 7,
                "file_name": "/dataset/frame.jpg",
                "width": 640,
                "height": 640,
                "source_width": 1280,
                "source_height": 720,
            }
        ]
    }
    predictions = [
        {
            "image_id": "frame",
            "file_name": "frame.jpg",
            # Ultralytics save_json uses 1-based category IDs for custom data.
            "category_id": 1,
            "bbox": [576.0, 288.0, 128.0, 144.0],
            "score": 0.9,
        }
    ]

    translated = translate_predictions_to_letterbox(predictions, ground_truth)

    assert translated[0]["image_id"] == 7
    assert translated[0]["category_id"] == 0
    assert translated[0]["bbox"] == pytest.approx([288.0, 284.0, 64.0, 72.0])


def test_ultralytics_one_based_categories_map_to_zero_based_ball_and_bat() -> None:
    ground_truth = {
        "images": [
            {
                "id": 1,
                "file_name": "frame.jpg",
                "source_width": 640,
                "source_height": 640,
            }
        ],
        "categories": [{"id": 0, "name": "ball"}, {"id": 1, "name": "bat"}],
    }
    predictions = [
        {"file_name": "frame.jpg", "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9},
        {"file_name": "frame.jpg", "category_id": 2, "bbox": [5, 6, 7, 8], "score": 0.8},
    ]

    translated = translate_predictions_to_letterbox(predictions, ground_truth)

    assert [prediction["category_id"] for prediction in translated] == [0, 1]


def test_ball_observations_match_each_prediction_at_most_once() -> None:
    ground_truth = {
        "images": [{"id": 1}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "short_side": 10, "blur_proxy": False},
            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [1, 1, 10, 10], "short_side": 10, "blur_proxy": True},
        ],
    }
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}
    ]

    observations = ball_observations_from_coco(ground_truth, predictions, iou_threshold=0.5)

    assert len(observations) == 2
    assert sum(observation.matched for observation in observations) == 1


def test_faster_coco_eval_reports_perfect_literal_detection() -> None:
    ground_truth = {
        "images": [{"id": 1, "file_name": "frame.jpg", "width": 640, "height": 640}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": []},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [100, 100, 40, 80], "area": 3200, "iscrowd": 0, "segmentation": []},
        ],
        "categories": [{"id": 0, "name": "ball"}, {"id": 1, "name": "bat"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 40, 80], "score": 0.99},
    ]

    metrics = evaluate_coco(ground_truth, predictions)

    assert metrics["map50_95"] == pytest.approx(1.0)
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["per_class"]["ball"]["ap"] == pytest.approx(1.0)
    assert metrics["per_class"]["bat"]["ap"] == pytest.approx(1.0)


def test_faster_coco_eval_handles_zero_predictions_without_crashing() -> None:
    ground_truth = {
        "images": [{"id": 1, "file_name": "frame.jpg", "width": 640, "height": 640}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": []},
        ],
        "categories": [{"id": 0, "name": "ball"}, {"id": 1, "name": "bat"}],
    }

    metrics = evaluate_coco(ground_truth, [])

    assert metrics["map50_95"] == pytest.approx(0.0)
    assert metrics["per_class"]["ball"]["ap"] == pytest.approx(0.0)
    assert metrics["per_class"]["bat"]["ap"] is None
