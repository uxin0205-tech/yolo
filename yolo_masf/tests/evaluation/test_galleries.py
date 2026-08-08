from __future__ import annotations

from masf_yolo.evaluation.galleries import false_positive_records


def test_false_positive_records_exclude_class_matched_detections() -> None:
    ground_truth = {
        "images": [{"id": 1, "file_name": "/data/frame.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
        ],
    }
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [100, 100, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]

    records = false_positive_records(ground_truth, predictions, iou_threshold=0.5)

    assert [record["score"] for record in records] == [0.8, 0.7]
    assert all(record["file_name"] == "/data/frame.jpg" for record in records)
