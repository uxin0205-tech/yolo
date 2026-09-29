from __future__ import annotations

from ultralytics.models.yolo.detect.val import DetectionValidator

from achitechure_1.evaluation import COCO2017Validator


def test_coco_validator_forces_sparse_official_category_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        DetectionValidator,
        "init_metrics",
        lambda self, model: setattr(self, "class_map", list(range(1, 81))),
    )
    validator = object.__new__(COCO2017Validator)

    validator.init_metrics(object())

    assert len(validator.class_map) == 80
    assert validator.class_map[:12] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
