from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from yolo_combine.validation import (
    ValidationSettings,
    extract_detect_metrics,
    extract_pose_metrics,
)


class FakeMetric:
    def __init__(self, ap0: float, ap1: float) -> None:
        self.p = np.array([0.8, 0.7])
        self.r = np.array([0.75, 0.65])
        self.all_ap = np.stack(
            [
                np.linspace(ap0 + 0.1, ap0 - 0.1, 10),
                np.linspace(ap1 + 0.1, ap1 - 0.1, 10),
            ]
        )
        self.ap_class_index = np.array([0, 1])
        self.nc = 2

    @property
    def mp(self):
        return float(self.p.mean())

    @property
    def mr(self):
        return float(self.r.mean())

    @property
    def map50(self):
        return float(self.all_ap[:, 0].mean())

    @property
    def map75(self):
        return float(self.all_ap[:, 5].mean())

    @property
    def map(self):
        return float(self.all_ap.mean())

    def class_result(self, index):
        return (
            self.p[index],
            self.r[index],
            self.all_ap[index, 0],
            self.all_ap[index].mean(),
        )


def test_detect_metric_extraction_includes_overall_and_person_diagnostics() -> None:
    metric = FakeMetric(0.6, 0.4)
    values = extract_detect_metrics(
        SimpleNamespace(box=metric),
        names={0: "person", 1: "other"},
    )

    assert values["coco/box/map50_95"] == metric.map
    assert values["coco/box/map75"] == metric.map75
    assert values["coco/person/box/precision"] == 0.8
    assert values["coco/person/box/recall"] == 0.75
    assert values["coco/person/box/map50_95"] == metric.all_ap[0].mean()
    assert values["coco/person/box/map75"] == metric.all_ap[0, 5]


def test_pose_metric_extraction_includes_ball_and_bat_box_and_keypoints() -> None:
    box = FakeMetric(0.55, 0.45)
    pose = FakeMetric(0.85, 0.75)
    values = extract_pose_metrics(
        SimpleNamespace(box=box, pose=pose),
        names={0: "ball", 1: "bat"},
    )

    assert values["bbat/box/map50_95"] == box.map
    assert values["bbat/pose/map50_95"] == pose.map
    assert values["bbat/ball/box/map50_95"] == box.all_ap[0].mean()
    assert values["bbat/bat/box/map50_95"] == box.all_ap[1].mean()
    assert values["bbat/ball/pose/map50_95"] == pose.all_ap[0].mean()
    assert values["bbat/bat/pose/map50_95"] == pose.all_ap[1].mean()


def test_validation_batches_are_independent_and_never_auto_doubled() -> None:
    settings = ValidationSettings(
        detect_batch_size=32,
        pose_batch_size=16,
    )
    assert settings.detect_batch_size == 32
    assert settings.pose_batch_size == 16
