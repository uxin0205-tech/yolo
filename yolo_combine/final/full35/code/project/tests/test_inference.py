from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from yolo_combine.inference import SharedDualPredictor


class FakeSharedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = 0
        self.detect_names = {0: "person"}
        self.pose_names = {0: "ball", 1: "bat"}
        self.detect_head = SimpleNamespace(end2end=True)
        self.pose_head = SimpleNamespace(end2end=True, kpt_shape=(2, 3))

    def forward(self, images, task=None):
        self.calls += 1
        batch = images.shape[0]
        detect = torch.tensor(
            [[[8.0, 8.0, 48.0, 48.0, 0.9, 0.0]]],
            device=images.device,
        ).repeat(batch, 1, 1)
        pose = torch.tensor(
            [[[10.0, 10.0, 50.0, 50.0, 0.8, 1.0, 16.0, 16.0, 0.9, 40.0, 40.0, 0.8]]],
            device=images.device,
        ).repeat(batch, 1, 1)
        if task == "detect":
            return {"detect": detect}
        if task == "pose":
            return {"pose": pose}
        return {"detect": detect, "pose": pose}


def test_both_inference_runs_shared_model_once_and_returns_two_result_sets() -> None:
    model = FakeSharedModel()
    predictor = SharedDualPredictor(
        model,
        imgsz=64,
        conf=0.25,
        iou=0.7,
        max_det=10,
    )
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    results = predictor.predict(image, task="both", paths=["frame.jpg"])

    assert model.calls == 1
    assert set(results) == {"detect", "pose"}
    assert len(results["detect"]) == len(results["pose"]) == 1
    assert results["detect"][0].names == {0: "person"}
    assert results["pose"][0].names == {0: "ball", 1: "bat"}
    assert results["pose"][0].keypoints.shape == (1, 2, 3)


def test_single_task_inference_does_not_compute_other_head() -> None:
    model = FakeSharedModel()
    predictor = SharedDualPredictor(model, imgsz=64)

    output = predictor.predict(
        np.zeros((64, 64, 3), dtype=np.uint8),
        task="detect",
    )

    assert set(output) == {"detect"}
    assert model.calls == 1

