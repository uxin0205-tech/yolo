from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from yolo_combine.materialize import build_validation_models
from yolo_combine.models import SharedDualHeadModel
from yolo_combine.source import SourceBundle


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.integration
@pytest.mark.gpu
def test_materialized_official_task_models_are_identical_to_shared_outputs(
    source_bundle: SourceBundle,
    shared_model: SharedDualHeadModel,
    cuda_device: torch.device,
):
    shared = copy.deepcopy(shared_model).to(cuda_device).eval()
    materialized = build_validation_models(shared, source_bundle)
    detect = materialized.detect.to(cuda_device).eval()
    pose = materialized.pose.to(cuda_device).eval()
    images = torch.rand(1, 3, 128, 128, device=cuda_device)

    with torch.inference_mode():
        shared_outputs = shared(images)
        detect_output = detect(images)
        pose_output = pose(images)

    assert materialized.detect_report.complete
    assert materialized.pose_report.complete
    assert materialized.detect_report.compatible_tensors == len(
        materialized.detect.state_dict()
    )
    assert materialized.pose_report.compatible_tensors == 998
    _assert_nested_equal(detect_output, shared_outputs["detect"])
    _assert_nested_equal(pose_output, shared_outputs["pose"])
