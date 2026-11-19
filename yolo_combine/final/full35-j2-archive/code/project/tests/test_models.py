from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from yolo_combine.contracts import Task
from yolo_combine.models import RoutedDualModel, SharedDualHeadModel
from yolo_combine.source import BuiltTaskModels


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
def test_shared_model_keeps_one_trunk_and_both_task_contracts(
    full35_models: BuiltTaskModels,
    shared_model: SharedDualHeadModel,
):
    routed = RoutedDualModel(full35_models.detect, full35_models.pose)
    routed_parameters = sum(parameter.numel() for parameter in routed.parameters())
    shared_parameters = sum(parameter.numel() for parameter in shared_model.parameters())

    assert routed_parameters == 45_580_762
    assert shared_parameters == 26_529_701
    assert shared_parameters < 0.6 * routed_parameters
    assert routed.contract()["model_kind"] == "routed_dual"
    assert shared_model.contract() == {
        "model_kind": "shared_dual_head",
        "head_inputs": [16, 19, 22],
        "detect_nc": 80,
        "pose_nc": 2,
        "kpt_shape": [2, 3],
        "detect_names": full35_models.detect.names,
        "pose_names": {0: "ball", 1: "bat"},
    }


@pytest.mark.integration
@pytest.mark.gpu
def test_shared_forward_is_identical_to_independent_task_graphs(
    full35_models: BuiltTaskModels,
    shared_model: SharedDualHeadModel,
    cuda_device: torch.device,
):
    detect = copy.deepcopy(full35_models.detect).to(cuda_device).eval()
    pose = copy.deepcopy(full35_models.pose).to(cuda_device).eval()
    shared = copy.deepcopy(shared_model).to(cuda_device).eval()
    images = torch.rand(1, 3, 128, 128, device=cuda_device)

    with torch.inference_mode():
        expected_detect = detect(images)
        expected_pose = pose(images)
        outputs = shared(images, tasks="both")
        detect_only = shared(images, tasks=Task.DETECT)

    assert outputs.keys() == {"detect", "pose"}
    assert detect_only.keys() == {"detect"}
    _assert_nested_equal(outputs["detect"], expected_detect)
    _assert_nested_equal(outputs["pose"], expected_pose)
