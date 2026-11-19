from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from yolo_combine.fusion_model import (
    GraphSharedDualHeadModel,
    assemble_graph_shared_model,
    audit_task_pair,
)
from yolo_combine.source import BuiltTaskModels
from yolo_combine.xnor import XNORExecutionConfig, install_xnor_backend


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
def test_task_pair_audit_covers_graph_and_yolo26_head_contract(
    full35_models: BuiltTaskModels,
) -> None:
    report = audit_task_pair(full35_models.detect, full35_models.pose)

    assert report.compatible
    assert report.shared_layers == 23
    assert report.differences == ()
    assert report.head_inputs == (16, 19, 22)
    assert report.feature_channels == (256, 512, 512)
    assert report.strides == (8.0, 16.0, 32.0)
    assert report.reg_max == 1
    assert report.end2end is True
    assert report.detect_nc == 80
    assert report.pose_nc == 2
    assert report.pose_kpt_shape == (2, 3)
    assert report.pose_flow_module == "RealNVP"


@pytest.mark.integration
def test_graph_shared_model_owns_one_trunk_and_preserves_final_metadata(
    full35_models: BuiltTaskModels,
) -> None:
    install_xnor_backend(XNORExecutionConfig(token_tile=32))
    independent_parameters = sum(
        parameter.numel()
        for model in (full35_models.detect, full35_models.pose)
        for parameter in model.parameters()
    )
    model, report = assemble_graph_shared_model(
        copy.deepcopy(full35_models.detect),
        copy.deepcopy(full35_models.pose),
    )

    assert isinstance(model, GraphSharedDualHeadModel)
    assert report.independent_parameters == independent_parameters == 45_580_762
    assert report.shared_parameters == 26_529_701
    assert report.parameter_reduction_fraction == pytest.approx(0.417963)
    assert len(model.trunk_layers) == 23
    assert model.prediction.f == [16, 19, 22]
    assert model.prediction.i == 23
    assert model.prediction.type == "ultralytics.nn.modules.head.Detect"
    assert model.prediction.detect_type == "ultralytics.nn.modules.head.Detect"
    assert model.prediction.pose_type == "ultralytics.nn.modules.head.Pose26"

    parameter_ids = [id(parameter) for parameter in model.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert not any(name.startswith("pose_model.") for name, _ in model.named_parameters())
    assert not any(name.startswith("detect_model.") for name, _ in model.named_parameters())


@pytest.mark.integration
def test_graph_shared_forward_matches_independent_models_on_cpu(
    full35_models: BuiltTaskModels,
) -> None:
    install_xnor_backend(XNORExecutionConfig(token_tile=32))
    detect = copy.deepcopy(full35_models.detect).eval()
    pose = copy.deepcopy(full35_models.pose).eval()
    shared, _ = assemble_graph_shared_model(
        copy.deepcopy(full35_models.detect),
        copy.deepcopy(full35_models.pose),
    )
    shared.eval()
    images = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(7))

    with torch.inference_mode():
        expected_detect = detect(images)
        expected_pose = pose(images)
        both = shared(images, task="both")
        detect_only = shared(images, task="detect")
        pose_only = shared(images, task="pose")

    assert both.keys() == {"detect", "pose"}
    assert detect_only.keys() == {"detect"}
    assert pose_only.keys() == {"pose"}
    _assert_nested_equal(both["detect"], expected_detect)
    _assert_nested_equal(both["pose"], expected_pose)
    _assert_nested_equal(detect_only["detect"], expected_detect)
    _assert_nested_equal(pose_only["pose"], expected_pose)
