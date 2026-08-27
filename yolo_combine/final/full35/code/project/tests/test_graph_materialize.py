from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.source import BuiltTaskModels, SourceBundle
from yolo_combine.xnor import XNORExecutionConfig, install_xnor_backend


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.integration
def test_float_validation_materialization_is_output_exact_on_cpu(
    source_bundle: SourceBundle,
    full35_models: BuiltTaskModels,
) -> None:
    install_xnor_backend(XNORExecutionConfig(token_tile=32))
    shared, _ = assemble_graph_shared_model(
        copy.deepcopy(full35_models.detect),
        copy.deepcopy(full35_models.pose),
    )
    shared.eval()
    materialized = build_graph_validation_models(
        shared,
        source_bundle,
        kind="float",
    )
    images = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(11))

    with torch.inference_mode():
        expected = shared(images, task="both")
        detect = materialized.detect(images)
        pose = materialized.pose(images)

    assert materialized.detect_report.complete
    assert materialized.pose_report.complete
    _assert_nested_equal(detect, expected["detect"])
    _assert_nested_equal(pose, expected["pose"])


@pytest.mark.integration
def test_bittrue_validation_materialization_has_complete_named_mapping(
    source_bundle: SourceBundle,
    full35_models: BuiltTaskModels,
) -> None:
    install_xnor_backend(XNORExecutionConfig(token_tile=32))
    shared, _ = assemble_graph_shared_model(
        copy.deepcopy(full35_models.detect),
        copy.deepcopy(full35_models.pose),
    )
    materialized = build_graph_validation_models(
        shared,
        source_bundle,
        kind="bittrue",
    )

    assert materialized.detect_report.complete
    assert materialized.pose_report.complete
    assert materialized.detect_report.missing_tensors == ()
    assert materialized.pose_report.shape_mismatches == ()
