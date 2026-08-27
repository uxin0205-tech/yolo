from __future__ import annotations

import copy

import pytest
import torch

from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.hardware_contract import HardwareContractGuard
from yolo_combine.source import BuiltTaskModels
from yolo_combine.xnor import XNORExecutionConfig, install_xnor_backend


@pytest.mark.integration
def test_hardware_guard_allows_tunable_bias_but_rejects_fixed_table_drift(
    full35_models: BuiltTaskModels,
) -> None:
    install_xnor_backend(XNORExecutionConfig(token_tile=32))
    model, _ = assemble_graph_shared_model(
        copy.deepcopy(full35_models.detect),
        copy.deepcopy(full35_models.pose),
    )
    guard = HardwareContractGuard.capture(model)
    modules = dict(model.named_modules())

    attention = modules["graph.model.10.m.0.attn"]
    with torch.no_grad():
        attention.bias.table_y.add_(0.25)
    guard.assert_unchanged(model)

    with torch.no_grad():
        attention.score.fixed_coefficients.add_(1.0)
    with pytest.raises(AssertionError, match="fixed_coefficients"):
        guard.assert_unchanged(model)


@pytest.mark.integration
def test_hardware_guard_includes_shared_bn_running_statistics(
    full35_models: BuiltTaskModels,
) -> None:
    install_xnor_backend(XNORExecutionConfig(token_tile=32))
    model, _ = assemble_graph_shared_model(
        copy.deepcopy(full35_models.detect),
        copy.deepcopy(full35_models.pose),
    )
    guard = HardwareContractGuard.capture(model)
    shared_bn = next(
        module
        for name, module in model.named_modules()
        if name.startswith("graph.model.0.")
        and isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )

    with torch.no_grad():
        shared_bn.running_mean.add_(1.0)
    with pytest.raises(AssertionError, match="running_mean"):
        guard.assert_unchanged(model)
