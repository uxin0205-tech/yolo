from __future__ import annotations

import pytest
import torch

from achitechure_2.freezing import FrozenStateGuard, apply_frozen_scope, enforce_frozen_eval
from achitechure_2.graph import inspect_graph


def test_graph_contract_and_missing_masf_rejection(toy_parent) -> None:
    report = inspect_graph(toy_parent)
    assert report.detect_inputs == (16, 19, 22)
    assert report.strides == (8, 16, 32)
    assert report.end2end
    del toy_parent.model[16].p3_masf
    with pytest.raises(ValueError, match="must contain P3 MASF"):
        inspect_graph(toy_parent)


def test_frozen_scope_excludes_parameters_and_bn_state(toy_parent) -> None:
    paths = apply_frozen_scope(toy_parent)
    assert paths == ("model.10.m.0.attn", "model.16.p3_masf", "model.22.m.0.1.attn")
    frozen_ids = {
        id(parameter) for path in paths for parameter in toy_parent.get_submodule(path).parameters()
    }
    assert all(
        not parameter.requires_grad for parameter in toy_parent.parameters() if id(parameter) in frozen_ids
    )
    assert all(
        parameter.requires_grad for parameter in toy_parent.parameters() if id(parameter) not in frozen_ids
    )

    guard = FrozenStateGuard.capture(toy_parent)
    toy_parent.train()
    enforce_frozen_eval(toy_parent)
    guard.assert_unchanged(toy_parent)
    with torch.no_grad():
        toy_parent.model[16].p3_masf.alpha.add_(1)
    with pytest.raises(AssertionError, match="changed"):
        guard.assert_unchanged(toy_parent)
