import pytest
from torch import nn

from yolo_combine.freezing import (
    InheritedFreezeGuard,
    enforce_inherited_eval,
    inherited_modules,
)
from yolo_combine.source import SourceBundle


def test_inherited_scope_fails_closed_when_graph_has_no_expected_modules():
    with pytest.raises(ValueError, match="exactly three"):
        inherited_modules(nn.Linear(2, 2))


@pytest.mark.integration
def test_full35_shared_inherited_scope_is_frozen_and_mode_guarded(
    source_bundle: SourceBundle,
):
    model = source_bundle.build_shared().train()
    guard = InheritedFreezeGuard.capture(model)

    assert guard.paths == (
        "trunk.layers.10.m.0.attn",
        "trunk.layers.16.p3_masf",
        "trunk.layers.22.m.0.1.attn",
    )
    selected = inherited_modules(model)
    assert all(
        not parameter.requires_grad
        for _, module in selected
        for parameter in module.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.parameters())
    guard.assert_unchanged(model)

    model.train()
    with pytest.raises(AssertionError, match="training mode"):
        guard.assert_unchanged(model)
    enforce_inherited_eval(model)
    guard.assert_unchanged(model)
