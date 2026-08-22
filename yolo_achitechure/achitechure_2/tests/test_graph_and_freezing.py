from __future__ import annotations

import pytest
import torch

from achitechure_2.candidate import build_candidate, resolve_candidate_matrix
from achitechure_2.freezing import (
    FrozenStateGuard,
    apply_frozen_paths,
    enforce_frozen_eval,
)
from achitechure_2.graph import inspect_fusion_graph
from achitechure_2.intake import CandidateRegion


def _region() -> CandidateRegion:
    return CandidateRegion(
        "shared",
        "shared",
        ("detect", "pose"),
        ("trunk.layers.0", "trunk.layers.1"),
        ("detect_head", "pose_head"),
    )


def test_fusion_graph_uses_handoff_paths_instead_of_fixed_layer_indices(
    combined_parent,
) -> None:
    report = inspect_fusion_graph(
        combined_parent,
        fusion_kind="shared_dual_head",
        candidate_regions=(_region(),),
        protected_module_paths=("trunk.layers.2", "detect_head", "pose_head"),
        frozen_module_paths=("trunk.layers.2",),
    )

    assert report.fusion_kind == "shared_dual_head"
    assert report.tasks == ("detect", "pose", "both")
    assert tuple(item.path for item in report.candidate_modules) == (
        "trunk.layers.0",
        "trunk.layers.1",
    )
    assert all(item.c3k2 is not None for item in report.candidate_modules)
    assert report.model_contract["detect_nc"] == 80
    assert report.model_contract["pose_nc"] == 2

    with pytest.raises(ValueError, match="不存在"):
        inspect_fusion_graph(
            combined_parent,
            fusion_kind="shared_dual_head",
            candidate_regions=(_region(),),
            protected_module_paths=("missing.path", "detect_head", "pose_head"),
            frozen_module_paths=("missing.path",),
        )


def test_candidate_graph_reports_the_single_resolved_factor(combined_parent) -> None:
    resolved = resolve_candidate_matrix("shared_dual_head", (_region(),))[1]
    candidate, build = build_candidate(combined_parent, resolved, seed=0)
    report = inspect_fusion_graph(
        candidate,
        fusion_kind="shared_dual_head",
        candidate_regions=(_region(),),
        protected_module_paths=("trunk.layers.2", "detect_head", "pose_head"),
        frozen_module_paths=("trunk.layers.2",),
        expected_candidate=build,
    )

    assert report.resolved_candidate_id == "C1"
    assert {item.c3k2.e for item in report.candidate_modules} == {0.375}
    assert report.changed_fields == ("e",)


def test_dynamic_frozen_paths_protect_parameters_buffers_and_mode(combined_parent) -> None:
    paths = apply_frozen_paths(combined_parent, ("trunk.layers.2",))
    assert paths == ("trunk.layers.2",)
    assert all(
        not parameter.requires_grad
        for parameter in combined_parent.get_submodule("trunk.layers.2").parameters()
    )
    assert combined_parent.get_submodule("trunk.layers.2").training is False
    assert combined_parent.detect_head.weight.requires_grad

    guard = FrozenStateGuard.capture(
        combined_parent,
        ("trunk.layers.2",),
        reset_trainable=False,
    )
    combined_parent.train()
    enforce_frozen_eval(combined_parent, guard.paths)
    guard.assert_unchanged(combined_parent)

    with torch.no_grad():
        combined_parent.trunk.layers[2].running_mean.add_(1)
    with pytest.raises(AssertionError, match="改變"):
        guard.assert_unchanged(combined_parent)


def test_frozen_paths_reject_parent_child_overlap(combined_parent) -> None:
    with pytest.raises(ValueError, match="重疊"):
        apply_frozen_paths(
            combined_parent,
            ("trunk", "trunk.layers.2"),
        )
