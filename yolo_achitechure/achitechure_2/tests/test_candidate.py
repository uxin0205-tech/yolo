from __future__ import annotations

import torch

from achitechure_2.candidate import (
    build_candidate,
    graph_snapshot,
    resolve_candidate_matrix,
)
from achitechure_2.intake import CandidateRegion
from achitechure_2.lite_c3k2 import LiteC3k2


def shared_region() -> CandidateRegion:
    return CandidateRegion(
        region_id="shared",
        role="shared",
        tasks=("detect", "pose"),
        module_paths=("trunk.layers.0", "trunk.layers.1"),
        head_paths=("detect_head", "pose_head"),
    )


def test_candidate_matrix_is_resolved_from_fusion_kind_and_regions() -> None:
    shared = resolve_candidate_matrix("shared_dual_head", (shared_region(),))
    assert tuple(item.resolved_id for item in shared) == ("C0", "C1", "C2", "C3")

    routed = resolve_candidate_matrix(
        "routed_dual",
        (
            CandidateRegion(
                "detect",
                "detect_specific",
                ("detect",),
                ("detect_model.model.6",),
                ("detect_model.model.23",),
            ),
            CandidateRegion(
                "pose",
                "pose_specific",
                ("pose",),
                ("pose_model.model.6",),
                ("pose_model.model.23",),
            ),
        ),
    )
    assert tuple(item.resolved_id for item in routed) == (
        "C0",
        "D-C1",
        "D-C2",
        "D-C3",
        "P-C1",
        "P-C2",
        "P-C3",
    )

    partial = resolve_candidate_matrix(
        "partial_shared",
        (
            shared_region(),
            CandidateRegion(
                "detect",
                "detect_specific",
                ("detect",),
                ("detect_branch.0",),
                ("detect_head",),
            ),
        ),
    )
    assert tuple(item.resolved_id for item in partial) == (
        "C0",
        "S-C1",
        "S-C2",
        "S-C3",
        "D-C1",
        "D-C2",
        "D-C3",
    )


def test_c0_handoff_is_an_independent_exact_copy(combined_parent) -> None:
    resolved = resolve_candidate_matrix("shared_dual_head", (shared_region(),))[0]
    candidate, report = build_candidate(combined_parent, resolved, seed=0)

    assert candidate is not combined_parent
    assert report.resolved_id == "C0"
    assert report.changed_module_paths == ()
    assert report.transfer.shape_mismatch == ()
    assert report.parent_unchanged
    assert all(
        torch.equal(value, candidate.state_dict()[name])
        for name, value in combined_parent.state_dict().items()
    )


def test_c1_c2_c3_each_change_one_factor_in_one_region(combined_parent) -> None:
    expected = {
        "C1": (0.375, 2, "3x3_3x3", False),
        "C2": (0.5, 1, "3x3_3x3", False),
        "C3": (0.5, 2, "1x1_3x3", False),
    }
    matrix = resolve_candidate_matrix("shared_dual_head", (shared_region(),))
    parent_state = {name: value.clone() for name, value in combined_parent.state_dict().items()}

    for resolved in matrix[1:]:
        model, report = build_candidate(combined_parent, resolved, seed=17)
        assert report.changed_module_paths == shared_region().module_paths
        assert report.changed_fields == ({"C1": "e", "C2": "inner_n", "C3": "kernel_mode"}[resolved.resolved_id],)
        assert report.region_id == "shared"
        assert report.parent_unchanged
        assert all(isinstance(model.get_submodule(path), LiteC3k2) for path in report.changed_module_paths)
        factors = {
            (
                item["e"],
                item["inner_n"],
                item["kernel_mode"],
                item["use_rep"],
            )
            for item in report.module_contracts
        }
        assert factors == {expected[resolved.resolved_id]}
        assert type(model.trunk.layers[2]) is type(combined_parent.trunk.layers[2])
        assert type(model.detect_head) is type(combined_parent.detect_head)
        assert type(model.pose_head) is type(combined_parent.pose_head)
    assert all(
        torch.equal(value, combined_parent.state_dict()[name]) for name, value in parent_state.items()
    )


def test_transfer_is_deterministic_and_snapshot_uses_resolved_paths(combined_parent) -> None:
    resolved = resolve_candidate_matrix("shared_dual_head", (shared_region(),))[1]
    first, first_report = build_candidate(combined_parent, resolved, seed=23)
    second, second_report = build_candidate(combined_parent, resolved, seed=23)

    assert first_report.transfer.shape_mismatch
    mismatched = {entry.name for entry in first_report.transfer.shape_mismatch}
    assert all(torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in mismatched)
    assert first_report.transfer == second_report.transfer
    snapshot = graph_snapshot(first, first_report)
    assert snapshot["standalone_loadable"] is False
    assert snapshot["builder"] == "achitechure_2"
    assert snapshot["resolved_candidate"] == "C1"
    assert snapshot["candidate_region"]["module_paths"] == [
        "trunk.layers.0",
        "trunk.layers.1",
    ]
    assert snapshot["model_contract"]["detect_nc"] == 80
