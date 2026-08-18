from __future__ import annotations

import torch

from achitechure_2.candidate import build_candidate
from achitechure_2.config import PROTECTED_C3K2_LAYERS, TARGET_LAYERS
from achitechure_2.graph import inspect_graph
from achitechure_2.lite_c3k2 import LiteC3k2


def test_c0_is_an_independent_unchanged_copy(toy_parent) -> None:
    candidate, report = build_candidate(toy_parent, "C0")
    assert candidate is not toy_parent
    assert all(
        torch.equal(value, candidate.state_dict()[name]) for name, value in toy_parent.state_dict().items()
    )
    assert report.graph.masf_variant == "full35"
    assert report.transfer.shape_mismatch == ()


def test_each_main_candidate_changes_only_one_factor(toy_parent) -> None:
    expected = {
        "C1": (0.375, 2, "3x3_3x3", False),
        "C2": (0.5, 1, "3x3_3x3", False),
        "C3": (0.5, 2, "1x1_3x3", False),
    }
    parent_state = {name: value.clone() for name, value in toy_parent.state_dict().items()}
    for candidate_id, factors in expected.items():
        model, report = build_candidate(toy_parent, candidate_id)
        assert (
            tuple((item.e, item.inner_n, item.kernel_mode, item.use_rep) for item in report.graph.layers)
            == (factors,) * 4
        )
        assert all(isinstance(model.model[index], LiteC3k2) for index in TARGET_LAYERS)
        assert all(not isinstance(model.model[index], LiteC3k2) for index in PROTECTED_C3K2_LAYERS)
        assert report.graph.attention_paths == ("model.10.m.0.attn", "model.22.m.0.1.attn")
    assert all(torch.equal(value, toy_parent.state_dict()[name]) for name, value in parent_state.items())


def test_weight_transfer_reports_shape_mismatch_and_is_deterministic(toy_parent) -> None:
    first, first_report = build_candidate(toy_parent, "C1", seed=17)
    second, second_report = build_candidate(toy_parent, "C1", seed=17)
    assert first_report.transfer.shape_mismatch
    mismatched = {entry.name for entry in first_report.transfer.shape_mismatch}
    assert all(torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in mismatched)
    assert first_report.transfer == second_report.transfer


def test_c3_p5_only_and_r1_are_independent(toy_parent) -> None:
    p5, p5_report = build_candidate(toy_parent, "C3-P5")
    r1, r1_report = build_candidate(toy_parent, "R1")
    assert isinstance(p5.model[8], LiteC3k2)
    assert all(not isinstance(p5.model[index], LiteC3k2) for index in (6, 13, 19))
    assert {item.index for item in p5_report.graph.layers if item.class_name == "LiteC3k2"} == {8}
    assert all(item.use_rep and item.inner_n == 1 for item in r1_report.graph.layers)
    assert inspect_graph(r1).masf_variant == "full35"
