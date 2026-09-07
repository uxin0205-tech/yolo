from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_quantize import UniformWeightSpec, WeightRegionAssignment
from yolo_quantize.policy_search import (
    CombinedPolicyCandidate,
    CombinedPolicyChain,
    Full35CombinedPolicySearchPlan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate(
    policy_id: str,
    predecessor_id: str,
    regions: tuple[str, ...],
) -> CombinedPolicyCandidate:
    return CombinedPolicyCandidate(
        policy_id=policy_id,
        chain_id="accuracy",
        predecessor_id=predecessor_id,
        assignments=tuple(
            WeightRegionAssignment(region, UniformWeightSpec(8)) for region in regions
        ),
    )


def test_combined_policy_chain_requires_strict_monotonic_region_prefix() -> None:
    first = _candidate("a2", "masf", ("masf", "pose_one2one_predictor"))
    invalid = _candidate(
        "a3",
        "a2",
        ("masf", "detect_one2one_predictor", "pose_one2one_predictor"),
    )

    with pytest.raises(ValueError, match="strict ordered extension"):
        CombinedPolicyChain(
            chain_id="accuracy",
            seed_candidate_id="masf",
            seed_regions=("masf",),
            candidates=(first, invalid),
        )


def test_combined_policy_chain_stops_after_first_failed_predecessor() -> None:
    first = _candidate("a2", "masf", ("masf", "pose_one2one_predictor"))
    second = _candidate(
        "a3",
        "a2",
        ("masf", "pose_one2one_predictor", "detect_one2one_predictor"),
    )
    chain = CombinedPolicyChain(
        chain_id="accuracy",
        seed_candidate_id="masf",
        seed_regions=("masf",),
        candidates=(first, second),
    )

    assert chain.runnable_candidate_ids({"masf": True}) == ("a2",)
    assert chain.runnable_candidate_ids({"masf": True, "a2": True}) == ("a3",)
    assert chain.runnable_candidate_ids({"masf": True, "a2": False}) == ()
    assert chain.blocked_candidate_ids({"masf": True, "a2": False}) == ("a3",)


def test_combined_policy_format_identity_keeps_w8_gate_applicable() -> None:
    candidate = _candidate("a2", "masf", ("masf", "pose_one2one_predictor"))

    assert candidate.regions == ("masf", "pose_one2one_predictor")
    assert candidate.format_id == "mixed-w8-mse_grid_v1"


def test_reviewed_combined_policy_plan_pins_green_isolated_regions() -> None:
    plan = Full35CombinedPolicySearchPlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v5-qsilu-cumulative-w8-policy-search-v1.yaml"
    )

    assert plan.gate_spec.metric_family == "map50"
    assert plan.gate_spec.total_max_drop == 0.015
    assert plan.base_plan.matched_policy_id == "qsilu_pq--lsq-plus-a8"
    assert tuple(chain.chain_id for chain in plan.chains) == (
        "accuracy",
        "compression",
    )
    assert len(plan.candidates) == 15
    assert "backbone_attention_safe" not in {
        region for candidate in plan.candidates for region in candidate.regions
    }


def test_reviewed_plan_rejects_unapproved_isolated_region(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT
        / "configs/experiments/v5-qsilu-cumulative-w8-policy-search-v1.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["chains"][0]["candidates"][0]["regions"].append("backbone_attention_safe")
    changed = tmp_path / "invalid.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="isolated W8 gate"):
        Full35CombinedPolicySearchPlan.from_yaml(changed)


def test_poly_shift_combined_policy_plan_uses_matched_parent_activation() -> None:
    plan = Full35CombinedPolicySearchPlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v17-poly-shift-cumulative-w8-policy-search-v1.yaml"
    )

    assert plan.base_plan.cell.parent.activation == "poly_shift"
    assert plan.base_plan.matched_policy_id == "poly_shift--lsq-plus-a8"
    assert len(plan.candidates) == 16
    assert all(
        assignment.spec.bits == 8
        for candidate in plan.candidates
        for assignment in candidate.assignments
    )
