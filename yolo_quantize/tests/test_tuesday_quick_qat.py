from __future__ import annotations

from pathlib import Path

from yolo_quantize.tuesday_quick_qat import (
    CANDIDATE_ID,
    build_qat_plan_payload,
    materialize_qat_plan,
)


def test_qat_plan_keeps_nonselected_layers_float_and_promotes_sd4_to_lsq() -> None:
    payload = build_qat_plan_payload()
    policy = payload["weight_policy"]
    assignments = policy["assignments"]

    assert policy["candidate_id"] == CANDIDATE_ID
    assert len(policy["float_regions"]) == 10
    assert len(assignments) == 3
    assert sum(len(item["paths"]) for item in assignments) == 8
    assert all(item["format"] == {
        "family": "ls_sd4",
        "scale_method": "optimal_scaled_codebook",
    } for item in assignments)


def test_qat_plan_uses_short_matched_budget_and_official_logical_batch() -> None:
    payload = build_qat_plan_payload()
    training = payload["training"]

    assert training["epochs"] == 4
    assert training["patience"] == 5
    assert training["warmup_epochs"] == 1
    assert training["scale_only_epochs"] == 1
    assert training["progressive_start_epoch"] == 0
    assert training["progressive_full_epoch"] == 1
    assert training["detect_logical_batch"] == 128
    assert training["detect_microbatch"] == 16
    assert training["pose_batch"] == 16
    assert training["added_noise"] is False
    assert payload["execution_authorization"]["arms"] == ["sham", "qat"]


def test_qat_plan_round_trips_through_strict_parser(tmp_path: Path) -> None:
    parsed = materialize_qat_plan(tmp_path / "qat-plan.yaml")

    assert parsed.candidate_id == CANDIDATE_ID
    assert parsed.training.epochs == 4
    assert parsed.training.detect_logical_batch == 128
    assert len(parsed.float_regions) == 10
    assert sum(len(item.paths) for item in parsed.assignments) == 8
    assert {item.spec.format_id for item in parsed.assignments} == {"fixed-sd4"}
