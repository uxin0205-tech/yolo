from __future__ import annotations

from pathlib import Path

from yolo_quantize.mixed_layer_qat import (
    CANDIDATE_ID,
    build_qat_plan_payload,
    materialize_qat_plan,
)


def test_mixed_qat_promotes_the_eleven_locked_paths() -> None:
    payload = build_qat_plan_payload()
    policy = payload["weight_policy"]
    assignments = policy["assignments"]
    paths = [path for item in assignments for path in item["paths"]]

    assert policy["candidate_id"] == CANDIDATE_ID
    assert len(policy["float_regions"]) == 10
    assert len(assignments) == 6
    assert len(paths) == len(set(paths)) == 11
    by_family = {
        "ls_sd4": sum(
            len(item["paths"])
            for item in assignments
            if item["format"]["family"] == "ls_sd4"
        ),
        "w6": sum(
            len(item["paths"])
            for item in assignments
            if item["format"]
            == {
                "family": "uniform",
                "bits": 6,
                "scale_method": "optimal_scaled_codebook",
            }
        ),
        "w4": sum(
            len(item["paths"])
            for item in assignments
            if item["format"]
            == {
                "family": "uniform",
                "bits": 4,
                "scale_method": "optimal_scaled_codebook",
            }
        ),
    }
    assert by_family == {"ls_sd4": 9, "w6": 1, "w4": 1}


def test_mixed_qat_uses_the_short_matched_training_contract() -> None:
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


def test_mixed_qat_round_trips_through_strict_parser(tmp_path: Path) -> None:
    parsed = materialize_qat_plan(tmp_path / "qat-plan.yaml")

    assert parsed.candidate_id == CANDIDATE_ID
    assert parsed.training.epochs == 4
    assert parsed.training.detect_logical_batch == 128
    assert len(parsed.float_regions) == 10
    assert sum(len(item.paths) for item in parsed.assignments) == 11
    assert {item.spec.format_id for item in parsed.assignments} == {
        "fixed-sd4",
        "w4",
        "w6",
    }
