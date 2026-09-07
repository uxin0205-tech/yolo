from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_quantize.qat_plan import Full35QATPlan
from yolo_quantize.weight_quantization import (
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    UniformWeightSpec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v1_qat_pilot_is_paired_poly_shift_all_w8_search_training() -> None:
    plan = Full35QATPlan.from_yaml(
        PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
    )

    assert plan.activation.activation == "poly_shift"
    assert plan.candidate_id == "accuracy-a10-add-backbone-attention-w8"
    assert plan.activation.bits == 8
    assert plan.arms == ("sham", "qat")
    assert len(plan.assignments) == 10
    assert all(
        isinstance(assignment.spec, UniformWeightSpec) and assignment.spec.bits == 8
        for assignment in plan.assignments
    )
    assert plan.training.epochs == 15
    assert plan.training.scale_only_epochs == 5
    assert plan.training.progressive_start_epoch == 0
    assert plan.training.progressive_full_epoch == 5
    assert plan.training.detect_logical_batch == 128
    assert plan.training.detect_microbatch == 16
    assert plan.training.pose_batch == 16
    assert plan.optimizer.name == "AdamW"
    assert plan.optimizer.qparam_lr_ratio == 1.0
    assert plan.gates.map50_max_drop == 0.015
    assert plan.gates.map50_95_max_drop == 0.04
    assert plan.data.pose_search.name == "pose-search.yaml"
    assert plan.expected_master_catalog["totals"] == {
        "modules": 251,
        "weight_elements": 26451392,
        "deployment_modules": 148,
        "deployment_weight_elements": 22571840,
        "training_only_modules": 99,
        "training_only_weight_elements": 3748480,
        "protected_modules": 4,
        "protected_weight_elements": 131072,
    }
    assert plan.expected_deployment_catalog["totals"] == {
        "modules": 152,
        "weight_elements": 22702912,
        "deployment_modules": 148,
        "deployment_weight_elements": 22571840,
        "training_only_modules": 0,
        "training_only_weight_elements": 0,
        "protected_modules": 4,
        "protected_weight_elements": 131072,
    }
    assert plan.formal_validation is False
    assert plan.monitoring.mode == "low_token"


def test_qat_plan_rejects_a_weight_policy_that_omits_a_deployment_region(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    payload["weight_policy"]["assignments"].pop()
    broken = tmp_path / "missing-region.yaml"
    broken.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly cover deployment regions"):
        Full35QATPlan.from_yaml(broken)


def test_qat_plan_allows_hash_pinned_defaults_plus_path_specific_ls_sd4(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    payload["weight_policy"]["assignments"].append(
        {
            "region": "neck_attention_safe",
            "paths": ["graph.model.22.m.0.1.attn.qkv.v.conv"],
            "format": {
                "family": "ls_sd4",
                "scale_method": "optimal_scaled_codebook",
            },
        }
    )
    routed = tmp_path / "path-routed-ls-sd4.yaml"
    routed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan = Full35QATPlan.from_yaml(routed)

    assert len(plan.assignments) == 11
    assert plan.assignments[-1].region == "neck_attention_safe"
    assert plan.assignments[-1].paths == ("graph.model.22.m.0.1.attn.qkv.v.conv",)
    assert isinstance(plan.assignments[-1].spec, FixedSD4WeightSpec)


def test_qat_plan_parses_exact_and_filterwise_ternary_path_overrides(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    payload["weight_policy"]["assignments"].extend(
        [
            {
                "region": "neck_attention_safe",
                "paths": ["graph.model.22.m.0.0.attn.qkv.v.conv"],
                "format": {
                    "family": "exact_scaled_ternary",
                    "scale_method": "optimal_scaled_codebook",
                },
            },
            {
                "region": "neck_attention_safe",
                "paths": ["graph.model.22.m.0.1.attn.qkv.v.conv"],
                "format": {
                    "family": "twn_filterwise",
                    "threshold_multiplier": 0.75,
                },
            },
        ]
    )
    routed = tmp_path / "path-routed-ternary.yaml"
    routed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan = Full35QATPlan.from_yaml(routed)

    assert isinstance(plan.assignments[-2].spec, ExactTernaryWeightSpec)
    assert isinstance(plan.assignments[-1].spec, FilterwiseTWNWeightSpec)
    assert plan.assignments[-1].spec.threshold_multiplier == 0.75


def test_qat_plan_can_explicitly_keep_one_deployment_region_float(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    payload["weight_policy"]["assignments"] = [
        assignment
        for assignment in payload["weight_policy"]["assignments"]
        if assignment["region"] != "backbone_attention_safe"
    ]
    payload["weight_policy"]["float_regions"] = ["backbone_attention_safe"]
    routed = tmp_path / "nine-quantized-one-float.yaml"
    routed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan = Full35QATPlan.from_yaml(routed)

    assert plan.float_regions == ("backbone_attention_safe",)
    assert len(plan.assignments) == 9


def test_qat_plan_accepts_hash_pinned_external_control_for_continuation(
    tmp_path: Path,
) -> None:
    source = PROJECT_ROOT / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1/short-qat-mixed-final-v1/generated/qat-plan.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    completion = PROJECT_ROOT / "artifacts/runs/qat/v35-qsilu-mixed-final-short-v1/v35-qsilu-mixed-lssd4-w6-w4-short-qat-v1-sham-seed1/qat-experiment.json"
    metrics = PROJECT_ROOT / "artifacts/runs/qat/v35-qsilu-mixed-final-short-v1/v35-qsilu-mixed-lssd4-w6-w4-short-qat-v1-sham-seed1/validation/epoch-0003/bittrue/metrics.json"
    payload["execution_authorization"]["arms"] = ["qat"]
    payload["execution_authorization"]["external_control"] = {
        "mode": "v35_external_sham_reference",
        "plan": {"path": str(source), "sha256": digest(source)},
        "completion": {"path": str(completion), "sha256": digest(completion)},
        "metrics": {"path": str(metrics), "sha256": digest(metrics)},
    }
    candidate = tmp_path / "external-control-plan.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan = Full35QATPlan.from_yaml(candidate)

    assert plan.arms == ("qat",)
    assert plan.external_control is not None
    assert plan.external_control["mode"] == "v35_external_sham_reference"


def test_v25_qat_plan_matches_the_nine_region_quality_recovery_candidate() -> None:
    plan = Full35QATPlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v25-poly-shift-nine-region-quality-qat-pilot-v1.yaml"
    )

    assert plan.candidate_id == "polyshift-nine-region-quality"
    assert plan.float_regions == ("backbone_attention_safe",)
    assert {
        assignment.region: assignment.spec.bits
        for assignment in plan.assignments
        if not assignment.paths
    } == {
        "backbone_early": 8,
        "backbone_deep": 8,
        "neck": 8,
        "masf": 4,
        "neck_attention_safe": 4,
        "detect_one2one_tower": 8,
        "detect_one2one_predictor": 5,
        "pose_one2one_tower": 4,
        "pose_one2one_predictor": 8,
    }
    assert plan.training.epochs == 15
    assert plan.training.patience == 5
    assert plan.gates.map50_max_drop == 0.015
    assert plan.gates.map50_95_max_drop == 0.04


def test_qat_plan_parses_hash_pinned_v19_full_resume_warm_start(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    payload["warm_start"] = {
        "locked_parent": {
            "path": "artifacts/manifests/v19-epoch5-locked-parent-v1.yaml",
            "sha256": (
                "6e2034bc6a84ee5f1dac2a22d38afbcdec9763b0fc22e9333b67bb0c1dbd48a7"
            ),
        },
        "checkpoint_role": "full_resume",
        "state_key": "ema_state",
        "reset_path_override_quantizers": True,
        "optimizer_state": "fresh",
    }
    routed = tmp_path / "v19-warm-start.yaml"
    routed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan = Full35QATPlan.from_yaml(routed)

    assert plan.warm_start is not None
    assert plan.warm_start.parent_id == "v19-poly-shift-a8-all-w8-qat-epoch5"
    assert plan.warm_start.selected_epoch == 5
    assert plan.warm_start.checkpoint.name == "best_joint.pt"
    assert plan.warm_start.state_key == "ema_state"
    assert plan.warm_start.reset_path_override_quantizers is True
    assert plan.warm_start.load_optimizer_state is False
