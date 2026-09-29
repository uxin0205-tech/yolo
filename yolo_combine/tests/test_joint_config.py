from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from yolo_combine.joint_config import JointExperimentConfig


def test_full35_config_resolves_locked_training_contract() -> None:
    config = JointExperimentConfig.load(
        Path("variants/full35/configs/joint.yaml")
    )

    assert config.architecture == "full35"
    assert config.enabled is True
    assert config.pose_data.name == "pose.yaml"
    assert config.detect_batch_size == 128
    assert config.detect_microbatch_size == 64
    assert config.pose_batch_size == 16
    assert config.detect_val_batch_size == 32
    assert config.pose_val_batch_size == 16
    assert config.detect_batches_per_macro == 2
    assert config.detect_microbatches_per_logical_batch == 2
    assert config.detect_microbatches_per_macro == 4
    assert config.detect_batch_size * config.detect_batches_per_macro == 256
    assert config.reference_batch_size == 64
    assert config.optimizer == "AdamW"
    assert config.amp_max_overflow_retries == 16
    assert config.stages == ("j0", "j1", "j2")
    assert config.pose_weight == pytest.approx(0.25)
    assert config.warmup_epochs == 1
    assert config.enable_j3 is False
    assert config.xnor_token_tile == 32
    assert config.qk_ste is False
    assert config.maximum_map_drop == pytest.approx(0.08)


def test_partial75_is_isolated_and_disabled_until_user_requests_run() -> None:
    config = JointExperimentConfig.load(
        Path("variants/partial75/configs/joint.yaml")
    )

    assert config.architecture == "partial75"
    assert config.enabled is False
    assert config.stages == ("j0", "j1", "j2")
    assert config.detect_batch_size == 128
    assert config.detect_microbatch_size == 64
    assert config.pose_batch_size == 16
    assert config.pose_weight == pytest.approx(0.25)


def test_formal_preflight_refuses_missing_pose_checkpoint_and_gate_baseline(
    tmp_path: Path,
) -> None:
    config = replace(
        JointExperimentConfig.load(Path("variants/full35/configs/joint.yaml")),
        pose_checkpoint=tmp_path / "missing-pose.pt",
        baseline_metrics_path=tmp_path / "missing-baseline.json",
    )
    report = config.preflight()

    assert report.ready is False
    assert any("Pose26" in reason for reason in report.blockers)
    assert any("八項" in reason for reason in report.blockers)
