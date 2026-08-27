from __future__ import annotations

from pathlib import Path

import pytest

from yolo_combine.joint_config import JointExperimentConfig
from yolo_combine.pose_stages import pose_stage
from yolo_combine.stage_policy import JOINT_STAGES


@pytest.mark.parametrize("architecture", ["full35", "partial75"])
def test_variant_configs_lock_the_same_patience_policy(architecture: str) -> None:
    config = JointExperimentConfig.load(
        Path(f"variants/{architecture}/configs/joint.yaml")
    )
    policy = config.j2_plateau_policy

    assert pose_stage("p1").trainer_overrides()["patience"] == 10
    assert pose_stage("p2").trainer_overrides()["patience"] == 12
    assert JOINT_STAGES["j2"].patience == 17
    assert policy.monitor == "bittrue_joint_score"
    assert policy.patience == 17
    assert policy.recovery_after == 8
    assert policy.min_delta == pytest.approx(1.0e-4)
    assert policy.lr_factor == pytest.approx(0.5)
    assert policy.max_reductions == 1
    assert policy.adjust_momentum is False
    assert config.as_dict()["j2_plateau"]["patience"] == 17
    assert config.stages == ("j0", "j1", "j2")
    assert config.enable_j3 is False
    assert config.detect_batch_size == 128
    assert config.detect_microbatch_size == 64
    assert config.pose_batch_size == 16
    assert config.detect_weight == pytest.approx(1.0)
    assert config.pose_weight == pytest.approx(0.25)
    assert config.warmup_epochs == 1

    resolved_stages = config.as_dict()["stage_policies"]
    assert resolved_stages["j0"]["task_mode"] == "pose"
    assert resolved_stages["j0"]["epochs"] == 8
    assert resolved_stages["j1"]["epochs"] == 20
    assert resolved_stages["j1"]["patience"] == 8
    assert resolved_stages["j2"]["epochs"] == 80
    assert resolved_stages["j3"]["epochs"] == 20
