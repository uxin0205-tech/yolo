from __future__ import annotations

from pathlib import Path

import pytest

from yolo_combine.formal_training import FormalJointTrainingSession
from yolo_combine.joint_config import JointExperimentConfig


def _config() -> JointExperimentConfig:
    return JointExperimentConfig.load(
        Path("variants/full35/configs/joint.yaml")
    )


def test_j3_runtime_microbatch_preserves_logical_detect_exposure() -> None:
    session = FormalJointTrainingSession(
        _config(),
        device="cpu",
        run_name="j3-b32-contract",
        detect_microbatch_size=32,
    )

    assert session.detect_microbatch_size == 32
    assert session.detect_microbatches_per_logical_batch == 4
    assert session.detect_microbatches_per_macro == 8
    assert (
        session.detect_microbatch_size
        * session.detect_microbatches_per_macro
        == 256
    )

    resolved = session._resolved_config()
    runtime = resolved["runtime_overrides"]
    assert runtime["detect_train_logical"] == 128
    assert runtime["detect_train_physical_microbatch"] == 32
    assert runtime["detect_physical_microbatches_per_macro"] == 8


def test_j3_runtime_microbatch_must_divide_logical_batch() -> None:
    with pytest.raises(ValueError, match="must divide logical Detect batch"):
        FormalJointTrainingSession(
            _config(),
            device="cpu",
            run_name="j3-invalid-batch",
            detect_microbatch_size=48,
        )
