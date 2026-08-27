from __future__ import annotations

from pathlib import Path

import pytest

from yolo_combine.pose_stages import pose_stage


def test_p1_is_locked_to_head_only_formal_defaults():
    stage = pose_stage("p1")
    overrides = stage.trainer_overrides()

    assert stage.epochs == 17
    assert stage.imgsz == 640
    assert stage.batch == 128
    assert stage.fraction == 1.0
    assert stage.val and stage.plots
    assert overrides["freeze"] == list(range(23))
    assert overrides["optimizer"] == "MuSGD"
    assert overrides["patience"] == 10
    assert overrides["fliplr"] == 0.0
    assert stage.validate_transition(None) is None


def test_staged_checkpoint_transition_is_fail_closed(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()

    with pytest.raises(ValueError, match="must start fresh"):
        pose_stage("p1").validate_transition(checkpoint)
    with pytest.raises(ValueError, match="requires --initial-checkpoint"):
        pose_stage("p2").validate_transition(None)
    assert pose_stage("p2").validate_transition(checkpoint) == checkpoint.resolve()


def test_stage_freeze_progression_and_unknown_name():
    p2 = pose_stage("p2")
    p3 = pose_stage("p3")
    assert (p2.epochs, p2.batch) == (22, 64)
    assert (p3.epochs, p3.batch) == (100, 32)
    assert round(p2.trainer_overrides()["nbs"] / p2.batch) == 2
    assert round(p3.trainer_overrides()["nbs"] / p3.batch) == 4
    assert p2.trainer_overrides()["freeze"] == 11
    assert p2.trainer_overrides()["patience"] == 12
    assert p3.trainer_overrides()["freeze"] is None
    assert p3.trainer_overrides()["patience"] == 20

    with pytest.raises(ValueError, match="unknown Pose stage"):
        pose_stage("not-a-stage")
