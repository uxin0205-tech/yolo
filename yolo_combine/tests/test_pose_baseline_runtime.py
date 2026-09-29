from __future__ import annotations

import copy

import pytest
import torch
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import PoseModel

from yolo_combine.baselines import MaterializedPoseTrainer


def _tiny_pose(*, nc: int = 2) -> PoseModel:
    return PoseModel(
        "yolo26n-pose.yaml",
        nc=nc,
        ch=3,
        data_kpt_shape=(2, 3),
        verbose=False,
    )


def test_pose_validation_batch_is_capped_without_changing_train_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str]] = []

    def fake_get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> tuple[int, str]:
        del self, dataset_path, rank
        calls.append((batch_size, mode))
        return batch_size, mode

    monkeypatch.setattr(PoseTrainer, "get_dataloader", fake_get_dataloader)
    trainer = object.__new__(MaterializedPoseTrainer)
    trainer.validation_batch_size = 16

    assert trainer.get_dataloader("train", batch_size=128, mode="train") == (
        128,
        "train",
    )
    assert trainer.get_dataloader("val", batch_size=256, mode="val") == (16, "val")
    assert calls == [(128, "train"), (16, "val")]


def test_materialized_resume_loads_checkpoint_ema_strictly() -> None:
    source = _tiny_pose()
    checkpoint_ema = copy.deepcopy(source)
    first = next(checkpoint_ema.parameters())
    with torch.no_grad():
        first.fill_(0.125)

    trainer = object.__new__(MaterializedPoseTrainer)
    trainer._materialized_source = source
    trainer.resume_weights_loaded = False
    loaded = trainer.get_model(weights=checkpoint_ema, verbose=False)

    assert trainer.resume_weights_loaded is True
    assert torch.equal(next(loaded.parameters()), first)
    assert trainer._materialized_source is None


def test_materialized_resume_fails_fast_on_incompatible_pose_graph() -> None:
    trainer = object.__new__(MaterializedPoseTrainer)
    trainer._materialized_source = _tiny_pose(nc=2)
    trainer.resume_weights_loaded = False

    with pytest.raises(RuntimeError, match="resume Pose graph/state mismatch"):
        trainer.get_model(weights=_tiny_pose(nc=3), verbose=False)


def test_pose_trainer_disables_ultralytics_automatic_batch_reduction() -> None:
    trainer = object.__new__(MaterializedPoseTrainer)
    trainer.physical_train_batch_size = 128
    trainer.batch_size = 128
    trainer._oom_retries = 0

    MaterializedPoseTrainer._enforce_physical_batch_contract(trainer)

    assert trainer._oom_retries == 3


def test_pose_trainer_fails_if_physical_batch_was_changed() -> None:
    trainer = object.__new__(MaterializedPoseTrainer)
    trainer.physical_train_batch_size = 128
    trainer.batch_size = 64
    trainer._oom_retries = 0

    with pytest.raises(RuntimeError, match="physical train batch changed"):
        MaterializedPoseTrainer._enforce_physical_batch_contract(trainer)
