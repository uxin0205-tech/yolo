from __future__ import annotations

import pytest
import torch

from yolo_combine.joint_trainer import (
    PlateauPolicy,
    StageWarmupCosineScheduler,
)


def _policy(**overrides: object) -> PlateauPolicy:
    values: dict[str, object] = {
        "monitor": "bittrue_joint_score",
        "patience": 17,
        "recovery_after": 8,
        "lr_factor": 0.5,
        "max_reductions": 1,
        "min_delta": 1.0e-4,
        "adjust_momentum": False,
    }
    values.update(overrides)
    return PlateauPolicy(**values)  # type: ignore[arg-type]


def _optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.AdamW(
        [
            {
                "params": [parameter],
                "group_name": "head",
                "lr": 1.0e-3,
            }
        ]
    )


def _scheduler() -> StageWarmupCosineScheduler:
    return StageWarmupCosineScheduler(
        _optimizer(),
        stage="j2",
        epochs=40,
        steps_per_epoch=1,
        warmup_epochs=0,
        final_lr_factor=0.5,
        plateau_policy=_policy(),
    )


def test_j2_plateau_reduces_lr_once_and_stops_at_patience() -> None:
    scheduler = _scheduler()
    first = scheduler.observe_metric(0.5)
    assert first.improved and first.stale_epochs == 0

    for stale in range(1, 8):
        decision = scheduler.observe_metric(0.50005)
        assert decision.stale_epochs == stale
        assert not decision.recovery_applied

    recovery = scheduler.observe_metric(0.5)
    assert recovery.stale_epochs == 8
    assert recovery.recovery_applied
    assert recovery.lr_multiplier == pytest.approx(0.5)
    assert scheduler.prepare_step()["head"] == pytest.approx(5.0e-4)

    improved = scheduler.observe_metric(0.501)
    assert improved.improved and improved.stale_epochs == 0
    assert improved.lr_multiplier == pytest.approx(0.5)
    for stale in range(1, 17):
        decision = scheduler.observe_metric(0.501)
        assert decision.stale_epochs == stale
        assert not decision.should_stop
    stopped = scheduler.observe_metric(0.501)
    assert stopped.stale_epochs == 17
    assert stopped.should_stop
    assert stopped.reductions == 1


def test_j2_plateau_state_round_trip_keeps_lr_multiplier() -> None:
    scheduler = _scheduler()
    scheduler.observe_metric(0.5)
    for _ in range(8):
        scheduler.observe_metric(0.5)
    scheduler.observe_metric(0.501)
    state = scheduler.state_dict()

    replacement = _scheduler()
    replacement.load_state_dict(state)

    assert replacement.state_dict() == state
    assert replacement.prepare_step() == scheduler.prepare_step()


def test_plateau_rejects_mid_run_momentum_changes_and_non_j2_use() -> None:
    with pytest.raises(ValueError, match="momentum"):
        _policy(adjust_momentum=True)
    with pytest.raises(ValueError, match="only supported"):
        StageWarmupCosineScheduler(
            _optimizer(),
            stage="j1",
            epochs=5,
            steps_per_epoch=1,
            plateau_policy=_policy(),
        )
