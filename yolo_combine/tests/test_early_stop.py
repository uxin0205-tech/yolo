from __future__ import annotations

import pytest

from yolo_combine.early_stop import StageEarlyStopping


def test_stage_early_stop_observes_patience_and_round_trips() -> None:
    stopper = StageEarlyStopping(stage="j1", patience=3, min_delta=0.001)

    assert stopper.observe(0.50).improved
    assert not stopper.observe(0.5005).improved
    state = stopper.state_dict()

    restored = StageEarlyStopping(stage="j1", patience=3, min_delta=0.001)
    restored.load_state_dict(state)
    second_stale = restored.observe(0.49)
    decision = restored.observe(0.48)

    assert second_stale.stale_epochs == 2
    assert not second_stale.should_stop
    assert decision.stale_epochs == 3
    assert decision.should_stop


def test_stage_early_stop_rejects_contract_drift() -> None:
    stopper = StageEarlyStopping(stage="j1", patience=8)
    stopper.observe(0.5)
    state = stopper.state_dict()
    state["patience"] = 7

    with pytest.raises(ValueError, match="contract changed"):
        StageEarlyStopping(stage="j1", patience=8).load_state_dict(state)
