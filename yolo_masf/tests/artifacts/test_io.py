from __future__ import annotations

import json
from pathlib import Path

import pytest

from masf_yolo.artifacts.io import PipelineLock, atomic_write_json


def test_failed_atomic_write_preserves_last_good_state(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    atomic_write_json(state, {"stage": "audit"})

    with pytest.raises(TypeError):
        atomic_write_json(state, {"invalid": object()})

    assert json.loads(state.read_text()) == {"stage": "audit"}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_only_one_pipeline_owner_can_hold_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "pipeline.lock"

    with PipelineLock(lock_path):
        with pytest.raises(RuntimeError, match="already locked"):
            with PipelineLock(lock_path):
                pass

    with PipelineLock(lock_path):
        assert lock_path.read_text().strip().isdigit()
