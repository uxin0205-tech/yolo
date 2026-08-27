from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_combine.gpu_queue import (
    GPUIdlePolicy,
    GPUProcess,
    GPUSnapshot,
    QueuedGPUJob,
)


class SequenceProbe:
    def __init__(self, snapshots: list[GPUSnapshot]) -> None:
        self.snapshots = iter(snapshots)

    def snapshot(self, index: int) -> GPUSnapshot:
        snapshot = next(self.snapshots)
        assert snapshot.index == index
        return snapshot


def _snapshot(
    *,
    free: int = 31_500,
    utilization: int = 0,
    processes: tuple[GPUProcess, ...] = (),
) -> GPUSnapshot:
    return GPUSnapshot(
        index=0,
        uuid="GPU-test",
        total_memory_mib=32_607,
        free_memory_mib=free,
        utilization_percent=utilization,
        compute_processes=processes,
    )


def test_idle_policy_requires_memory_utilization_and_no_compute_process() -> None:
    policy = GPUIdlePolicy(
        min_free_memory_mib=30_000,
        max_utilization_percent=10,
        stable_polls=3,
        poll_seconds=1,
    )

    assert policy.accepts(_snapshot())
    assert not policy.accepts(_snapshot(free=29_999))
    assert not policy.accepts(_snapshot(utilization=11))
    assert not policy.accepts(
        _snapshot(processes=(GPUProcess(123, "trainer", 1),))
    )


def test_queue_waits_for_stable_polls_then_launches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy = _snapshot(processes=(GPUProcess(123, "trainer", 20_000),))
    idle = _snapshot()
    probe = SequenceProbe([busy, idle, idle])
    monkeypatch.setattr("yolo_combine.gpu_queue.time.sleep", lambda _: None)
    marker = tmp_path / "launched.txt"
    job = QueuedGPUJob(
        queue_id="test-job",
        gpu_index=0,
        policy=GPUIdlePolicy(stable_polls=2, poll_seconds=1),
        command=("/usr/bin/touch", str(marker)),
        working_directory=tmp_path,
        state_path=tmp_path / "state.json",
        log_path=tmp_path / "job.log",
        lock_path=tmp_path / "job.lock",
        probe=probe,
    )

    assert job.run() == 0
    assert marker.is_file()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["return_code"] == 0
    assert state["poll_count"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("min_free_memory_mib", 0),
        ("max_utilization_percent", 101),
        ("stable_polls", 0),
        ("poll_seconds", 0),
    ),
)
def test_idle_policy_rejects_invalid_thresholds(field: str, value: int) -> None:
    arguments = {
        "min_free_memory_mib": 30_000,
        "max_utilization_percent": 10,
        "stable_polls": 3,
        "poll_seconds": 60,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        GPUIdlePolicy(**arguments)
