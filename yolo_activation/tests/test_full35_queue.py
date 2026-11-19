from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full35_queue as queue_module
from full35_queue import _command, load_queue, queue_status

QUEUE = PROJECT_ROOT / "training/full35/experiment-queue.yaml"


def test_full35_queue_is_serial_topological_and_bounded() -> None:
    config = load_queue(QUEUE)
    positions = {job.job_id: index for index, job in enumerate(config.jobs)}

    assert len(config.jobs) == 19
    assert config.maximum_drop == 0.015
    assert len(positions) == len(config.jobs)
    assert len({job.run_name for job in config.jobs}) == len(config.jobs)
    assert all(
        positions[dependency] < positions[job.job_id]
        for job in config.jobs
        for dependency in job.depends_on
    )
    assert config.jobs[0].job_id == "zero-shot-uniform-qsilu-pq"


def test_full35_queue_commands_preserve_policy_and_single_device() -> None:
    config = load_queue(QUEUE)
    mixed = next(job for job in config.jobs if job.job_id == "mixed-low-risk-small")
    command = _command(mixed, device="0")

    assert command.count("--region") == 3
    assert command[command.index("--phase") + 1] == "policy_search"
    assert command[command.index("--policy-id") + 1] == mixed.job_id
    assert command[command.index("--device") + 1] == "0"


def test_qsilu_recovery_uses_memory_safe_physical_microbatch() -> None:
    config = load_queue(QUEUE)
    job = next(
        value
        for value in config.jobs
        if value.job_id == "short-recovery-uniform-qsilu-pq"
    )
    command = _command(job, device="0")

    assert command[command.index("--detect-microbatch") + 1] == "16"


def test_hardswish_recovery_preserves_matched_physical_microbatch() -> None:
    config = load_queue(QUEUE)
    job = next(
        value
        for value in config.jobs
        if value.job_id == "short-recovery-uniform-hardswish"
    )
    command = _command(job, device="0")

    assert command[command.index("--detect-microbatch") + 1] == "32"


def test_poly_shift_recovery_uses_memory_safe_physical_microbatch() -> None:
    config = load_queue(QUEUE)
    job = next(
        value
        for value in config.jobs
        if value.job_id == "short-recovery-uniform-poly-shift"
    )
    command = _command(job, device="0")

    assert command[command.index("--detect-microbatch") + 1] == "16"


def test_poly_quality_recovery_uses_memory_safe_physical_microbatch() -> None:
    config = load_queue(QUEUE)
    job = next(
        value
        for value in config.jobs
        if value.job_id == "short-recovery-uniform-poly-quality"
    )
    command = _command(job, device="0")

    assert command[command.index("--detect-microbatch") + 1] == "16"


def test_full35_queue_next_job_is_qsilu_zero_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(queue_module, "_gate", lambda _config, _job: None)
    status = queue_status(load_queue(QUEUE))

    assert status["next_job"] == "zero-shot-uniform-qsilu-pq"
    assert status["counts"] == {"completed": 0, "pending": 19, "blocked": 0}
