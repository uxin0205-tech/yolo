from __future__ import annotations

import json
from pathlib import Path

from achitechure_1.queue import (
    PHASE_C_HEADROOM_BYTES,
    PHASE_C_PROFILE_PEAK_BYTES,
    QueueJournal,
    _phase_run_id,
    phase_c_has_capacity,
    _fixed_queue_settings,
)


def test_phase_run_ids_are_deterministic() -> None:
    assert (
        _phase_run_id("full35", "a2", "rtx4080super-batch16-workers4-r1")
        == "a1-full35-phase-a2-rtx4080super-batch16-workers4-r1"
    )
    assert (
        _phase_run_id("partial75", "b", "rtx4080super-batch16-workers4-r1")
        == "a2-partial75-phase-b-rtx4080super-batch16-workers4-r1"
    )


def test_phase_c_capacity_requires_measured_peak_plus_headroom() -> None:
    for architecture, measured_peak in PHASE_C_PROFILE_PEAK_BYTES.items():
        threshold = measured_peak + PHASE_C_HEADROOM_BYTES
        assert not phase_c_has_capacity(architecture, threshold - 1)
        assert phase_c_has_capacity(architecture, threshold)


def test_eight_worker_queue_requires_seven_gib_available_ram() -> None:
    settings = _fixed_queue_settings(8)
    assert settings["maximum_concurrent_data_workers"] == 8
    assert settings["in_training_validation_workers"] == 0
    assert settings["minimum_available_ram_bytes"] == 7 << 30


def test_phase_c_recovery_queue_records_batch_eight_accumulate_two() -> None:
    settings = _fixed_queue_settings(6, training_batch=8, nbs=16, validation_batch=16)

    assert settings["batch"] == 8
    assert settings["nbs"] == settings["effective_batch"] == 16
    assert settings["gradient_accumulation"] is True
    assert settings["validation_batch"] == 16


def test_queue_journal_writes_atomic_state(tmp_path: Path) -> None:
    state_path = tmp_path / "queue" / "state.json"
    journal = QueueJournal(state_path, run_tag="tag", source_a1_run="a1")

    journal.update("running", current_job="train:a2", value=1)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["current_job"] == "train:a2"
    assert payload["value"] == 1
    assert not state_path.with_suffix(".tmp").exists()


def test_phase_c_queue_journal_exposes_runtime_settings(tmp_path: Path) -> None:
    journal = QueueJournal(
        tmp_path / "queue" / "state.json",
        run_tag="phase-c",
        workers=6,
        training_batch=8,
        nbs=16,
        validation_batch=16,
    )

    assert journal.training_batch == 8
    assert journal.nbs == 16
    assert journal.validation_batch == 16
    assert journal.workers == 6
