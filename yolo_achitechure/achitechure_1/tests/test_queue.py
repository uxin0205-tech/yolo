from __future__ import annotations

import hashlib
import json
from pathlib import Path

import achitechure_1.queue as queue_module
from achitechure_1.queue import (
    PHASE_C_HEADROOM_BYTES,
    PHASE_C_PROFILE_PEAK_BYTES,
    QueueJournal,
    _descriptor_checkpoint,
    _phase_run_id,
    phase_c_has_capacity,
    _fixed_queue_settings,
    _train_phase,
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


def test_fraction03_queue_records_ram_vram_and_amp_contract() -> None:
    settings = _fixed_queue_settings(
        6,
        training_batch=16,
        nbs=16,
        phase_c_training_batch=8,
        phase_c_nbs=16,
        validation_batch=8,
        minimum_available_ram_bytes=8 << 30,
        minimum_free_vram_bytes=12 << 30,
        fraction=0.3,
        amp=True,
    )

    assert settings["minimum_available_ram_bytes"] == 8 << 30
    assert settings["minimum_free_vram_bytes"] == 12 << 30
    assert settings["fraction"] == 0.3 and settings["amp"] is True
    assert settings["batch"] == 16 and settings["gradient_accumulation"] is False
    assert settings["phase_c_batch"] == settings["validation_batch"] == 8
    assert settings["phase_c_effective_batch"] == 16
    assert settings["phase_c_gradient_accumulation"] is True


def test_continuation_descriptor_binds_variant_boundary_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "float-best.pt"
    checkpoint.write_bytes(b"accepted-a2")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    descriptor = tmp_path / "candidate.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "full35",
                "boundary": "accepted_after_phase_a2",
                "float_checkpoint": {"path": checkpoint.name, "sha256": checksum},
            }
        ),
        encoding="utf-8",
    )

    resolved, payload = _descriptor_checkpoint(descriptor, "full35")

    assert resolved == checkpoint.resolve()
    assert payload["boundary"] == "accepted_after_phase_a2"


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


def test_fraction_queue_separates_phase_b_and_phase_c_batches(tmp_path: Path, monkeypatch) -> None:
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    calls: list[tuple[object, ...]] = []
    verified: list[dict[str, object]] = []
    monkeypatch.setattr(queue_module, "_run_worker", lambda *args: calls.append(args))
    monkeypatch.setattr(
        queue_module,
        "_verify_completed_run",
        lambda *args, **kwargs: verified.append(kwargs) or (tmp_path / "best.pt"),
    )
    journal = QueueJournal(
        tmp_path / "queue/state.json",
        run_tag="split-batches",
        workers=6,
        training_batch=16,
        nbs=16,
        phase_c_training_batch=8,
        phase_c_nbs=16,
        validation_batch=8,
        fraction=0.3,
        amp=True,
        phase_c_patience=7,
    )

    _train_phase(
        tmp_path,
        journal,
        architecture="full35",
        phase="b",
        parent=parent,
        run_tag="split-batches",
    )
    _train_phase(
        tmp_path,
        journal,
        architecture="full35",
        phase="c",
        parent=parent,
        run_tag="split-batches",
    )

    b_arguments = calls[0][3:]
    c_arguments = calls[1][3:]
    assert b_arguments[b_arguments.index("--batch") + 1] == "16"
    assert b_arguments[b_arguments.index("--nbs") + 1] == "16"
    assert c_arguments[c_arguments.index("--batch") + 1] == "8"
    assert c_arguments[c_arguments.index("--nbs") + 1] == "16"
    assert b_arguments[b_arguments.index("--fraction") + 1] == "0.3"
    assert c_arguments[c_arguments.index("--fraction") + 1] == "0.3"
    assert "--patience" not in b_arguments
    assert c_arguments[c_arguments.index("--patience") + 1] == "7"
    assert verified[0]["expected_batch"] == 16
    assert verified[1]["expected_batch"] == 8
    assert all(item["expected_validation_batch"] == 8 for item in verified)
    assert verified[0]["expected_patience"] is None
    assert verified[1]["expected_patience"] == 7


def test_train_phase_resumes_valid_incomplete_run(tmp_path: Path, monkeypatch) -> None:
    run_tag = "interrupted"
    run_id = _phase_run_id("full35", "b", run_tag)
    run_dir = tmp_path / "artifacts/runs" / run_id
    (run_dir / "ultralytics/weights").mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "ultralytics/weights/last.pt").write_bytes(b"checkpoint")
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    expected = run_dir / "ultralytics/weights/best.pt"
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(queue_module, "_run_worker", lambda *args: calls.append(args))
    monkeypatch.setattr(queue_module, "_verify_completed_run", lambda *args, **kwargs: expected)
    journal = QueueJournal(
        tmp_path / "queue/state.json",
        run_tag=run_tag,
        workers=6,
        training_batch=16,
        nbs=16,
        phase_c_training_batch=8,
        phase_c_nbs=16,
        validation_batch=8,
        fraction=0.3,
        amp=True,
    )

    actual_run_id, checkpoint = _train_phase(
        tmp_path,
        journal,
        architecture="full35",
        phase="b",
        parent=parent,
        run_tag=run_tag,
    )

    assert actual_run_id == run_id and checkpoint == expected
    assert "--resume-incomplete" in calls[0]
    assert calls[0][calls[0].index("--batch") + 1] == "16"
