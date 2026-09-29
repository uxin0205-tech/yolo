from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_quantize.qat_queue import (
    QueueRecorder,
    _is_expected_sham_command,
    _resolve_qat_resume,
    _validate_qat_completion,
)


def test_queue_matches_only_reviewed_sham_command() -> None:
    command = (
        "/venv/bin/python",
        "-m",
        "yolo_quantize.qat_runtime",
        "--plan",
        "configs/experiments/v19.yaml",
        "--arm",
        "sham",
        "--device",
        "0",
        "--execute-reviewed-plan",
    )

    assert _is_expected_sham_command(command, "v19.yaml")
    assert not _is_expected_sham_command(
        tuple("qat" if token == "sham" else token for token in command),
        "v19.yaml",
    )
    assert not _is_expected_sham_command(command[:-1], "v19.yaml")
    assert not _is_expected_sham_command(command, "different.yaml")


def test_queue_recorder_deduplicates_identical_transitions(tmp_path: Path) -> None:
    recorder = QueueRecorder(tmp_path, "fixture")

    recorder.transition("waiting", pid=12)
    recorder.transition("waiting", pid=12)
    recorder.transition("ready", pid=12)

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [record["kind"] for record in records] == ["waiting", "ready"]
    assert json.loads((tmp_path / "status.json").read_text())["kind"] == "ready"


def test_qat_resume_requires_explicit_opt_in_and_in_scope_last_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "candidate-qat"
    checkpoint = run_dir / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"resume")

    with pytest.raises(FileExistsError, match="without completion evidence"):
        _resolve_qat_resume(run_dir, allowed=False)
    assert _resolve_qat_resume(run_dir, allowed=True) == checkpoint.resolve()
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="last checkpoint"):
        _resolve_qat_resume(run_dir, allowed=True)


def test_qat_completion_requires_hash_pinned_best_joint(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "candidate-qat"
    checkpoint = run_dir / "checkpoints" / "best_joint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    plan = SimpleNamespace(
        run_root=run_root,
        config_sha256="plan-sha",
        candidate_id="candidate",
        training=SimpleNamespace(epochs=15),
    )
    runtime = SimpleNamespace(
        plan=plan,
        run_name=lambda arm: f"candidate-{arm}",
    )
    manifest = {
        "plan_sha256": "plan-sha",
        "candidate_id": "candidate",
        "arm": "qat",
        "run_name": "candidate-qat",
        "formal_validation": False,
        "completed_stages": ["j3"],
        "epochs_completed": 15,
        "checkpoint_paths": {"best_joint": str(checkpoint)},
        "checkpoint_sha256": {"best_joint": digest},
    }
    (run_dir / "qat-experiment.json").write_text(json.dumps(manifest))

    evidence = _validate_qat_completion(runtime)

    assert evidence["checkpoint_sha256"] == digest
    runtime.effective_patience = 5
    manifest["epochs_completed"] = 11
    manifest["early_stop"] = {
        "values": {
            "patience": 5,
            "stale_epochs": 5,
            "should_stop": 1,
        }
    }
    (run_dir / "qat-experiment.json").write_text(json.dumps(manifest))
    early_evidence = _validate_qat_completion(runtime)
    assert early_evidence["epochs_completed"] == 11

    manifest["checkpoint_sha256"]["best_joint"] = "drifted"
    (run_dir / "qat-experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        _validate_qat_completion(runtime)
