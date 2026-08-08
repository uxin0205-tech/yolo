from __future__ import annotations

from pathlib import Path
import json
import subprocess

import pytest

from masf_yolo.training.resume import PermanentTrainingError, TransientTrainingError
from masf_yolo.training.worker import (
    TrainingWorkerRequest,
    build_worker_command,
    launch_worker_process,
)


def test_training_worker_request_round_trips_and_rejects_unknown_keys() -> None:
    request = TrainingWorkerRequest(
        config_path=Path("/work/configs/static-phase1.yaml"),
        stage="formal_m7",
        variant_id="M7",
        profile={"epochs": 100, "batch": 16},
        resume_path=None,
    )

    assert TrainingWorkerRequest.from_dict(request.to_dict()) == request
    raw = request.to_dict() | {"mystery": True}
    with pytest.raises(ValueError, match="unknown training worker keys"):
        TrainingWorkerRequest.from_dict(raw)


def test_training_worker_command_is_a_fresh_python_process() -> None:
    command = build_worker_command(
        python=Path("/work/.venv/bin/python"),
        request_path=Path("/work/request.json"),
        output_path=Path("/work/result.json"),
    )

    assert command == [
        "/work/.venv/bin/python",
        "-m",
        "masf_yolo.training.worker",
        "--request",
        "/work/request.json",
        "--output",
        "/work/result.json",
    ]


def test_parent_launches_worker_and_reads_atomic_result(tmp_path: Path, monkeypatch) -> None:
    request = TrainingWorkerRequest(Path("config.yaml"), "formal_m7", "M7", {"epochs": 100}, None)

    def fake_run(command, **kwargs):
        output = Path(command[-1])
        output.write_text(json.dumps({"best": "/x/best.pt", "last": "/x/last.pt"}))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("masf_yolo.training.worker.subprocess.run", fake_run)

    report = launch_worker_process(
        request,
        request_path=tmp_path / "request.json",
        output_path=tmp_path / "result.json",
        python=Path("/work/python"),
    )

    assert report["best"] == "/x/best.pt"
    assert json.loads((tmp_path / "request.json").read_text())["variant_id"] == "M7"


def test_parent_classifies_killed_worker_as_transient(tmp_path: Path, monkeypatch) -> None:
    request = TrainingWorkerRequest(Path("config.yaml"), "formal_m7", "M7", {"epochs": 100}, None)
    monkeypatch.setattr(
        "masf_yolo.training.worker.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, -9, stdout="", stderr="killed"),
    )

    with pytest.raises(TransientTrainingError, match="-9"):
        launch_worker_process(
            request,
            request_path=tmp_path / "request.json",
            output_path=tmp_path / "result.json",
            python=Path("/work/python"),
        )


def test_parent_rejects_success_without_result_manifest(tmp_path: Path, monkeypatch) -> None:
    request = TrainingWorkerRequest(Path("config.yaml"), "formal_m7", "M7", {"epochs": 100}, None)
    monkeypatch.setattr(
        "masf_yolo.training.worker.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    with pytest.raises(PermanentTrainingError, match="result manifest"):
        launch_worker_process(
            request,
            request_path=tmp_path / "request.json",
            output_path=tmp_path / "result.json",
            python=Path("/work/python"),
        )
