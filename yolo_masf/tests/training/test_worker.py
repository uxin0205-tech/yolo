from __future__ import annotations

from pathlib import Path
import json
import subprocess

import pytest

import masf_yolo.training.worker as worker_module
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


def _worker_config(tmp_path: Path, monkeypatch) -> Path:
    config_path = tmp_path / "configs" / "static-phase1.yaml"
    config_path.parent.mkdir()
    config_path.write_text("unused: true\n", encoding="utf-8")
    config = type(
        "Config",
        (),
        {"values": {"artifacts_root": "artifacts", "model": {"source_weights": "source.pt"}}},
    )()
    monkeypatch.setattr("masf_yolo.cli.load_config", lambda path: config)
    return config_path


def test_sp2_b_initializes_only_from_sp2_a_best(tmp_path: Path, monkeypatch) -> None:
    config_path = _worker_config(tmp_path, monkeypatch)
    record = tmp_path / "artifacts" / "training" / "sp2_a"
    record.mkdir(parents=True)
    best = record / "best.pt"
    (record / "run.json").write_text(json.dumps({"best": str(best)}), encoding="utf-8")
    loaded = type("LoadedModel", (), {})()
    monkeypatch.setattr("ultralytics.YOLO", lambda path, task: type("Y", (), {"model": loaded})())
    monkeypatch.setattr(
        worker_module,
        "build_model",
        lambda *args, **kwargs: pytest.fail("SP2-B must not rebuild from B1-B"),
    )

    model, transfer = worker_module._initial_model(
        TrainingWorkerRequest(config_path, "sp2_b", "SP2", {"epochs": 90}, None)
    )

    assert model is loaded
    assert transfer is None


def test_sp2p_a_merges_sp2_b_and_selected_partial_only(tmp_path: Path, monkeypatch) -> None:
    config_path = _worker_config(tmp_path, monkeypatch)
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "selection.json").parent.mkdir(parents=True, exist_ok=True)
    (artifact_root / "selection.json").write_text(
        json.dumps({"selected": "M3", "val_hashes": {"M2": "a", "M3": "b"}}),
        encoding="utf-8",
    )
    calls: list[tuple[str, str | None]] = []

    class FakeModel:
        def __init__(self, label: str) -> None:
            self.label = label

        def state_dict(self):
            return {"label": self.label}

    def fake_build(variant, checkpoint=None, **kwargs):
        calls.append((variant, str(checkpoint) if checkpoint else None))
        return FakeModel(variant)

    class FakeReport:
        def to_dict(self):
            return {"selected_partial": "M3"}

    transfer_calls = []

    def fake_transfer(target, sp2_state, partial_state, *, selected_partial):
        transfer_calls.append((target.label, sp2_state["label"], partial_state["label"], selected_partial))
        return FakeReport()

    monkeypatch.setattr(worker_module, "build_model", fake_build)
    monkeypatch.setattr(worker_module, "transfer_sp2p_parents", fake_transfer, raising=False)

    model, transfer = worker_module._initial_model(
        TrainingWorkerRequest(config_path, "sp2p_a", "SP2M3", {"epochs": 10}, None)
    )

    assert model.label == "SP2M3"
    assert calls == [
        ("SP2", str(artifact_root / "training" / "sp2_b" / "canonical.pt")),
        ("M3", str(artifact_root / "training" / "formal_m3" / "canonical.pt")),
        ("SP2M3", None),
    ]
    assert transfer_calls == [("SP2M3", "SP2", "M3", "M3")]
    assert transfer == {"selected_partial": "M3"}
