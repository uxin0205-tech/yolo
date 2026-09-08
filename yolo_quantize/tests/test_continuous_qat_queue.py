import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_continuous_qat_queue")


def result(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    return {
        "plan_sha256": "plan",
        "arm": "qat",
        "epochs_completed": 5,
        "epochs_planned": 5,
        "checkpoint_paths": {"best": str(checkpoint)},
        "checkpoint_sha256": {
            "best": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        },
    }


def test_complete_result_is_verified(queue, tmp_path):
    queue.validate_result(result(tmp_path), SimpleNamespace(config_sha256="plan"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs_completed", 3),
        ("plan_sha256", "wrong"),
        ("arm", "sham"),
        ("checkpoint_paths", {}),
    ],
)
def test_incomplete_or_wrong_result_rejected(queue, tmp_path, field, value):
    payload = result(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError):
        queue.validate_result(payload, SimpleNamespace(config_sha256="plan"))


def test_checkpoint_hash_drift_rejected(queue, tmp_path):
    payload = result(tmp_path)
    Path(payload["checkpoint_paths"]["best"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        queue.validate_result(payload, SimpleNamespace(config_sha256="plan"))


def test_child_failure_is_not_ignored(queue, monkeypatch, tmp_path):
    monkeypatch.setattr(
        queue.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(pid=1, wait=lambda timeout: 2),
    )
    monkeypatch.setattr(queue, "write_status", lambda **k: None)
    with pytest.raises(RuntimeError, match="退出 2"):
        queue.wait_child(["fake"], tmp_path / "child.log", {})
