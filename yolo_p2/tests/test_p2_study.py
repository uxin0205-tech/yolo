"""Focused architecture and transfer tests for the YOLO11m P2 study."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from p2_study import ROOT, worker
from p2_study.models import A2_HEAD_FREEZE, build_model, transfer_pretrained
from p2_study.run import stages
from p2_study.worker import load_config


def test_p2_models_have_expected_scales_and_unchanged_neck():
    baseline = build_model("A0")
    baseline_state = baseline.model.state_dict()
    for experiment in ("A1", "A2"):
        model = build_model(experiment)
        detect = model.model.model[-1]
        assert detect.nl == 4
        assert detect.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
        state = model.model.state_dict()
        for key, tensor in baseline_state.items():
            if not key.startswith("model.23."):
                assert key in state and state[key].shape == tensor.shape


def test_detect_transfer_is_exact_for_all_old_branches(tmp_path):
    source = build_model("A0")
    for experiment in ("A1",):
        target = build_model(experiment)
        manifest = transfer_pretrained(source, target, tmp_path / f"{experiment}.json")
        assert manifest["verified"]
        assert manifest["detect_tensors"] > 0
        for mapping in manifest["mappings"]:
            assert mapping["equal"]


def test_a2_stage_one_trains_only_new_p2_parameters():
    model = build_model("A2").model
    freeze_names = [f"model.{layer}." for layer in A2_HEAD_FREEZE] + [".dfl"]
    trainable = [name for name, _ in model.named_parameters() if not any(token in name for token in freeze_names)]
    assert trainable
    assert all(name.startswith(("model.25.", "model.26.cv2.0.", "model.26.cv3.0.")) for name in trainable)


def test_a2_runs_after_a1_and_before_b1():
    stage_ids = [item["id"] for item in stages()]
    selected = [stage_id for stage_id in stage_ids if stage_id.startswith(("formal_", "staged_"))]
    assert selected == ["formal_A1", "staged_A2_head", "formal_A2"]


def test_load_config_resolves_relative_pretrained_from_repo_root(tmp_path):
    config = {
        "study": {
            "dataset": "p2_study/coco2017.yaml",
            "dataset_root": "p2_study/data/coco2017",
            "annotation_root": "p2_study/data/annotations",
            "pretrained": "p2_study/results/weights/A0_yolo11m.pt",
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    loaded = load_config(path)

    assert loaded["study"]["pretrained"] == str((ROOT / "p2_study/results/weights/A0_yolo11m.pt").resolve())


def test_staged_training_uses_deterministic_config(monkeypatch, tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "batch_manifest.json").write_text(json.dumps({"batch": 9}))
    captured = {}

    class FakeYOLO:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint

        def train(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(worker, "ARTIFACTS", artifacts)
    monkeypatch.setattr(worker, "YOLO", FakeYOLO)
    monkeypatch.setattr(worker, "initial_checkpoint", lambda experiment, config: tmp_path / "initial.pt")
    monkeypatch.setattr(worker, "_validate_training_outputs", lambda run_dir: None)
    config = {
        "study": {
            "dataset": "p2_study/coco2017.yaml",
            "imgsz": 640,
            "staged_head_epochs": 20,
            "seed": 0,
            "deterministic": False,
            "workers": 8,
            "device": 0,
        }
    }

    worker.train_staged(config, "head")

    assert captured["deterministic"] is False


def test_controller_paths_are_portable(tmp_path):
    controller = (ROOT / "p2_study/ctl.sh").read_text()

    assert "/home/uxin" not in controller
    assert ".worktrees" not in controller
    assert "SCRIPT_DIR=" in controller
    assert "P2_PYTHON" in controller
    assert "readlink -f" in controller
    assert "python3" in controller

    project = tmp_path / "project"
    study = project / "p2_study"
    artifacts = study / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "state.json").write_text("{\"ok\": true}\n")
    copied_controller = study / "ctl.sh"
    shutil.copy2(ROOT / "p2_study/ctl.sh", copied_controller)

    default_env = os.environ.copy()
    default_env["PATH"] = "/usr/bin:/bin"
    default_env.pop("P2_PYTHON", None)
    default = subprocess.run(
        ["bash", "-x", str(copied_controller), "status"],
        cwd=project,
        env=default_env,
        capture_output=True,
        text=True,
        check=False,
    )
    default_python = Path(shutil.which("python3", path=default_env["PATH"])).resolve()
    assert default.returncode == 0, default.stderr
    assert f"+ PYTHON_BIN={default_python}" in default.stderr

    relative_env = os.environ.copy()
    relative_env["P2_PYTHON"] = os.path.relpath(sys.executable, project)
    relative = subprocess.run(
        ["bash", "-x", str(copied_controller), "status"],
        cwd=project,
        env=relative_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert relative.returncode == 0, relative.stderr
    assert f"+ PYTHON_BIN={Path(sys.executable).resolve()}" in relative.stderr
