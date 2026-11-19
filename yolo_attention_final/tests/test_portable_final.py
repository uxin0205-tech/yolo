from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final"


def _load_training_recipe():
    path = FINAL / "training_recipe.py"
    spec = importlib.util.spec_from_file_location("portable_training_recipe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_bundle_contains_all_training_inputs() -> None:
    required = (
        FINAL / "pwl-final-best.pt",
        FINAL / "weights/v1-br-best.pt",
        FINAL / "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt",
        FINAL / "configs/variants/float-pwl-final.yaml",
        FINAL / "configs/variants/bittrue-pwl-final.yaml",
        FINAL / "configs/evaluation/coco2017.yaml",
        FINAL / "data/coco2017.yaml",
        FINAL / "yolo_attention/cli.py",
        FINAL / "BINARY_QK_PWL_ALGORITHM.md",
        FINAL / "pyproject.toml",
    )
    assert not [str(path.relative_to(ROOT)) for path in required if not path.is_file()]


def test_portable_modules_match_production_source() -> None:
    production = {path.name: path for path in (ROOT / "src/yolo_attention").glob("*.py")}
    portable = {path.name: path for path in (FINAL / "yolo_attention").glob("*.py")}
    assert portable.keys() == production.keys()
    for name, source in production.items():
        assert portable[name].read_bytes() == source.read_bytes(), name


def test_configure_runtime_updates_isolated_recipes(tmp_path: Path, monkeypatch) -> None:
    module = _load_training_recipe()
    project = tmp_path / "portable"
    (project / "data").mkdir(parents=True)
    (project / "configs/training").mkdir(parents=True)
    (project / "configs/evaluation").mkdir(parents=True)
    (project / "data/coco2017.yaml").write_text("path: datasets/coco2017\n", encoding="utf-8")
    (project / "configs/training/example.yaml").write_text(
        "data: old.yaml\ndevice: cpu\nbatch: 1\nworkers: 0\n",
        encoding="utf-8",
    )
    (project / "configs/evaluation/coco2017.yaml").write_text(
        "data: old.yaml\ndevice: cpu\nbatch: 1\nworkers: 0\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "coco2017"
    (dataset / "images/train2017").mkdir(parents=True)
    (dataset / "images/val2017").mkdir(parents=True)

    monkeypatch.setattr(module, "PROJECT_ROOT", project)
    monkeypatch.setattr(module, "DEFAULT_QUEUE_ROOT", project / "artifacts/final-reproduction-queue")
    report = module.configure_runtime(data_root=dataset, device="1", batch=8, workers=2)

    assert report["training_recipes_updated"] == 1
    assert yaml.safe_load((project / "data/coco2017.yaml").read_text())["path"] == str(dataset)
    training = yaml.safe_load((project / "configs/training/example.yaml").read_text())
    evaluation = yaml.safe_load((project / "configs/evaluation/coco2017.yaml").read_text())
    assert training == {
        "data": "data/coco2017.yaml",
        "device": "1",
        "batch": 8,
        "workers": 2,
    }
    assert evaluation == training


def test_portable_train_defaults_to_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, "run.py", "train"],
        cwd=FINAL,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    preview = json.loads(completed.stdout)
    assert preview["execute"] is False
    assert preview["missing_inputs"] == []
    assert Path(preview["queue_root"]).parent == FINAL / "artifacts"
