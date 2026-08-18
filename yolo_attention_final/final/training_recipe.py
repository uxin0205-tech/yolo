"""Executable description of the exact queue-based training and selection method."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Block 1: locate the production project and choose a new immutable queue.
FINAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FINAL_ROOT.parent
DEFAULT_QUEUE_ROOT = PROJECT_ROOT / "artifacts/final-reproduction-queue"
PROJECT_CLI = (sys.executable, "-m", "yolo_attention.cli")


# Block 2: keep the complete experimental recipe visible in one data structure.
TRAINING_RECIPE = {
    "fixed_model": "official yolo26m.yaml, scale=m, 80 classes",
    "training_normalization": "differentiable Float-PWL",
    "formal_evaluation": "Bit-True PWL on full 5000-image COCO2017 val",
    "common": {
        "seed": 0,
        "optimizer": "AdamW",
        "batch": 16,
        "imgsz": 640,
        "workers": 8,
        "amp": True,
        "deterministic": True,
        "scheduler": "constant",
        "warmup_epochs": 0,
        "warmup_bias_lr": 0,
        "weight_decay": 0.0005,
        "selection_metric": "mAP50-95",
    },
    "stages": [
        {
            "name": "block LR x1/x2/x4 sweep",
            "max_epochs": 8,
            "patience": 3,
            "attention_lr": [5e-6, 1e-5, 2e-5],
            "adjacent_block_lr": [1e-6, 2e-6, 4e-6],
        },
        {
            "name": "Neck/Detect recovery",
            "max_epochs": 16,
            "patience": 5,
            "lrs": {"attention": 5e-6, "adjacent_block": 1e-6, "neck_detect": 5e-7},
        },
        {
            "name": "Backbone-last recovery",
            "max_epochs": 16,
            "patience": 5,
            "lrs": {
                "attention": 5e-6,
                "adjacent_block": 1e-6,
                "neck_detect": 5e-7,
                "backbone": 1e-7,
            },
        },
        {
            "name": "full-model recovery",
            "max_epochs": 20,
            "patience": 6,
            "lrs": {
                "attention": 5e-6,
                "adjacent_block": 1e-6,
                "neck_detect": 5e-7,
                "backbone": 1e-7,
            },
        },
    ],
    "gate": "evaluate actual Bit-True best.pt; roll back only if child loses >0.001 to direct parent",
    "global_selection": "choose highest Bit-True candidate; trained formal winner requires three-seed mean +0.001",
    "batchnorm": "freeze all running mean, variance, and counters during recovery",
}


# Block 3: call the repository's tested training engine without duplicating it.
def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(PROJECT_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else source + os.pathsep + current
    return environment


def _call(*arguments: str) -> int:
    command = (*PROJECT_CLI, *arguments)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=_environment(), check=False).returncode


def _required_inputs() -> tuple[Path, ...]:
    return (
        PROJECT_ROOT / "src/yolo_attention/cli.py",
        PROJECT_ROOT / "configs/variants/float-pwl-final.yaml",
        PROJECT_ROOT / "configs/variants/bittrue-pwl-final.yaml",
        PROJECT_ROOT / "configs/evaluation/coco2017.yaml",
        PROJECT_ROOT / "data/coco2017.yaml",
        PROJECT_ROOT / "weights/v1-br-best.pt",
        PROJECT_ROOT / "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt",
    )


# Block 4: dry-run by default; only --execute initializes or resumes GPU work.
def print_recipe() -> None:
    print(json.dumps(TRAINING_RECIPE, indent=2, ensure_ascii=False))


def run_training(queue_root: Path, *, execute: bool) -> int:
    queue_root = queue_root.expanduser().resolve()
    missing = [str(path) for path in _required_inputs() if not path.is_file()]
    preview = {
        "execute": execute,
        "queue_root": str(queue_root),
        "recipe": TRAINING_RECIPE,
        "missing_inputs": missing,
    }
    if not execute:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0 if not missing else 1
    if missing:
        raise FileNotFoundError("missing training inputs: " + ", ".join(missing))

    if not (queue_root / "queue.json").is_file():
        code = _call(
            "queue",
            "init-pwl-lr-sweep",
            "--project-root",
            str(PROJECT_ROOT),
            "--queue-root",
            str(queue_root),
        )
        if code:
            return code
    code = _call("queue", "validate", "--queue-root", str(queue_root))
    if code:
        return code
    return _call("queue", "run", "--queue-root", str(queue_root), "--execute")


def queue_status(queue_root: Path) -> int:
    queue_root = queue_root.expanduser().resolve()
    if not (queue_root / "queue.json").is_file():
        print(json.dumps({"queue_root": str(queue_root), "status": "not_initialized"}, ensure_ascii=False))
        return 1
    return _call("queue", "status", "--queue-root", str(queue_root))
