"""Durable single-GPU queue for Clean smoke, formal, and P2 control stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..artifacts.io import atomic_write_json
from .contracts import CLEAN_EXPERIMENTS, load_clean_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "clean" / "clean_ablation.yaml"
ARTIFACTS = ROOT / "artifacts" / "clean-bbt5-ablation"
RESULTS = ARTIFACTS / "worker"
LOGS = ARTIFACTS / "queue_logs"
STATE = ARTIFACTS / "queue_state.json"
ACCEPTANCE = ARTIFACTS / "acceptance.json"


def _state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"status": "starting", "current": None, "completed": [], "error": None}


def _save(state: dict) -> None:
    atomic_write_json(STATE, state)


def _run_logged(command: list[str], log_name: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / log_name).open("a", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _accept(state: dict) -> None:
    if ACCEPTANCE.is_file():
        payload = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        if payload.get("ok"):
            return
    state.update(status="acceptance_running", current="cuda-acceptance", error=None)
    _save(state)
    _run_logged(
        [sys.executable, "-m", "masf_yolo.clean.acceptance", "--config", str(CONFIG),
         "--output", str(ACCEPTANCE)],
        "acceptance.log",
    )


def _slug(experiment: str, seed: int, stage: str) -> str:
    return f"{stage}-{experiment.lower()}-seed{seed}"


def _last_checkpoint(experiment: str, seed: int, stage: str) -> Path | None:
    suffix = "-smoke" if stage == "smoke" else ""
    path = ARTIFACTS / "training" / f"{experiment.lower()}-seed{seed}{suffix}" / "weights" / "last.pt"
    return path if path.is_file() else None


def _run_job(
    state: dict,
    *,
    experiment: str,
    seed: int,
    stage: str,
    parent: Path | None = None,
) -> Path:
    key = _slug(experiment, seed, stage)
    result = RESULTS / f"{key}.json"
    if result.is_file():
        if key not in state["completed"]:
            state["completed"].append(key)
            _save(state)
        return result
    RESULTS.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "masf_yolo.clean.worker",
        "--config", str(CONFIG), "--experiment", experiment,
        "--seed", str(seed), "--stage", stage, "--output", str(result),
    ]
    if parent is not None:
        command.extend(["--parent-checkpoint", str(parent)])
    resume = _last_checkpoint(experiment, seed, stage)
    if resume is not None:
        command.extend(["--resume-checkpoint", str(resume)])
    state.update(status=f"{stage}_running", current=key, error=None)
    _save(state)
    _run_logged(command, f"{key}.log")
    state["completed"].append(key)
    _save(state)
    return result


def _best(result: Path) -> Path:
    path = Path(json.loads(result.read_text(encoding="utf-8"))["best"])
    if not path.is_file():
        raise RuntimeError(f"worker best checkpoint is missing: {path}")
    return path


def main() -> None:
    config = load_clean_config(CONFIG)
    state = _state()
    try:
        _accept(state)
        strict = [
            name for name, spec in CLEAN_EXPERIMENTS.items()
            if spec.comparison_tier == "strict_fair"
        ]
        for experiment in strict:
            _run_job(state, experiment=experiment, seed=42, stage="smoke")
        for experiment in strict:
            for seed in config.values["seeds"]:
                _run_job(state, experiment=experiment, seed=seed, stage="formal")
        for seed in config.values["seeds"]:
            head = _run_job(
                state, experiment="P2-Control-Clean-Head", seed=seed, stage="formal"
            )
            _run_job(
                state,
                experiment="P2-Control-Clean-Full",
                seed=seed,
                stage="formal",
                parent=_best(head),
            )
        state.update(status="training_complete", current=None, error=None)
        _save(state)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save(state)
        raise


if __name__ == "__main__":
    main()
