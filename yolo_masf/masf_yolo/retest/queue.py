"""Durable single-GPU queue for the B1R/P2/P3 retest stages.

The queue is intentionally conservative: one worker subprocess at a time,
with a JSON state file written after every completed stage.  An already
running stage is detected by its result manifest and is never duplicated.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "b1r-p2-p3-retest"
REQUESTS = ARTIFACTS / "requests"
RESULTS = ARTIFACTS / "worker"
LOGS = ARTIFACTS / "queue_logs"
STATE = ARTIFACTS / "queue_state.json"
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
DATA = ROOT / "artifacts" / "static-phase1" / "dataset" / "data.yaml"
SOURCE = ROOT / "bbt5-detect-baseline" / "weights" / "yolo11m_bat_detect_init.pt"
PROJECT = "artifacts/b1r-p2-p3-retest/training"
PARENT_DIRECT = Path("/home/uxin/single_view_pipeline/runs/detect/artifacts/b1r-p2-p3-retest/training/direct/weights/best.pt")
PARENT_CONTROL = Path("/home/uxin/single_view_pipeline/runs/detect/artifacts/b1r-p2-p3-retest/training/bbt5-p2-control-full/weights/best.pt")
PARENT_HEAD = Path("/home/uxin/single_view_pipeline/runs/detect/artifacts/b1r-p2-p3-retest/training/bbt5-p2-control-head/weights/best.pt")

VARIANTS = ("PaperFormula-Full", "Lite-35", "Lite-35-F7", "Partial50-35", "Partial25-35")


def _read_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"status": "running", "completed": [], "current": None}


def _write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _request(stage: str, family: str, variant: str | None, parent: Path | None) -> tuple[Path, Path]:
    suffix = (variant or 'base').lower().replace('-', '_')
    request_path = REQUESTS / f"queue_{family.lower()}_{stage}_{suffix}.json"
    result_path = RESULTS / (request_path.stem + ".json")
    payload = {
        "config_path": "configs/retest/b1r_p2_p3_retest.yaml",
        "stage": stage,
        "family": family,
        "variant": variant,
        "source_weights": str(SOURCE),
        "data_yaml": str(DATA),
        "project": PROJECT,
        "resume_path": None,
        "parent_checkpoint": str(parent) if parent is not None else None,
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return request_path, result_path


def _run(stage: str, family: str, variant: str | None, parent: Path | None) -> None:
    request, result = _request(stage, family, variant, parent)
    log = LOGS / f"{request.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), "-m", "masf_yolo.retest.worker", "--request", str(request), "--output", str(result)]
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    state = _read_state()
    # control_full is already running in the foreground worker.  Wait for its
    # existing manifest instead of launching a duplicate process.
    control_result = RESULTS / "control_full.json"
    while not control_result.is_file():
        state["current"] = "control_full (existing worker)"
        _write_state(state)
        time.sleep(30)
    if "control_full" not in state["completed"]:
        state["completed"].append("control_full")
        _write_state(state)

    parent = PARENT_HEAD if PARENT_HEAD.exists() else (PARENT_CONTROL if PARENT_CONTROL.exists() else PARENT_DIRECT)
    for family, family_parent in (("P2", parent), ("P3", SOURCE)):
        for variant in VARIANTS:
            key = f"{family}-smoke-{variant}"
            result = RESULTS / f"queue_{family.lower()}_smoke_{variant.lower().replace('-', '_')}.json"
            if key not in state["completed"]:
                state["current"] = key
                _write_state(state)
                _run("smoke", family, variant, family_parent)
                state["completed"].append(key)
                _write_state(state)
    for family, family_parent in (("P2", parent), ("P3", SOURCE)):
        for variant in VARIANTS:
            key = f"{family}-formal-{variant}"
            if key not in state["completed"]:
                state["current"] = key
                state["status"] = "formal_running"
                _write_state(state)
                _run("formal", family, variant, family_parent)
                state["completed"].append(key)
                _write_state(state)
    state["current"] = None
    state["status"] = "formal_complete"
    _write_state(state)


if __name__ == "__main__":
    main()
