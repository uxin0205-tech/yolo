"""Fail-closed environment, lineage, graph, and dataset preflight."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.optim import MuSGD

from .checkpoint import build_training_model, file_sha256
from .config import CommonTrainingConfig
from .lineage import load_parent_study
from .model import inspect_yolo26_graph
from .runtime import nvidia_driver_version


def _fresh_process_reload(project_root: Path, checkpoint: Path) -> dict[str, Any]:
    parent_src = project_root.parents[1] / "yolo_attention_final" / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root / "src"), str(parent_src), env.get("PYTHONPATH", "")))
    )
    code = (
        "from ultralytics import YOLO; "
        f"m=YOLO({str(checkpoint)!r}).model; "
        "print(int(m.end2end), len(m.model[-1].f))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "ok": result.returncode == 0 and result.stdout.strip().endswith("1 3"),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_preflight(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    parent = root / "inputs/parent/best.pt"
    parent_study_path = root / "inputs/parent/provenance/final-results.json"
    common_path = root / "configs/training/common.yaml"
    dataset_path = root / "configs/coco2017.yaml"
    for required in (parent, parent_study_path, common_path, dataset_path):
        if not required.is_file():
            errors.append(f"missing required input: {required}")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    common = CommonTrainingConfig.from_yaml(common_path)
    try:
        parent_study = load_parent_study(parent_study_path, parent)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"parent study provenance failed validation: {exc}")
        parent_study = None
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    dataset_root = Path(dataset["path"])
    names = dataset.get("names", {})
    if names.get(32) != "sports ball" or names.get(34) != "baseball bat":
        errors.append("COCO classes 32/34 are not sports ball/baseball bat")
    if not dataset_root.is_dir():
        errors.append(f"dataset root does not exist: {dataset_root}")
    yolo = YOLO(str(parent))
    try:
        graph = inspect_yolo26_graph(yolo.model)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        graph = None
    fresh = _fresh_process_reload(root, parent)
    if not fresh["ok"]:
        errors.append(f"fresh-process checkpoint reload failed: {fresh['stderr']}")
    try:
        _, transfer = build_training_model(
            cfg=yolo.model.yaml,
            nc=80,
            channels=3,
            weights=yolo.model,
            masf_variant="full35",
            attention_config=root / "configs/attention/float-pwl-final.yaml",
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Float-PWL training model construction failed: {exc}")
        transfer = None
    try:
        import pycocotools  # noqa: F401
    except ImportError:
        warnings.append(
            "pycocotools is not installed; install the evaluation extra before formal AP_S/AP_M/AP_L validation"
        )
    if not torch.cuda.is_available():
        errors.append("CUDA GPU is required for formal batch-16 training")
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_source": ultralytics.__path__[0],
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": nvidia_driver_version(),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "vram_free_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
        "musgd": f"{MuSGD.__module__}.{MuSGD.__name__}",
    }
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "parent": {"path": str(parent), "sha256": file_sha256(parent)},
        "parent_study": parent_study,
        "fresh_process_reload": fresh,
        "graph": graph.__dict__ if graph else None,
        "transfer": transfer.__dict__ if transfer else None,
        "training_contract": {
            "batch": common.batch,
            "imgsz": common.imgsz,
            "workers": common.workers,
            "seed": common.seed,
            "amp": common.amp,
            "accumulation": common.nbs != common.batch,
        },
        "dataset": {"path": str(dataset_root), "sports_ball": names.get(32), "baseball_bat": names.get(34)},
        "environment": environment,
    }


def write_preflight(report: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
