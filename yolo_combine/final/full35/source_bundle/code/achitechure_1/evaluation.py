"""Formal Bit-True COCO2017 validation and standardized metrics export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics.data import converter
from ultralytics.models.yolo.detect.val import DetectionValidator

from .checkpoint import file_sha256
from .model import inspect_yolo26_graph


class COCO2017Validator(DetectionValidator):
    """強制套用 COCO80→COCO91 category ID，避免 directory-based val path 匯出錯誤 ID。"""

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.class_map = converter.coco80_to_coco91_class()


def _canonical_size_metrics(predictions: Path, data_yaml: Path) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("formal validation requires pycocotools for AP_S/AP_M/AP_L") from exc
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    annotations = Path(data["path"]) / "annotations/instances_val2017.json"
    if not annotations.is_file():
        raise FileNotFoundError(annotations)
    ground_truth = COCO(str(annotations))
    detections = ground_truth.loadRes(str(predictions))
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"ap_s": float(evaluator.stats[3]), "ap_m": float(evaluator.stats[4]), "ap_l": float(evaluator.stats[5])}


def _alpha(model: Any) -> float:
    graph = inspect_yolo26_graph(model)
    masf = getattr(model.model[graph.p3_index], "p3_masf", None)
    if masf is None:
        return 0.0
    return float(masf.alpha.detach().cpu())


def validate_bittrue(
    *,
    checkpoint: Path,
    data: Path,
    run_dir: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
) -> Path:
    """Run full COCO validation and emit every contractual winner metric."""

    model = YOLO(str(checkpoint.resolve()))
    inspect_yolo26_graph(model.model)
    attention = [module for module in model.model.modules() if module.__class__.__name__ == "HardwareFriendlyAttention"]
    if len(attention) != 2 or any(module.config.normalization.value != "bit_true_pwl" for module in attention):
        raise ValueError("formal validation checkpoint must contain exactly two Bit-True attention sites")
    run_dir.mkdir(parents=True, exist_ok=False)
    results = model.val(
        validator=COCO2017Validator,
        data=str(data.resolve()),
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        split="val",
        save_json=True,
        plots=True,
        project=str(run_dir),
        name="ultralytics",
        exist_ok=False,
    )
    prediction_candidates = tuple(run_dir.rglob("predictions.json"))
    if len(prediction_candidates) != 1:
        raise RuntimeError(f"expected one predictions.json, found {len(prediction_candidates)}")
    maps = [float(value) for value in results.box.maps]
    size_metrics = _canonical_size_metrics(prediction_candidates[0], data)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)},
        "selection_backend": "bit_true_pwl",
        "map50_95": float(results.box.map),
        "map50": float(results.box.map50),
        "map75": float(results.box.map75),
        "recall": float(results.box.mr),
        "per_class_ap": maps,
        "sports_ball_class_32_ap": maps[32],
        "baseball_bat_class_34_ap": maps[34],
        "alpha": _alpha(model.model),
        **size_metrics,
        "speed_ms": {name: float(value) for name, value in results.speed.items()},
        "cuda_peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "artifacts": {"predictions": str(prediction_candidates[0].resolve())},
    }
    destination = run_dir / "metrics.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
