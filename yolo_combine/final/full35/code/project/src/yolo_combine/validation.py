"""Independent official Detect/Pose validation with stable metric names."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator

from .fusion_model import GraphSharedDualHeadModel
from .graph_materialize import (
    GraphValidationModels,
    build_graph_validation_models,
)
from .source import CheckpointKind, SourceBundle


@dataclass(frozen=True)
class ValidationSettings:
    imgsz: int = 640
    detect_batch_size: int = 32
    pose_batch_size: int = 16
    detect_workers: int = 4
    pose_workers: int = 8
    device: str = "0"
    plots: bool = True
    save_coco_json: bool = False

    def __post_init__(self) -> None:
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("validation imgsz must be a multiple of 32")
        if self.detect_batch_size < 1 or self.pose_batch_size < 1:
            raise ValueError("validation batch sizes must be positive")
        if self.detect_workers < 0 or self.pose_workers < 0:
            raise ValueError("validation workers cannot be negative")


@dataclass(frozen=True)
class ValidationBackendResult:
    checkpoint_kind: CheckpointKind
    metrics: dict[str, float]
    detect_raw: dict[str, float]
    pose_raw: dict[str, float]
    output_dir: Path
    materialized: GraphValidationModels


def _overall(
    metric: Any,
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"{prefix}/precision": float(metric.mp),
        f"{prefix}/recall": float(metric.mr),
        f"{prefix}/map50": float(metric.map50),
        f"{prefix}/map75": float(metric.map75),
        f"{prefix}/map50_95": float(metric.map),
    }


def _class_values(
    metric: Any,
    *,
    class_id: int,
    prefix: str,
) -> dict[str, float]:
    indices = [
        int(value)
        for value in np.asarray(metric.ap_class_index).reshape(-1).tolist()
    ]
    if class_id not in indices:
        raise ValueError(
            f"validation metric contains no result for class_id={class_id}; "
            f"available={indices}"
        )
    position = indices.index(class_id)
    precision, recall, map50, map50_95 = metric.class_result(position)
    all_ap = np.asarray(metric.all_ap)
    if all_ap.ndim != 2 or all_ap.shape[1] < 6:
        raise ValueError(f"per-class AP table has unexpected shape {all_ap.shape}")
    return {
        f"{prefix}/precision": float(precision),
        f"{prefix}/recall": float(recall),
        f"{prefix}/map50": float(map50),
        f"{prefix}/map75": float(all_ap[position, 5]),
        f"{prefix}/map50_95": float(map50_95),
    }


def extract_detect_metrics(
    metrics: Any,
    *,
    names: Mapping[int, str],
) -> dict[str, float]:
    """Flatten COCO aggregate and person class diagnostics."""

    person_ids = [
        int(class_id)
        for class_id, name in names.items()
        if str(name) == "person"
    ]
    if person_ids != [0]:
        raise ValueError(f"Detect class schema must contain person=0, got {names}")
    return {
        **_overall(metrics.box, prefix="coco/box"),
        **_class_values(
            metrics.box,
            class_id=0,
            prefix="coco/person/box",
        ),
    }


def extract_pose_metrics(
    metrics: Any,
    *,
    names: Mapping[int, str],
) -> dict[str, float]:
    """Flatten BBAT aggregate and ball/bat box/keypoint diagnostics."""

    if dict(names) != {0: "ball", 1: "bat"}:
        raise ValueError(f"Pose class schema must be ball=0, bat=1, got {names}")
    values = {
        **_overall(metrics.box, prefix="bbat/box"),
        **_overall(metrics.pose, prefix="bbat/pose"),
    }
    for class_id, name in names.items():
        values.update(
            _class_values(
                metrics.box,
                class_id=int(class_id),
                prefix=f"bbat/{name}/box",
            )
        )
        values.update(
            _class_values(
                metrics.pose,
                class_id=int(class_id),
                prefix=f"bbat/{name}/pose",
            )
        )
    return values


class JointValidator:
    """Materialize EMA once per backend and run two official validators."""

    def __init__(
        self,
        source: SourceBundle,
        *,
        detect_data_yaml: str | Path,
        pose_data_yaml: str | Path,
        output_root: str | Path,
        settings: ValidationSettings = ValidationSettings(),
    ) -> None:
        self.source = source
        self.detect_data_yaml = Path(detect_data_yaml).expanduser().resolve()
        self.pose_data_yaml = Path(pose_data_yaml).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.settings = settings

    def _args(
        self,
        *,
        task: Literal["detect", "pose"],
        data: Path,
        batch: int,
        workers: int,
    ) -> dict[str, Any]:
        return {
            "task": task,
            "data": str(data),
            "imgsz": self.settings.imgsz,
            "batch": batch,
            "workers": workers,
            "device": self.settings.device,
            "plots": self.settings.plots,
            "save_json": (
                self.settings.save_coco_json if task == "detect" else False
            ),
            "compile": False,
            "rect": True,
            "split": "val",
            "mode": "val",
        }

    def validate(
        self,
        shared_ema: GraphSharedDualHeadModel,
        *,
        epoch: int,
        kind: CheckpointKind = "bittrue",
    ) -> ValidationBackendResult:
        if epoch < 0:
            raise ValueError("validation epoch cannot be negative")
        materialized = build_graph_validation_models(
            shared_ema,
            self.source,
            kind=kind,
        )
        root = self.output_root / f"epoch-{epoch:04d}" / kind
        detect_dir = root / "detect"
        pose_dir = root / "pose"
        detect_validator = DetectionValidator(
            save_dir=detect_dir,
            args=self._args(
                task="detect",
                data=self.detect_data_yaml,
                batch=self.settings.detect_batch_size,
                workers=self.settings.detect_workers,
            ),
        )
        pose_validator = PoseValidator(
            save_dir=pose_dir,
            args=self._args(
                task="pose",
                data=self.pose_data_yaml,
                batch=self.settings.pose_batch_size,
                workers=self.settings.pose_workers,
            ),
        )
        detect_raw = detect_validator(model=materialized.detect)
        pose_raw = pose_validator(model=materialized.pose)
        if not isinstance(detect_raw, dict) or not isinstance(pose_raw, dict):
            raise RuntimeError("official validators returned no metrics")
        metrics = {
            **extract_detect_metrics(
                detect_validator.metrics,
                names=materialized.detect.names,
            ),
            **extract_pose_metrics(
                pose_validator.metrics,
                names=materialized.pose.names,
            ),
        }
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch,
            "checkpoint_kind": kind,
            "detect_batch_size": self.settings.detect_batch_size,
            "pose_batch_size": self.settings.pose_batch_size,
            "metrics": metrics,
            "detect_raw": {
                str(name): float(value) for name, value in detect_raw.items()
            },
            "pose_raw": {
                str(name): float(value) for name, value in pose_raw.items()
            },
        }
        (root / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ValidationBackendResult(
            checkpoint_kind=kind,
            metrics=metrics,
            detect_raw=payload["detect_raw"],
            pose_raw=payload["pose_raw"],
            output_dir=root,
            materialized=materialized,
        )

    def validate_backends(
        self,
        shared_ema: GraphSharedDualHeadModel,
        *,
        epoch: int,
        kinds: Sequence[CheckpointKind] = ("float", "bittrue"),
    ) -> dict[str, ValidationBackendResult]:
        if not kinds:
            raise ValueError("at least one validation backend is required")
        if len(set(kinds)) != len(kinds):
            raise ValueError("validation backends cannot repeat")
        return {
            kind: self.validate(shared_ema, epoch=epoch, kind=kind)
            for kind in kinds
        }
