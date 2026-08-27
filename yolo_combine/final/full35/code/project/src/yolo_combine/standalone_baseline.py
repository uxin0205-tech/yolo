"""Same-evaluator standalone Detect/Pose baselines for the eight hard gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator

from .metrics import GATE_METRICS
from .source import CheckpointKind, SourceBundle, file_sha256
from .validation import (
    ValidationSettings,
    extract_detect_metrics,
    extract_pose_metrics,
)

_ENDPOINT_SUFFIX = "attn.normalize.endpoint_table"
_FLOAT_SUFFIXES = (
    "attn.normalize.knots",
    "attn.normalize.values",
)


@dataclass(frozen=True)
class RepresentationCopyReport:
    copied_tensors: int
    preserved_bittrue_tensors: int
    ignored_float_tensors: int
    missing_tensors: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (
            self.missing_tensors
            or self.unexpected_tensors
            or self.shape_mismatches
        )


def copy_float_state_to_bittrue(
    source: nn.Module,
    target: nn.Module,
) -> RepresentationCopyReport:
    """Copy by explicit names and retain deterministic Bit-True PWL tables."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    preserved = {
        name for name in target_state if name.endswith(_ENDPOINT_SUFFIX)
    }
    missing = tuple(
        sorted(name for name in target_state if name not in source_state and name not in preserved)
    )
    mismatches = tuple(
        sorted(
            name
            for name in target_state.keys() & source_state.keys()
            if target_state[name].shape != source_state[name].shape
        )
    )
    compatible = tuple(
        name
        for name in target_state
        if name in source_state and target_state[name].shape == source_state[name].shape
    )
    ignored_float = {
        name
        for name in source_state
        if any(name.endswith(suffix) for suffix in _FLOAT_SUFFIXES)
        and preserved
    }
    unexpected = tuple(
        sorted(set(source_state) - set(compatible) - ignored_float)
    )
    report = RepresentationCopyReport(
        copied_tensors=len(compatible),
        preserved_bittrue_tensors=len(preserved),
        ignored_float_tensors=len(ignored_float),
        missing_tensors=missing,
        unexpected_tensors=unexpected,
        shape_mismatches=mismatches,
    )
    if not report.complete or len(compatible) + len(preserved) != len(target_state):
        return report
    with torch.no_grad():
        for name in compatible:
            target_state[name].copy_(source_state[name])
    target.load_state_dict(target_state, strict=True)
    target.requires_grad_(False)
    target.eval()
    return report


@dataclass(frozen=True)
class StandaloneBaselineResult:
    backend: CheckpointKind
    metrics: dict[str, float]
    detect_raw: dict[str, float]
    pose_raw: dict[str, float]
    output_dir: Path
    pose_representation: RepresentationCopyReport | None


class StandaloneBaselineValidator:
    """Validate the independent Detect trunk and independent Pose P3 trunk."""

    def __init__(
        self,
        source: SourceBundle,
        *,
        pose_checkpoint: str | Path,
        detect_data_yaml: str | Path,
        pose_data_yaml: str | Path,
        output_root: str | Path,
        settings: ValidationSettings = ValidationSettings(),
    ) -> None:
        self.source = source
        self.pose_checkpoint = Path(pose_checkpoint).expanduser().resolve()
        if not self.pose_checkpoint.is_file():
            raise FileNotFoundError(self.pose_checkpoint)
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
            "save_json": self.settings.save_coco_json if task == "detect" else False,
            "compile": False,
            "rect": True,
            "split": "val",
            "mode": "val",
        }

    def validate(self, kind: CheckpointKind) -> StandaloneBaselineResult:
        detect = self.source.load_detect_model(kind)
        pose_float = self.source.load_pose_checkpoint(self.pose_checkpoint)
        pose_report = None
        if kind == "float":
            pose = pose_float.eval()
        else:
            pose, _ = self.source.build_pose_model(kind="bittrue")
            pose_report = copy_float_state_to_bittrue(pose_float, pose)
            if not pose_report.complete:
                raise RuntimeError(f"Pose Bit-True conversion incomplete: {pose_report}")
        root = self.output_root / kind
        detect_validator = DetectionValidator(
            save_dir=root / "detect",
            args=self._args(
                task="detect",
                data=self.detect_data_yaml,
                batch=self.settings.detect_batch_size,
                workers=self.settings.detect_workers,
            ),
        )
        pose_validator = PoseValidator(
            save_dir=root / "pose",
            args=self._args(
                task="pose",
                data=self.pose_data_yaml,
                batch=self.settings.pose_batch_size,
                workers=self.settings.pose_workers,
            ),
        )
        detect_raw = detect_validator(model=detect)
        pose_raw = pose_validator(model=pose)
        if not isinstance(detect_raw, dict) or not isinstance(pose_raw, dict):
            raise RuntimeError("official standalone validators returned no metrics")
        metrics = {
            **extract_detect_metrics(detect_validator.metrics, names=detect.names),
            **extract_pose_metrics(pose_validator.metrics, names=pose.names),
        }
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "backend": kind,
            "pose_checkpoint": str(self.pose_checkpoint),
            "pose_checkpoint_sha256": file_sha256(self.pose_checkpoint),
            "metrics": metrics,
            "gate_metrics": {name: metrics[name] for name in GATE_METRICS},
            "detect_raw": {str(name): float(value) for name, value in detect_raw.items()},
            "pose_raw": {str(name): float(value) for name, value in pose_raw.items()},
            "pose_representation": (
                None if pose_report is None else pose_report.__dict__
            ),
        }
        (root / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return StandaloneBaselineResult(
            backend=kind,
            metrics=metrics,
            detect_raw=payload["detect_raw"],
            pose_raw=payload["pose_raw"],
            output_dir=root,
            pose_representation=pose_report,
        )

    def validate_backends(
        self,
        kinds: Sequence[CheckpointKind] = ("float", "bittrue"),
    ) -> dict[str, StandaloneBaselineResult]:
        if not kinds or len(set(kinds)) != len(kinds):
            raise ValueError("baseline backends must be unique and non-empty")
        return {kind: self.validate(kind) for kind in kinds}

    def write_gate_file(
        self,
        result: StandaloneBaselineResult,
        destination: str | Path,
    ) -> Path:
        if result.backend != "bittrue":
            raise ValueError("formal gate baseline must use Bit-True metrics")
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": "bittrue",
                    "pose_checkpoint": str(self.pose_checkpoint),
                    "pose_checkpoint_sha256": file_sha256(self.pose_checkpoint),
                    "metrics": {
                        name: result.metrics[name] for name in GATE_METRICS
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

