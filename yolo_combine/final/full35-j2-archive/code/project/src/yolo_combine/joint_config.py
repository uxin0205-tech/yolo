"""Public formal config with executable augmentation and training policy."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import yaml

from . import _joint_config_impl as _impl
from .diagnostic_sampling import (
    checkpoint_is_diagnostic,
    diagnostic_marker_for_checkpoint,
)
from .plateau import PlateauPolicy
from .stage_policy import JOINT_STAGES

FormalPreflightReport = _impl.FormalPreflightReport


class JointExperimentConfig(_impl.JointExperimentConfig):
    def _raw_section(self, name: str) -> dict[str, Any]:
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get(name), dict):
            raise ValueError(f"joint config section {name!r} must be a mapping")
        return payload[name]

    @property
    def detect_mosaic(self) -> float:
        return float(self._raw_section("data").get("detect_mosaic"))

    @property
    def pose_mosaic(self) -> float:
        return float(self._raw_section("data").get("pose_mosaic"))

    @property
    def detect_fliplr(self) -> float:
        return float(self._raw_section("data").get("detect_fliplr"))

    @property
    def pose_fliplr(self) -> float:
        return float(self._raw_section("data").get("pose_fliplr"))

    @property
    def shared_bn_affine_trainable(self) -> bool:
        value = self._raw_section("training").get(
            "shared_bn_affine_trainable",
            True,
        )
        if not isinstance(value, bool):
            raise ValueError("training.shared_bn_affine_trainable must be boolean")
        return value

    @property
    def j2_plateau_policy(self) -> PlateauPolicy:
        raw = self._raw_section("training").get("j2_plateau")
        if not isinstance(raw, dict):
            raise ValueError("training.j2_plateau must be a mapping")
        required = {
            "monitor",
            "patience",
            "recovery_after",
            "lr_factor",
            "max_reductions",
            "min_delta",
            "adjust_momentum",
        }
        if set(raw) != required:
            raise ValueError(
                "training.j2_plateau keys differ: "
                f"missing={sorted(required - set(raw))}, "
                f"unexpected={sorted(set(raw) - required)}"
            )
        if not isinstance(raw["adjust_momentum"], bool):
            raise ValueError("training.j2_plateau.adjust_momentum must be boolean")
        return PlateauPolicy(
            monitor=str(raw["monitor"]),
            patience=int(raw["patience"]),
            recovery_after=int(raw["recovery_after"]),
            lr_factor=float(raw["lr_factor"]),
            max_reductions=int(raw["max_reductions"]),
            min_delta=float(raw["min_delta"]),
            adjust_momentum=raw["adjust_momentum"],
        )

    def _validate(self) -> None:
        super()._validate()
        if self.stages != ("j0", "j1", "j2"):
            raise ValueError(
                "formal default stages must be exactly j0 -> j1 -> j2; "
                "J3 remains opt-in"
            )
        if self.warmup_epochs != 1:
            raise ValueError(
                "training.warmup_epochs must be 1 for the accepted J0-J2 recipe"
            )
        probabilities = {
            "data.detect_mosaic": self.detect_mosaic,
            "data.pose_mosaic": self.pose_mosaic,
            "data.detect_fliplr": self.detect_fliplr,
            "data.pose_fliplr": self.pose_fliplr,
        }
        invalid = {
            name: value
            for name, value in probabilities.items()
            if not 0.0 <= value <= 1.0
        }
        if invalid:
            raise ValueError(f"augmentation probabilities must be in [0,1]: {invalid}")
        self.shared_bn_affine_trainable
        plateau = self.j2_plateau_policy
        if plateau.patience != JOINT_STAGES["j2"].patience:
            raise ValueError(
                "training.j2_plateau.patience must match JOINT_STAGES['j2']"
            )

    def preflight(self) -> FormalPreflightReport:
        report = super().preflight()
        blockers = list(report.blockers)
        if (
            self.pose_checkpoint is not None
            and self.pose_checkpoint.is_file()
            and checkpoint_is_diagnostic(self.pose_checkpoint)
        ):
            marker = diagnostic_marker_for_checkpoint(self.pose_checkpoint)
            blockers.append(
                "Pose26 checkpoint 屬於 fraction 診斷產物，"
                "不得解除正式 blocker；"
                f"checkpoint={self.pose_checkpoint}, marker={marker}"
            )
        return FormalPreflightReport(
            ready=not blockers,
            blockers=tuple(blockers),
            warnings=report.warnings,
            baseline=report.baseline,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["augmentation"] = {
            "detect_mosaic": self.detect_mosaic,
            "pose_mosaic": self.pose_mosaic,
            "detect_fliplr": self.detect_fliplr,
            "pose_fliplr": self.pose_fliplr,
        }
        payload["shared_bn"] = {
            "running_statistics_frozen": True,
            "affine_trainable": self.shared_bn_affine_trainable,
        }
        payload["j2_plateau"] = asdict(self.j2_plateau_policy)
        payload["stage_policies"] = {
            name: {
                "task_mode": stage.task_mode,
                "epochs": stage.epochs,
                "patience": stage.patience,
                "warmup_epochs": stage.warmup_epochs,
                "backbone_start_layer": stage.backbone_start_layer,
                "tune_attention": stage.tune_attention,
                "learning_rates": dict(stage.learning_rates),
            }
            for name, stage in JOINT_STAGES.items()
        }
        return payload


__all__ = ("FormalPreflightReport", "JointExperimentConfig")
