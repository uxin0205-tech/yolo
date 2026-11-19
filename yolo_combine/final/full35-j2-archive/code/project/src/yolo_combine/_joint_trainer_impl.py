"""Stage-aware joint epoch loop and formal training orchestration primitives."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from .contracts import Task
from .experiment_log import ExperimentLogger
from .joint_data import JointEpochScheduler


class MacroEngineLike(Protocol):
    def run(
        self,
        *,
        detect_batches: tuple[Mapping[str, Any], ...],
        pose_batches: tuple[Mapping[str, Any], ...],
        record_gradient_statistics: bool,
    ) -> Any: ...

    def advance_epoch(
        self,
        tasks: Iterable[Task | str] | None = None,
    ) -> None: ...


class StepSchedulerLike(Protocol):
    def prepare_step(self) -> Mapping[str, float]: ...

    def advance(self) -> None: ...


@dataclass(frozen=True)
class EpochTrainingReport:
    stage: str
    epoch: int
    macros: int
    detect_batches: int
    pose_batches: int
    detect_images: int
    pose_images: int
    detect_dataset_images: int
    pose_dataset_images: int
    detect_dataset_passes: float
    pose_dataset_passes: float
    pose_wraps: int
    detect_mean_loss: float
    pose_mean_loss: float
    joint_mean_loss: float
    next_global_macro_step: int


class StageWarmupCosineScheduler:
    """Per-macro warmup plus cosine decay with strict resumable state."""

    schema_version = 1

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        stage: str,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 3,
        warmup_start_factor: float = 0.1,
        final_lr_factor: float = 0.5,
    ) -> None:
        if not stage:
            raise ValueError("scheduler stage cannot be empty")
        if epochs < 1 or steps_per_epoch < 1:
            raise ValueError("scheduler epochs and steps_per_epoch must be positive")
        if not 0 <= warmup_epochs <= epochs:
            raise ValueError("warmup_epochs must be in [0, epochs]")
        if not 0 <= warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be in [0,1]")
        if not 0 <= final_lr_factor <= 1:
            raise ValueError("final_lr_factor must be in [0,1]")
        self.optimizer = optimizer
        self.stage = stage
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_start_factor = float(warmup_start_factor)
        self.final_lr_factor = float(final_lr_factor)
        self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.current_step = 0
        self.base_lrs = {
            str(group.get("group_name", index)): float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        }
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("optimizer group names must be unique")

    def _factor(self) -> float:
        step = min(self.current_step, self.total_steps - 1)
        if self.warmup_steps and step < self.warmup_steps:
            progress = (step + 1) / self.warmup_steps
            return self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * progress
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            return self.final_lr_factor
        decay_index = step - self.warmup_steps
        progress = min(max(decay_index / (decay_steps - 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr_factor + (
            1.0 - self.final_lr_factor
        ) * cosine

    def prepare_step(self) -> dict[str, float]:
        if self.current_step >= self.total_steps:
            raise RuntimeError("scheduler was advanced beyond its stage")
        factor = self._factor()
        values: dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get("group_name", index))
            lr = self.base_lrs[name] * factor
            group["lr"] = lr
            values[name] = lr
        return values

    def advance(self) -> None:
        if self.current_step >= self.total_steps:
            raise RuntimeError("scheduler was advanced beyond its stage")
        self.current_step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "warmup_epochs": self.warmup_epochs,
            "warmup_start_factor": self.warmup_start_factor,
            "final_lr_factor": self.final_lr_factor,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "current_step": self.current_step,
            "base_lrs": dict(self.base_lrs),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for name in (
            "schema_version",
            "stage",
            "epochs",
            "steps_per_epoch",
            "warmup_epochs",
            "warmup_start_factor",
            "final_lr_factor",
            "total_steps",
            "warmup_steps",
            "base_lrs",
        ):
            if state.get(name) != expected[name]:
                raise ValueError(
                    f"scheduler contract changed at {name}: "
                    f"{state.get(name)!r} != {expected[name]!r}"
                )
        current = int(state.get("current_step", -1))
        if not 0 <= current <= self.total_steps:
            raise ValueError("scheduler current_step is out of range")
        self.current_step = current


class JointEpochRunner:
    """Own one detect-primary epoch without owning model construction."""

    def __init__(
        self,
        *,
        engine: MacroEngineLike,
        detect_loader: Iterable[Mapping[str, Any]],
        pose_loader: Iterable[Mapping[str, Any]],
        scheduler: StepSchedulerLike,
        logger: ExperimentLogger,
        apply_training_mode: Callable[[], object],
        assert_hardware_contract: Callable[[], None],
        detect_batches_per_macro: int = 2,
        gradient_statistics_interval: int = 100,
    ) -> None:
        if detect_batches_per_macro < 1:
            raise ValueError("detect_batches_per_macro must be positive")
        if gradient_statistics_interval < 1:
            raise ValueError("gradient_statistics_interval must be positive")
        self.engine = engine
        self.detect_loader = detect_loader
        self.pose_loader = pose_loader
        self.scheduler = scheduler
        self.logger = logger
        self.apply_training_mode = apply_training_mode
        self.assert_hardware_contract = assert_hardware_contract
        self.detect_batches_per_macro = detect_batches_per_macro
        self.gradient_statistics_interval = gradient_statistics_interval

    @staticmethod
    def _component_values(prefix: str, values: tuple[float, ...]) -> dict[str, float]:
        return {
            f"{prefix}/component_{index}": float(value)
            for index, value in enumerate(values)
        }

    def run_epoch(
        self,
        *,
        epoch: int,
        global_macro_step: int,
        stage: str,
    ) -> EpochTrainingReport:
        if epoch < 0 or global_macro_step < 0:
            raise ValueError("epoch and global macro step cannot be negative")
        self.apply_training_mode()
        joint = JointEpochScheduler(
            detect_loader=self.detect_loader,
            pose_loader=self.pose_loader,
            detect_batches_per_macro=self.detect_batches_per_macro,
        )
        detect_weighted_loss = 0.0
        pose_weighted_loss = 0.0
        joint_loss_sum = 0.0
        detect_images = 0
        pose_images = 0
        for macro_index, macro in enumerate(joint):
            lrs = dict(self.scheduler.prepare_step())
            record_gradients = (
                global_macro_step % self.gradient_statistics_interval == 0
            )
            report = self.engine.run(
                detect_batches=macro.detect_batches,
                pose_batches=macro.pose_batches,
                record_gradient_statistics=record_gradients,
            )
            self.scheduler.advance()
            detect_weighted_loss += report.detect_mean_loss * report.detect_images
            pose_weighted_loss += report.pose_mean_loss * report.pose_images
            joint_loss_sum += report.joint_mean_loss
            detect_images += report.detect_images
            pose_images += report.pose_images
            values = {
                "loss/detect_mean": report.detect_mean_loss,
                "loss/pose_mean": report.pose_mean_loss,
                "loss/joint_mean": report.joint_mean_loss,
                "loss/backward": report.loss_for_backward,
                "gradient/clipped_norm": report.clipped_gradient_norm,
                "amp/scale": report.amp_scale,
                "amp/overflow_retries": report.amp_overflow_retries,
                "images/detect": report.detect_images,
                "images/pose": report.pose_images,
                **{f"lr/{name}": value for name, value in lrs.items()},
                **self._component_values("detect", report.detect_components),
                **self._component_values("pose", report.pose_components),
            }
            if report.gradient_statistics is not None:
                values.update(
                    {
                        "gradient/detect_shared_norm": report.gradient_statistics.detect_norm,
                        "gradient/pose_shared_norm": report.gradient_statistics.pose_norm,
                        "gradient/shared_cosine": report.gradient_statistics.cosine_similarity,
                    }
                )
            self.logger.log(
                "macro",
                step=global_macro_step,
                values=values,
                context={
                    "stage": stage,
                    "epoch": epoch,
                    "macro_in_epoch": macro_index,
                    "detect_batch_sizes": list(report.detect_batch_sizes),
                    "pose_batch_sizes": list(report.pose_batch_sizes),
                    "gradient_presence": dict(report.gradient_presence),
                },
            )
            global_macro_step += 1
        epoch_data = joint.report()
        if detect_images != epoch_data.detect_images or pose_images != epoch_data.pose_images:
            raise AssertionError("epoch loss accounting differs from scheduler accounting")
        detect_mean = detect_weighted_loss / detect_images
        pose_mean = pose_weighted_loss / pose_images
        joint_mean = joint_loss_sum / epoch_data.macros
        self.engine.advance_epoch()
        self.assert_hardware_contract()
        result = EpochTrainingReport(
            stage=stage,
            epoch=epoch,
            macros=epoch_data.macros,
            detect_batches=epoch_data.detect_batches,
            pose_batches=epoch_data.pose_batches,
            detect_images=epoch_data.detect_images,
            pose_images=epoch_data.pose_images,
            detect_dataset_images=epoch_data.detect_dataset_images,
            pose_dataset_images=epoch_data.pose_dataset_images,
            detect_dataset_passes=epoch_data.detect_dataset_passes,
            pose_dataset_passes=epoch_data.pose_dataset_passes,
            pose_wraps=epoch_data.pose_wraps,
            detect_mean_loss=detect_mean,
            pose_mean_loss=pose_mean,
            joint_mean_loss=joint_mean,
            next_global_macro_step=global_macro_step,
        )
        self.logger.log(
            "epoch",
            step=epoch,
            values={
                "loss/detect_mean": detect_mean,
                "loss/pose_mean": pose_mean,
                "loss/joint_mean": joint_mean,
                "images/detect": epoch_data.detect_images,
                "images/pose": epoch_data.pose_images,
                "passes/detect": epoch_data.detect_dataset_passes,
                "passes/pose": epoch_data.pose_dataset_passes,
                "pose_wraps": epoch_data.pose_wraps,
                "macros": epoch_data.macros,
            },
            context={"stage": stage, "report": asdict(result)},
        )
        return result


class PoseEpochRunner:
    """Run one J0 epoch using only the Pose loader and Pose criterion."""

    def __init__(
        self,
        *,
        engine: MacroEngineLike,
        pose_loader: Iterable[Mapping[str, Any]],
        scheduler: StepSchedulerLike,
        logger: ExperimentLogger,
        apply_training_mode: Callable[[], object],
        assert_hardware_contract: Callable[[], None],
    ) -> None:
        self.engine = engine
        self.pose_loader = pose_loader
        self.scheduler = scheduler
        self.logger = logger
        self.apply_training_mode = apply_training_mode
        self.assert_hardware_contract = assert_hardware_contract

    def run_epoch(
        self,
        *,
        epoch: int,
        global_macro_step: int,
        stage: str,
    ) -> EpochTrainingReport:
        if epoch < 0 or global_macro_step < 0:
            raise ValueError("epoch and global macro step cannot be negative")
        self.apply_training_mode()
        pose_weighted_loss = 0.0
        pose_images = 0
        macros = 0
        for macro_index, pose_batch in enumerate(self.pose_loader):
            lrs = dict(self.scheduler.prepare_step())
            report = self.engine.run(
                detect_batches=(),
                pose_batches=(pose_batch,),
                record_gradient_statistics=False,
            )
            self.scheduler.advance()
            if report.detect_images or report.detect_batch_sizes:
                raise AssertionError("Pose-only runner unexpectedly consumed Detect data")
            pose_weighted_loss += report.pose_mean_loss * report.pose_images
            pose_images += report.pose_images
            self.logger.log(
                "macro",
                step=global_macro_step,
                values={
                    "loss/detect_mean": 0.0,
                    "loss/pose_mean": report.pose_mean_loss,
                    "loss/joint_mean": report.joint_mean_loss,
                    "loss/backward": report.loss_for_backward,
                    "gradient/clipped_norm": report.clipped_gradient_norm,
                    "amp/scale": report.amp_scale,
                    "amp/overflow_retries": report.amp_overflow_retries,
                    "images/detect": 0,
                    "images/pose": report.pose_images,
                    **{f"lr/{name}": value for name, value in lrs.items()},
                    **JointEpochRunner._component_values(
                        "pose",
                        report.pose_components,
                    ),
                },
                context={
                    "stage": stage,
                    "epoch": epoch,
                    "macro_in_epoch": macro_index,
                    "detect_batch_sizes": [],
                    "pose_batch_sizes": list(report.pose_batch_sizes),
                    "gradient_presence": dict(report.gradient_presence),
                },
            )
            global_macro_step += 1
            macros += 1
        if macros < 1 or pose_images < 1:
            raise RuntimeError("Pose loader produced no batches")
        dataset = getattr(self.pose_loader, "dataset", None)
        pose_dataset_images = len(dataset) if dataset is not None else pose_images
        if pose_dataset_images < 1:
            raise RuntimeError("Pose dataset has no images")
        pose_mean = pose_weighted_loss / pose_images
        self.engine.advance_epoch((Task.POSE,))
        self.assert_hardware_contract()
        result = EpochTrainingReport(
            stage=stage,
            epoch=epoch,
            macros=macros,
            detect_batches=0,
            pose_batches=macros,
            detect_images=0,
            pose_images=pose_images,
            detect_dataset_images=0,
            pose_dataset_images=pose_dataset_images,
            detect_dataset_passes=0.0,
            pose_dataset_passes=pose_images / pose_dataset_images,
            pose_wraps=0,
            detect_mean_loss=0.0,
            pose_mean_loss=pose_mean,
            joint_mean_loss=pose_mean,
            next_global_macro_step=global_macro_step,
        )
        self.logger.log(
            "epoch",
            step=epoch,
            values={
                "loss/detect_mean": 0.0,
                "loss/pose_mean": pose_mean,
                "loss/joint_mean": pose_mean,
                "images/detect": 0,
                "images/pose": pose_images,
                "passes/detect": 0.0,
                "passes/pose": pose_images / pose_dataset_images,
                "pose_wraps": 0,
                "macros": macros,
            },
            context={"stage": stage, "report": asdict(result)},
        )
        return result
