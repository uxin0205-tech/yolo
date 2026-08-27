"""Native YOLO26 task criteria and memory-bounded joint macro-steps."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.utils.loss import E2ELoss, PoseLoss26, v8DetectionLoss

from .contracts import Task
from .fusion_model import GraphSharedDualHeadModel

Batch = Mapping[str, Any]
BatchPreprocessor = Callable[[Task, Batch], Batch]


class LossProvider(Protocol):
    def loss_for(self, task: Task | str, batch: Batch) -> "TaskLossResult": ...

    def advance_epoch(
        self,
        tasks: Iterable[Task | str] | None = None,
    ) -> None: ...


class EMALike(Protocol):
    def update(self, model: nn.Module) -> None: ...


class GradScalerLike(Protocol):
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...

    def update(self) -> None: ...

    def get_scale(self) -> float: ...


    def is_enabled(self) -> bool: ...

class _IdentityGradScaler:
    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> Any:
        return optimizer.step()

    def update(self) -> None:
        return None

    def get_scale(self) -> float:
        return 1.0


    def is_enabled(self) -> bool:
        return False

@dataclass(frozen=True)
class TaskLossResult:
    """One native raw loss plus its true physical batch size."""

    task: Task
    raw_total: torch.Tensor
    components: torch.Tensor
    actual_batch_size: int

    def __post_init__(self) -> None:
        if self.raw_total.ndim != 0:
            raise ValueError("raw_total must be a scalar tensor")
        if self.actual_batch_size < 1:
            raise ValueError("actual_batch_size must be positive")

    @property
    def mean_total(self) -> torch.Tensor:
        return self.raw_total / self.actual_batch_size


@dataclass(frozen=True)
class JointMacroPlan:
    """Loss algebra for one optimizer update."""

    detect_batch_sizes: tuple[int, ...]
    pose_batch_sizes: tuple[int, ...]
    reference_batch_size: int
    detect_weight: float
    pose_weight: float

    @classmethod
    def from_batch_sizes(
        cls,
        *,
        detect_batch_sizes: Sequence[int],
        pose_batch_sizes: Sequence[int],
        reference_batch_size: int = 64,
        detect_weight: float = 1.0,
        pose_weight: float = 1.0,
    ) -> "JointMacroPlan":
        plan = cls(
            detect_batch_sizes=tuple(int(value) for value in detect_batch_sizes),
            pose_batch_sizes=tuple(int(value) for value in pose_batch_sizes),
            reference_batch_size=reference_batch_size,
            detect_weight=float(detect_weight),
            pose_weight=float(pose_weight),
        )
        plan._validate()
        return plan

    def _validate(self) -> None:
        if not self.detect_batch_sizes and not self.pose_batch_sizes:
            raise ValueError("a macro-step requires at least one task")
        if any(value < 1 for value in (*self.detect_batch_sizes, *self.pose_batch_sizes)):
            raise ValueError("physical batch sizes must be positive")
        if self.reference_batch_size < 1:
            raise ValueError("reference_batch_size must be positive")
        if self.detect_weight < 0 or self.pose_weight < 0:
            raise ValueError("task weights cannot be negative")
        if self.detect_batch_sizes and self.detect_weight <= 0:
            raise ValueError("an active Detect task requires positive weight")
        if self.pose_batch_sizes and self.pose_weight <= 0:
            raise ValueError("an active Pose task requires positive weight")

    @property
    def detect_images(self) -> int:
        return sum(self.detect_batch_sizes)

    @property
    def pose_images(self) -> int:
        return sum(self.pose_batch_sizes)

    @property
    def weight_sum(self) -> float:
        return (
            (self.detect_weight if self.detect_batch_sizes else 0.0)
            + (self.pose_weight if self.pose_batch_sizes else 0.0)
        )

    @property
    def detect_backward_scale(self) -> float:
        if not self.detect_images:
            return 0.0
        return (
            self.reference_batch_size
            * self.detect_weight
            / self.weight_sum
            / self.detect_images
        )

    @property
    def pose_backward_scale(self) -> float:
        if not self.pose_images:
            return 0.0
        return (
            self.reference_batch_size
            * self.pose_weight
            / self.weight_sum
            / self.pose_images
        )

    def joint_mean(
        self,
        *,
        detect_raw_total: float,
        pose_raw_total: float,
    ) -> float:
        detect_mean = (
            detect_raw_total / self.detect_images if self.detect_images else 0.0
        )
        pose_mean = pose_raw_total / self.pose_images if self.pose_images else 0.0
        return (
            (self.detect_weight * detect_mean if self.detect_images else 0.0)
            + (self.pose_weight * pose_mean if self.pose_images else 0.0)
        ) / self.weight_sum

    def backward_total(
        self,
        *,
        detect_raw_total: float,
        pose_raw_total: float,
    ) -> float:
        return self.reference_batch_size * self.joint_mean(
            detect_raw_total=detect_raw_total,
            pose_raw_total=pose_raw_total,
        )


@dataclass(frozen=True)
class SharedGradientStatistics:
    detect_norm: float
    pose_norm: float
    cosine_similarity: float


@dataclass(frozen=True)
class MacroStepReport:
    detect_mean_loss: float
    pose_mean_loss: float
    joint_mean_loss: float
    loss_for_backward: float
    detect_components: tuple[float, ...]
    pose_components: tuple[float, ...]
    detect_batch_sizes: tuple[int, ...]
    pose_batch_sizes: tuple[int, ...]
    detect_images: int
    pose_images: int
    gradient_presence: dict[str, bool]
    gradient_statistics: SharedGradientStatistics | None
    amp_scale: float
    amp_overflow_retries: int
    clipped_gradient_norm: float


class _CriterionModelView:
    """Task-specific adapter that never registers another shared parameter tree."""

    def __init__(
        self,
        owner: GraphSharedDualHeadModel,
        task: Task,
        args: Any,
    ) -> None:
        self._owner = owner
        self.model = (owner.head_for(task),)
        self.args = args
        self.class_weights = None

    def parameters(self) -> Iterable[nn.Parameter]:
        return self._owner.parameters()


class NativeTaskLossRouter:
    """Expose current-version native Detect and Pose26 criteria by task."""

    def __init__(
        self,
        model: GraphSharedDualHeadModel,
        *,
        epochs: int,
        imgsz: int = 640,
        detect_overrides: Mapping[str, Any] | None = None,
        pose_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        self.model = model
        detect_args = get_cfg(
            DEFAULT_CFG,
            overrides={
                "task": "detect",
                "epochs": epochs,
                "imgsz": imgsz,
                **dict(detect_overrides or {}),
            },
        )
        pose_args = get_cfg(
            DEFAULT_CFG,
            overrides={
                "task": "pose",
                "epochs": epochs,
                "imgsz": imgsz,
                **dict(pose_overrides or {}),
            },
        )
        detect_view = _CriterionModelView(model, Task.DETECT, detect_args)
        pose_view = _CriterionModelView(model, Task.POSE, pose_args)
        self.criteria: dict[Task, Any] = {
            Task.DETECT: (
                E2ELoss(detect_view, v8DetectionLoss)
                if model.detect_head.end2end
                else v8DetectionLoss(detect_view)
            ),
            Task.POSE: (
                E2ELoss(pose_view, PoseLoss26)
                if model.pose_head.end2end
                else PoseLoss26(pose_view)
            ),
        }
        self.device = next(model.parameters()).device

    def loss_for(self, task: Task | str, batch: Batch) -> TaskLossResult:
        selected = Task(task)
        image = batch.get("img")
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"{selected.value} batch has no image tensor")
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("batch img must have shape [B,C,H,W] with B > 0")
        current_device = next(self.model.parameters()).device
        if current_device != self.device:
            raise RuntimeError(
                "model device changed after criterion construction; rebuild criteria"
            )
        if image.device != self.device:
            raise RuntimeError(
                f"{selected.value} batch is on {image.device}, expected {self.device}"
            )
        predictions = self.model(image, task=selected)[selected.value]
        raw_items, detached_components = self.criteria[selected](
            predictions,
            dict(batch),
        )
        raw_total = raw_items.sum()
        if not torch.isfinite(raw_total):
            raise FloatingPointError(
                f"non-finite {selected.value} loss: "
                f"{raw_items.detach().cpu().tolist()}"
            )
        return TaskLossResult(
            task=selected,
            raw_total=raw_total,
            components=detached_components,
            actual_batch_size=int(image.shape[0]),
        )

    def advance_epoch(
        self,
        tasks: Iterable[Task | str] | None = None,
    ) -> None:
        """Advance progressive one-to-many/one-to-one weights exactly once."""

        selected = (
            tuple(Task(task) for task in tasks)
            if tasks is not None
            else tuple(Task)
        )
        if len(set(selected)) != len(selected):
            raise ValueError("criterion task selection contains duplicates")
        for task in selected:
            criterion = self.criteria[task]
            if isinstance(criterion, E2ELoss):
                criterion.update()

    def state_dict(self) -> dict[str, dict[str, float | int | bool]]:
        state: dict[str, dict[str, float | int | bool]] = {}
        for task, criterion in self.criteria.items():
            if isinstance(criterion, E2ELoss):
                state[task.value] = {
                    "end2end": True,
                    "updates": int(criterion.updates),
                    "o2m": float(criterion.o2m),
                    "o2o": float(criterion.o2o),
                    "o2m_copy": float(criterion.o2m_copy),
                    "final_o2m": float(criterion.final_o2m),
                    "total": float(criterion.total),
                }
            else:
                state[task.value] = {"end2end": False}
        return state

    def load_state_dict(
        self,
        state: Mapping[str, Mapping[str, float | int | bool]],
    ) -> None:
        for task in Task:
            if task.value not in state:
                raise ValueError(f"criterion state is missing {task.value}")
            payload = state[task.value]
            criterion = self.criteria[task]
            expected_end2end = isinstance(criterion, E2ELoss)
            if bool(payload.get("end2end")) != expected_end2end:
                raise ValueError(f"{task.value} criterion end2end contract changed")
            if not expected_end2end:
                continue
            for name in ("updates", "o2m", "o2o", "o2m_copy", "final_o2m", "total"):
                if name not in payload:
                    raise ValueError(f"{task.value} criterion state is missing {name}")
                setattr(
                    criterion,
                    name,
                    int(payload[name]) if name == "updates" else float(payload[name]),
                )


class MacroStepEngine:
    """Perform sequential task backward passes and exactly one optimizer step."""

    def __init__(
        self,
        *,
        model: nn.Module,
        losses: LossProvider,
        optimizer: torch.optim.Optimizer,
        reference_batch_size: int = 64,
        task_weights: Mapping[Task | str, float] | None = None,
        gradient_groups: Mapping[str, Sequence[nn.Parameter]] | None = None,
        scaler: GradScalerLike | None = None,
        ema: EMALike | None = None,
        max_grad_norm: float = 10.0,
        max_amp_retries: int = 8,
        preprocess: BatchPreprocessor | None = None,
    ) -> None:
        if reference_batch_size < 1:
            raise ValueError("reference_batch_size must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if max_amp_retries < 0:
            raise ValueError("max_amp_retries cannot be negative")
        weights = {Task.DETECT: 1.0, Task.POSE: 1.0}
        for task, value in (task_weights or {}).items():
            weights[Task(task)] = float(value)
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("task weights must be non-negative with positive sum")
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.reference_batch_size = reference_batch_size
        self.task_weights = weights
        self.gradient_groups = {
            name: tuple(parameters)
            for name, parameters in (gradient_groups or {}).items()
        }
        self.scaler: GradScalerLike = scaler or _IdentityGradScaler()
        self.ema = ema
        self.max_grad_norm = max_grad_norm
        self.max_amp_retries = max_amp_retries
        self.preprocess = preprocess or (lambda _task, batch: batch)

    @staticmethod
    def _batch_size(batch: Batch) -> int:
        image = batch.get("img")
        if not isinstance(image, torch.Tensor) or image.ndim < 1:
            raise TypeError("batch must contain an image tensor")
        return int(image.shape[0])

    @staticmethod
    def _mean_components(
        components: list[tuple[torch.Tensor, int]],
        total_images: int,
    ) -> tuple[float, ...]:
        if not components:
            return ()
        shape = components[0][0].shape
        if any(value.shape != shape for value, _ in components):
            raise ValueError("loss component shapes changed within one task")
        weighted = sum(
            (value.detach().float().cpu() * batch_size for value, batch_size in components),
            start=torch.zeros(shape, dtype=torch.float32),
        )
        return tuple(float(value) for value in weighted / total_images)

    def _gradient_presence(self) -> dict[str, bool]:
        return {
            name: any(parameter.grad is not None for parameter in parameters)
            for name, parameters in self.gradient_groups.items()
        }

    @staticmethod
    def _shared_statistics(
        shared_parameters: Sequence[nn.Parameter],
        detect_scaled: Sequence[torch.Tensor],
        scale_value: float,
    ) -> SharedGradientStatistics:
        detect_squared = 0.0
        pose_squared = 0.0
        dot = 0.0
        for parameter, scaled_detect in zip(
            shared_parameters,
            detect_scaled,
            strict=True,
        ):
            detect = scaled_detect.float() / scale_value
            joint = (
                parameter.grad.detach().float()
                if parameter.grad is not None
                else torch.zeros_like(detect)
            )
            pose = joint - detect
            detect_squared += float(detect.square().sum())
            pose_squared += float(pose.square().sum())
            dot += float((detect * pose).sum())
        detect_norm = math.sqrt(detect_squared)
        pose_norm = math.sqrt(pose_squared)
        cosine = (
            dot / (detect_norm * pose_norm)
            if detect_norm > 0 and pose_norm > 0
            else 0.0
        )
        cosine = max(-1.0, min(1.0, cosine))
        return SharedGradientStatistics(
            detect_norm=detect_norm,
            pose_norm=pose_norm,
            cosine_similarity=cosine,
        )

    def run(
        self,
        *,
        detect_batches: Sequence[Batch],
        pose_batches: Sequence[Batch],
        record_gradient_statistics: bool = False,
        _amp_overflow_retries: int = 0,
    ) -> MacroStepReport:
        if _amp_overflow_retries < 0:
            raise ValueError("_amp_overflow_retries cannot be negative")
        plan = JointMacroPlan.from_batch_sizes(
            detect_batch_sizes=tuple(self._batch_size(batch) for batch in detect_batches),
            pose_batch_sizes=tuple(self._batch_size(batch) for batch in pose_batches),
            reference_batch_size=self.reference_batch_size,
            detect_weight=self.task_weights[Task.DETECT],
            pose_weight=self.task_weights[Task.POSE],
        )
        raw_totals = {Task.DETECT: 0.0, Task.POSE: 0.0}
        component_values: dict[Task, list[tuple[torch.Tensor, int]]] = {
            Task.DETECT: [],
            Task.POSE: [],
        }
        self.optimizer.zero_grad(set_to_none=True)
        shared = tuple(self.gradient_groups.get("shared", ()))
        detect_snapshot: tuple[torch.Tensor, ...] = ()
        scale_value = float(self.scaler.get_scale())
        record_joint_statistics = bool(
            record_gradient_statistics and detect_batches and pose_batches
        )
        try:
            for task, batches, backward_scale in (
                (Task.DETECT, detect_batches, plan.detect_backward_scale),
                (Task.POSE, pose_batches, plan.pose_backward_scale),
            ):
                for raw_batch in batches:
                    batch = self.preprocess(task, raw_batch)
                    result = self.losses.loss_for(task, batch)
                    expected_size = self._batch_size(raw_batch)
                    if result.actual_batch_size != expected_size:
                        raise ValueError(
                            f"{task.value} loss reported batch={result.actual_batch_size}, "
                            f"loader batch={expected_size}"
                        )
                    raw_totals[task] += float(result.raw_total.detach().cpu())
                    component_values[task].append(
                        (result.components.detach(), result.actual_batch_size)
                    )
                    self.scaler.scale(result.raw_total * backward_scale).backward()
                    del batch, result
                if task is Task.DETECT and record_joint_statistics:
                    if not shared:
                        raise ValueError(
                            "record_gradient_statistics requires a shared gradient group"
                        )
                    detect_snapshot = tuple(
                        (
                            parameter.grad.detach().clone()
                            if parameter.grad is not None
                            else torch.zeros_like(parameter)
                        )
                        for parameter in shared
                    )

            self.scaler.unscale_(self.optimizer)
            non_finite: list[str] = []
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                finite = torch.isfinite(parameter.grad)
                if bool(finite.all()):
                    continue
                invalid = int((~finite).sum().item())
                non_finite.append(
                    f"{name}(non_finite={invalid}/{parameter.grad.numel()},"
                    f"nan={int(torch.isnan(parameter.grad).sum().item())},"
                    f"posinf={int(torch.isposinf(parameter.grad).sum().item())},"
                    f"neginf={int(torch.isneginf(parameter.grad).sum().item())})"
                )
            if non_finite:
                if self.scaler.is_enabled() and _amp_overflow_retries < self.max_amp_retries:
                    previous_scale = scale_value
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    reduced_scale = float(self.scaler.get_scale())
                    self.optimizer.zero_grad(set_to_none=True)
                    detect_snapshot = ()
                    if reduced_scale < previous_scale:
                        return self.run(
                            detect_batches=detect_batches,
                            pose_batches=pose_batches,
                            record_gradient_statistics=record_gradient_statistics,
                            _amp_overflow_retries=_amp_overflow_retries + 1,
                        )
                preview = "; ".join(non_finite[:32])
                omitted = max(0, len(non_finite) - 32)
                raise FloatingPointError(
                    "non-finite gradient before optimizer step; "
                    f"scale={scale_value}; retries={_amp_overflow_retries}; "
                    f"total_parameters={len(non_finite)}; parameters={preview}; "
                    f"omitted_parameters={omitted}"
                )
            gradient_presence = self._gradient_presence()
            gradient_statistics = (
                self._shared_statistics(shared, detect_snapshot, scale_value)
                if record_joint_statistics
                else None
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                self.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        detect_mean = (
            raw_totals[Task.DETECT] / plan.detect_images
            if plan.detect_images
            else 0.0
        )
        pose_mean = (
            raw_totals[Task.POSE] / plan.pose_images
            if plan.pose_images
            else 0.0
        )
        joint_mean = plan.joint_mean(
            detect_raw_total=raw_totals[Task.DETECT],
            pose_raw_total=raw_totals[Task.POSE],
        )
        return MacroStepReport(
            detect_mean_loss=detect_mean,
            pose_mean_loss=pose_mean,
            joint_mean_loss=joint_mean,
            loss_for_backward=plan.reference_batch_size * joint_mean,
            detect_components=self._mean_components(
                component_values[Task.DETECT],
                plan.detect_images,
            ),
            pose_components=self._mean_components(
                component_values[Task.POSE],
                plan.pose_images,
            ),
            detect_batch_sizes=plan.detect_batch_sizes,
            pose_batch_sizes=plan.pose_batch_sizes,
            detect_images=plan.detect_images,
            pose_images=plan.pose_images,
            gradient_presence=gradient_presence,
            gradient_statistics=gradient_statistics,
            clipped_gradient_norm=float(gradient_norm.detach().cpu()),
            amp_scale=scale_value,
            amp_overflow_retries=_amp_overflow_retries,
        )

    def advance_epoch(
        self,
        tasks: Iterable[Task | str] | None = None,
    ) -> None:
        if tasks is None:
            self.losses.advance_epoch()
        else:
            self.losses.advance_epoch(tasks)
