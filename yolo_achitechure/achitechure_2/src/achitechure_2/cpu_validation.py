"""不使用 GPU 的融合候選結構與數值 smoke 驗證。"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .freezing import FrozenStateGuard

SCOPE = "CPU 結構／數值 smoke；不代表準確度或正式效能"


@dataclass(frozen=True)
class CpuValidationReport:
    passed: bool
    scope: str
    device: str
    tasks: tuple[str, ...]
    smoke_imgsz: int
    geometry_imgsz: int
    output_shapes: dict[str, dict[str, Any]]
    geometry_shapes: dict[str, Any]
    loss_is_finite: bool
    gradients_are_finite: bool
    trainable_gradient_count: int
    frozen_gradient_count: int
    frozen_module_paths: tuple[str, ...]
    state_dict_reload: bool
    contract_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contract(model: nn.Module) -> Any:
    method = getattr(model, "contract", None)
    if not callable(method):
        raise TypeError("模型缺少 contract()")
    return copy.deepcopy(method())


def _task_forward(model: nn.Module, images: torch.Tensor, task: str) -> dict[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    if "tasks" in parameters:
        outputs = model(images, tasks=task)
    elif "task" in parameters:
        outputs = model(images, task=task)
    else:
        raise TypeError("模型 forward 必須提供 task 或 tasks 參數")
    if not isinstance(outputs, dict):
        raise TypeError(f"task={task} 必須回傳 dict")
    expected = {"detect", "pose"} if task == "both" else {task}
    if set(outputs) != expected:
        raise ValueError(f"task={task} 回傳 keys={sorted(outputs)}，預期 {sorted(expected)}")
    return outputs


def _leaf_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for child in value.values() for tensor in _leaf_tensors(child)]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in _leaf_tensors(child)]
    return []


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {name: _shape(child) for name, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shape(child) for child in value]
    return type(value).__name__


def _outputs_equal(first: Any, second: Any) -> bool:
    first_tensors = _leaf_tensors(first)
    second_tensors = _leaf_tensors(second)
    return len(first_tensors) == len(second_tensors) and all(
        left.shape == right.shape and torch.equal(left, right)
        for left, right in zip(first_tensors, second_tensors)
    )


def validate_cpu_candidate(
    source_model: nn.Module,
    *,
    builder: Callable[[], nn.Module],
    frozen_module_paths: Iterable[str],
    smoke_imgsz: int = 64,
    geometry_imgsz: int = 640,
) -> CpuValidationReport:
    """驗證三種 task、synthetic loss/gradient、640 geometry 與重建重載。"""

    if smoke_imgsz <= 0 or geometry_imgsz <= 0:
        raise ValueError("imgsz 必須為正整數")
    model = copy.deepcopy(source_model).cpu()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise AssertionError("CPU dry-run 不得保留 CUDA tensors")
    contract = _contract(model)
    tasks = ("detect", "pose", "both")

    model.eval()
    smoke = torch.zeros(1, 3, smoke_imgsz, smoke_imgsz)
    with torch.no_grad():
        task_outputs = {task: _task_forward(model, smoke, task) for task in tasks}
    output_shapes = {
        task: {name: _shape(value) for name, value in outputs.items()}
        for task, outputs in task_outputs.items()
    }

    geometry = torch.zeros(1, 3, geometry_imgsz, geometry_imgsz)
    with torch.no_grad():
        geometry_output = _task_forward(model, geometry, "both")
    geometry_shapes = {
        name: _shape(value) for name, value in geometry_output.items()
    }
    del geometry, geometry_output

    model.train()
    guard = FrozenStateGuard.capture(model, frozen_module_paths)
    training_output = _task_forward(model, smoke, "both")
    tensors = [tensor for tensor in _leaf_tensors(training_output) if tensor.is_floating_point()]
    if not tensors:
        raise ValueError("model output 沒有可建立 synthetic loss 的 floating tensor")
    loss = sum(tensor.square().mean() for tensor in tensors)
    loss_is_finite = bool(torch.isfinite(loss).item())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradients_are_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    frozen_ids = {
        id(parameter)
        for path in guard.paths
        for parameter in model.get_submodule(path).parameters()
    }
    frozen_gradient_count = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if id(parameter) in frozen_ids
    )
    trainable_gradient_count = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if id(parameter) not in frozen_ids and parameter.requires_grad
    )
    guard.assert_unchanged(model)

    reloaded = builder().cpu()
    contract_unchanged = _contract(reloaded) == contract
    reloaded.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        expected = _task_forward(model, smoke, "both")
        actual = _task_forward(reloaded, smoke, "both")
    state_dict_reload = _outputs_equal(expected, actual)

    passed = all(
        (
            loss_is_finite,
            gradients_are_finite,
            trainable_gradient_count > 0,
            frozen_gradient_count == 0,
            state_dict_reload,
            contract_unchanged,
        )
    )
    return CpuValidationReport(
        passed=passed,
        scope=SCOPE,
        device="cpu",
        tasks=tasks,
        smoke_imgsz=smoke_imgsz,
        geometry_imgsz=geometry_imgsz,
        output_shapes=output_shapes,
        geometry_shapes=geometry_shapes,
        loss_is_finite=loss_is_finite,
        gradients_are_finite=gradients_are_finite,
        trainable_gradient_count=trainable_gradient_count,
        frozen_gradient_count=frozen_gradient_count,
        frozen_module_paths=guard.paths,
        state_dict_reload=state_dict_reload,
        contract_unchanged=contract_unchanged,
    )
