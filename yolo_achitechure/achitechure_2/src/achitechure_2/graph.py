"""融合模型的動態 graph 契約與 C3k2 結構稽核。"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True)
class LayerReport:
    """一個可被 C1–C3 改寫的 C3k2 結構。"""

    index: int
    class_name: str
    input_channels: int
    output_channels: int
    hidden_channels: int
    outer_repeats: int
    e: float
    inner_n: int
    kernel_mode: str
    use_rep: bool


@dataclass(frozen=True)
class ModuleReport:
    path: str
    class_name: str
    parameter_count: int
    buffer_count: int
    c3k2: LayerReport | None


@dataclass(frozen=True)
class FusionGraphReport:
    fusion_kind: str
    tasks: tuple[str, ...]
    model_contract: dict[str, Any]
    candidate_modules: tuple[ModuleReport, ...]
    protected_modules: tuple[ModuleReport, ...]
    frozen_module_paths: tuple[str, ...]
    resolved_candidate_id: str | None
    changed_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inner_blocks(layer: nn.Module) -> tuple[nn.Module, ...]:
    blocks: list[nn.Module] = []
    for outer in getattr(layer, "m", ()):
        inner = getattr(outer, "m", None)
        if not isinstance(inner, (nn.Sequential, nn.ModuleList)):
            raise TypeError(
                f"target layer contains non-C3k inner module {type(outer).__name__}"
            )
        blocks.extend(inner)
    if not blocks:
        raise ValueError("target C3k2 has no inner Bottleneck")
    return tuple(blocks)


def _raw_conv(value: object) -> nn.Conv2d:
    while not isinstance(value, nn.Conv2d) and hasattr(value, "conv"):
        value = value.conv
    if not isinstance(value, nn.Conv2d):
        raise TypeError("cannot unwrap Conv2d")
    return value


def _kernel_size(block: nn.Module, name: str) -> int:
    conv_wrapper = getattr(block, name, None)
    conv = getattr(conv_wrapper, "conv", None)
    if conv is None and name == "cv1":
        conv = getattr(conv_wrapper, "conv1", None)
        conv = getattr(conv, "conv", conv)
    conv = _raw_conv(conv)
    kernel = getattr(conv, "kernel_size", None)
    if isinstance(kernel, tuple) and len(set(kernel)) == 1:
        return int(kernel[0])
    raise ValueError(f"cannot infer {name} kernel for {type(block).__name__}")


def inspect_c3k2_layer(index: int, layer: nn.Module) -> LayerReport:
    """以實際 module 內容辨認 C3k2/LiteC3k2，不依賴固定 layer index。"""

    cv1 = getattr(getattr(layer, "cv1", None), "conv", None)
    cv2 = getattr(getattr(layer, "cv2", None), "conv", None)
    if cv1 is None or cv2 is None or not hasattr(layer, "c"):
        raise TypeError(f"layer {index} is not a compatible C3k2: {type(layer).__name__}")
    cv1 = _raw_conv(cv1)
    cv2 = _raw_conv(cv2)
    blocks = _inner_blocks(layer)
    kernels = {
        (_kernel_size(block, "cv1"), _kernel_size(block, "cv2")) for block in blocks
    }
    if len(kernels) != 1:
        raise ValueError(f"layer {index} uses mixed inner kernels: {kernels}")
    kernel_pair = kernels.pop()
    kernel_mode = {(3, 3): "3x3_3x3", (1, 3): "1x1_3x3"}.get(kernel_pair)
    if kernel_mode is None:
        raise ValueError(f"layer {index} has unsupported kernel pair {kernel_pair}")
    output_channels = int(cv2.out_channels)
    hidden_channels = int(layer.c)
    return LayerReport(
        index=index,
        class_name=type(layer).__name__,
        input_channels=int(cv1.in_channels),
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        outer_repeats=len(layer.m),
        e=hidden_channels / output_channels,
        inner_n=len(blocks) // len(layer.m),
        kernel_mode=kernel_mode,
        use_rep=all(type(block).__name__ == "RepBottleneck" for block in blocks),
    )


def _contract(model: nn.Module) -> dict[str, Any]:
    method = getattr(model, "contract", None)
    if not callable(method):
        raise TypeError("Fusion Winner 必須提供 contract()")
    contract = method()
    if not isinstance(contract, dict):
        raise TypeError("Fusion Winner contract() 必須回傳 mapping")
    return copy.deepcopy(contract)


def _module(model: nn.Module, path: str) -> nn.Module:
    try:
        return model.get_submodule(path)
    except AttributeError as error:
        raise ValueError(f"module path 不存在：{path}") from error


def _module_report(model: nn.Module, path: str, *, c3k2: bool) -> ModuleReport:
    module = _module(model, path)
    c3k2_report = inspect_c3k2_layer(-1, module) if c3k2 else None
    return ModuleReport(
        path=path,
        class_name=type(module).__name__,
        parameter_count=sum(parameter.numel() for parameter in module.parameters()),
        buffer_count=sum(buffer.numel() for buffer in module.buffers()),
        c3k2=c3k2_report,
    )


def _unique(values: Iterable[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} 含重複 module path")
    return result


def inspect_fusion_graph(
    model: nn.Module,
    *,
    fusion_kind: str,
    candidate_regions: Iterable[Any],
    protected_module_paths: Iterable[str],
    frozen_module_paths: Iterable[str],
    expected_candidate: Any | None = None,
) -> FusionGraphReport:
    """依 handoff 宣告的路徑稽核 graph，不猜測 winner 的固定層號。"""

    contract = _contract(model)
    if contract.get("model_kind") != fusion_kind:
        raise ValueError("model contract 與 fusion_kind 不一致")
    if contract.get("interface") != "model(images, tasks=detect|pose|both)":
        raise ValueError("model contract 不支援 detect/pose/both")

    regions = tuple(candidate_regions)
    candidate_paths = _unique(
        (path for region in regions for path in region.module_paths),
        "candidate regions",
    )
    protected_paths = _unique(protected_module_paths, "protected modules")
    frozen_paths = _unique(frozen_module_paths, "frozen modules")
    if set(candidate_paths) & set(protected_paths):
        raise ValueError("candidate 與 protected module paths 重疊")
    if not set(frozen_paths).issubset(protected_paths):
        raise ValueError("frozen module paths 必須是 protected paths 的子集合")
    head_paths = {path for region in regions for path in region.head_paths}
    if not head_paths.issubset(protected_paths):
        raise ValueError("所有 Detect/Pose head 都必須列入 protected paths")

    candidate_reports = tuple(
        _module_report(model, path, c3k2=True) for path in candidate_paths
    )
    protected_reports = tuple(
        _module_report(model, path, c3k2=False) for path in protected_paths
    )
    for path in frozen_paths:
        _module(model, path)

    resolved_id: str | None = None
    changed_fields: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    if expected_candidate is not None:
        if expected_candidate.fusion_kind != fusion_kind:
            raise ValueError("candidate build report 與 fusion_kind 不一致")
        resolved_id = str(expected_candidate.resolved_id)
        changed_fields = tuple(expected_candidate.changed_fields)
        changed_paths = tuple(expected_candidate.changed_module_paths)
        if not set(changed_paths).issubset(candidate_paths):
            raise ValueError("candidate build report 逸出 handoff Candidate Region")
        actual = {item.path: asdict(item.c3k2) for item in candidate_reports}
        for expected in expected_candidate.module_contracts:
            path = expected["path"]
            observed = actual[path]
            for field in (
                "e",
                "inner_n",
                "kernel_mode",
                "use_rep",
                "input_channels",
                "output_channels",
            ):
                if observed[field] != expected[field]:
                    raise ValueError(f"{path}: candidate graph 的 {field} 與 build report 不一致")

    for item in candidate_reports:
        factors = item.c3k2
        assert factors is not None
        if item.path not in changed_paths and (
            factors.e != 0.5
            or factors.inner_n != 2
            or factors.kernel_mode != "3x3_3x3"
            or factors.use_rep
        ):
            raise ValueError(f"{item.path}: 未選取的 C0 factor 已漂移")

    return FusionGraphReport(
        fusion_kind=fusion_kind,
        tasks=("detect", "pose", "both"),
        model_contract=contract,
        candidate_modules=candidate_reports,
        protected_modules=protected_reports,
        frozen_module_paths=frozen_paths,
        resolved_candidate_id=resolved_id,
        changed_fields=changed_fields,
    )
