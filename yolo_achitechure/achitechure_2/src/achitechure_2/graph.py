"""Fail-closed graph inspection for the inherited YOLO26m system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from .config import (
    ATTENTION_PATHS,
    DETECT_INPUTS,
    SPEC_PATH,
    SPEC_VERSION,
    STRIDES,
    TARGET_LAYERS,
    file_sha256,
)
from .lite_c3k2 import LiteC3k2


@dataclass(frozen=True)
class LayerReport:
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
class GraphReport:
    task: str
    head_type: str
    detect_inputs: tuple[int, ...]
    strides: tuple[int, ...]
    end2end: bool
    masf_variant: str
    masf_path: str
    attention_paths: tuple[str, ...]
    attention_normalizations: tuple[str, ...]
    layers: tuple[LayerReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def network_layers(model: nn.Module) -> nn.Sequential:
    layers = getattr(model, "model", None)
    if not isinstance(layers, nn.Sequential):
        raise TypeError("expected an Ultralytics model with a Sequential .model graph")
    return layers


def _normalization_name(module: nn.Module) -> str:
    config = getattr(module, "config", None)
    normalization = getattr(config, "normalization", None)
    value = getattr(normalization, "value", normalization)
    return str(value) if value is not None else "unknown"


def _inner_blocks(layer: nn.Module) -> tuple[nn.Module, ...]:
    blocks: list[nn.Module] = []
    for outer in getattr(layer, "m", ()):
        inner = getattr(outer, "m", None)
        if not isinstance(inner, (nn.Sequential, nn.ModuleList)):
            raise TypeError(f"target layer contains non-C3k inner module {type(outer).__name__}")
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
    cv1 = getattr(getattr(layer, "cv1", None), "conv", None)
    cv2 = getattr(getattr(layer, "cv2", None), "conv", None)
    if cv1 is None or cv2 is None or not hasattr(layer, "c"):
        raise TypeError(f"layer {index} is not a compatible C3k2: {type(layer).__name__}")
    cv1 = _raw_conv(cv1)
    cv2 = _raw_conv(cv2)
    blocks = _inner_blocks(layer)
    kernels = {(_kernel_size(block, "cv1"), _kernel_size(block, "cv2")) for block in blocks}
    if len(kernels) != 1:
        raise ValueError(f"layer {index} uses mixed inner kernels: {kernels}")
    kernel_pair = kernels.pop()
    kernel_mode = {((3, 3)): "3x3_3x3", ((1, 3)): "1x1_3x3"}.get(kernel_pair)
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


def inspect_graph(model: nn.Module, *, require_masf: bool = True) -> GraphReport:
    """Validate the inherited graph, including both attention paths and P3 MASF."""

    layers = network_layers(model)
    if len(layers) != 24:
        raise ValueError(f"expected 24 YOLO26m layers, got {len(layers)}")
    detect = layers[23]
    inputs = tuple(int(index) for index in getattr(detect, "f", ()))
    if inputs != DETECT_INPUTS:
        raise ValueError(f"expected Detect inputs {DETECT_INPUTS}, got {inputs}")
    stride = getattr(detect, "stride", None)
    strides = tuple(int(value) for value in stride.detach().cpu().tolist()) if stride is not None else ()
    if strides != STRIDES:
        raise ValueError(f"expected Detect strides {STRIDES}, got {strides}")
    end2end = bool(getattr(model, "end2end", False) and getattr(detect, "end2end", False))
    if not end2end:
        raise ValueError("YOLO26 end2end=True is required")

    attention = tuple(
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ == "HardwareFriendlyAttention"
    )
    paths = tuple(name for name, _ in attention)
    if paths != ATTENTION_PATHS:
        raise ValueError(f"expected attention paths {ATTENTION_PATHS}, got {paths}")

    p3 = layers[16]
    masf = getattr(p3, "p3_masf", None)
    if masf is None:
        if require_masf:
            raise ValueError("formal parent must contain P3 MASF at model.16.p3_masf")
        variant = "missing"
    else:
        name = type(masf).__name__.lower()
        if "full35" in name:
            variant = "full35"
        elif "partial75" in name:
            variant = "partial75"
        else:
            raise ValueError(f"unsupported P3 MASF class {type(masf).__name__}")

    head_type = type(detect).__name__.lstrip("_")
    task = "pose" if "pose" in head_type.lower() else "detect"

    return GraphReport(
        task=task,
        head_type=head_type,
        detect_inputs=inputs,
        strides=strides,
        end2end=end2end,
        masf_variant=variant,
        masf_path="model.16.p3_masf",
        attention_paths=paths,
        attention_normalizations=tuple(_normalization_name(module) for _, module in attention),
        layers=tuple(inspect_c3k2_layer(index, layers[index]) for index in TARGET_LAYERS),
    )


def assert_candidate_graph(
    model: nn.Module, candidate_id: str, expected_layers: tuple[int, ...]
) -> GraphReport:
    report = inspect_graph(model)
    changed = {entry.index for entry in report.layers if entry.class_name == LiteC3k2.__name__}
    if changed != set(expected_layers):
        raise AssertionError(
            f"{candidate_id} changed LiteC3k2 layers {sorted(changed)}, expected {expected_layers}"
        )
    return report


def graph_snapshot(model: nn.Module, candidate_id: str) -> dict[str, Any]:
    """Create a review-only Ultralytics-like snapshot from the materialized graph."""

    report = inspect_graph(model)
    layers = network_layers(model)
    entries: list[dict[str, Any]] = []
    reports = {item.index: item for item in report.layers}
    for index, layer in enumerate(layers):
        entry: dict[str, Any] = {
            "index": index,
            "from": getattr(layer, "f", -1),
            "module": type(layer).__name__,
        }
        if index in reports:
            entry["c3k2_contract"] = asdict(reports[index])
        entries.append(entry)
    return {
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "standalone_loadable": False,
        "builder": "achitechure_2",
        "candidate_id": candidate_id,
        "task": report.task,
        "head_contract": {
            "type": report.head_type,
            "inputs": list(report.detect_inputs),
            "strides": list(report.strides),
            "end2end": report.end2end,
        },
        "backbone": entries[:11],
        "head": entries[11:],
    }


def write_graph_snapshot(model: nn.Module, candidate_id: str, destination: str | Path) -> Path:
    """Write the non-standalone graph snapshot used for review and reports."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(graph_snapshot(model, candidate_id), sort_keys=False), encoding="utf-8")
    return path
