"""Graph-aware, index-preserving P3 MASF grafting for YOLO26m."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.head import Detect

from .masf import P3MASFFull35, P3MASFPartial75

MASF_VARIANTS = frozenset({"full35", "partial75"})
ATTENTION_PATHS = ("model.10.m.0.attn", "model.22.m.0.1.attn")


@dataclass(frozen=True)
class GraphReport:
    p3_index: int
    detect_inputs: tuple[int, ...]
    strides: tuple[int, ...]
    end2end: bool
    attention_paths: tuple[str, ...]


@dataclass(frozen=True)
class GraftReport:
    variant: str
    p3_index: int
    channels: int
    preserved_tensors: int
    new_tensors: int


class C3k2P3MASFFull35(C3k2):
    """C3k2 upgraded in place, preserving all original state-dict names."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p3_masf(super().forward(x))


class C3k2P3MASFPartial75(C3k2):
    """C3k2 upgraded in place with a 25%-context/75%-bypass MASF."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p3_masf(super().forward(x))


def _network_layers(model: nn.Module) -> nn.Sequential:
    layers = getattr(model, "model", None)
    if not isinstance(layers, nn.Sequential):
        raise TypeError("expected an Ultralytics model with a Sequential .model graph")
    return layers


def inspect_yolo26_graph(model: nn.Module) -> GraphReport:
    """Fail closed unless the actual graph is a three-scale end-to-end detector."""

    layers = _network_layers(model)
    detect_entries = [(index, layer) for index, layer in enumerate(layers) if isinstance(layer, Detect)]
    if len(detect_entries) != 1:
        raise ValueError(f"expected exactly one Detect module, found {len(detect_entries)}")
    _, detect = detect_entries[0]
    inputs = tuple(int(index) for index in detect.f)
    if len(inputs) != 3:
        raise ValueError(f"Detect must consume exactly P3/P4/P5, got {inputs}")
    strides = tuple(int(value) for value in detect.stride.detach().cpu().tolist())
    if strides != (8, 16, 32):
        raise ValueError(f"expected P3/P4/P5 strides (8, 16, 32) with no P2, got {strides}")
    end2end = bool(getattr(model, "end2end", False) and getattr(detect, "end2end", False))
    if not end2end:
        raise ValueError("YOLO26 end2end=True is required")
    attention_paths = tuple(
        name for name, module in model.named_modules() if module.__class__.__name__ == "HardwareFriendlyAttention"
    )
    if attention_paths != ATTENTION_PATHS:
        raise ValueError(f"expected parent attention paths {ATTENTION_PATHS}, got {attention_paths}")
    return GraphReport(
        p3_index=inputs[0],
        detect_inputs=inputs,
        strides=strides,
        end2end=end2end,
        attention_paths=attention_paths,
    )


def _output_channels(layer: C3k2) -> int:
    cv2 = getattr(layer, "cv2", None)
    conv = getattr(cv2, "conv", None)
    channels = getattr(conv, "out_channels", None)
    if not isinstance(channels, int) or channels < 1:
        raise ValueError("cannot infer the P3 C3k2 output channels")
    return channels


def graft_p3_masf(model: nn.Module, variant: str) -> GraftReport:
    """Attach MASF after the real P3 C3k2 without adding or renumbering graph nodes."""

    variant = variant.lower()
    if variant not in MASF_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(MASF_VARIANTS)}")
    graph = inspect_yolo26_graph(model)
    layer = _network_layers(model)[graph.p3_index]
    expected_class = C3k2P3MASFFull35 if variant == "full35" else C3k2P3MASFPartial75
    module_class = P3MASFFull35 if variant == "full35" else P3MASFPartial75
    if isinstance(layer, (C3k2P3MASFFull35, C3k2P3MASFPartial75)):
        if not isinstance(layer, expected_class):
            raise TypeError("P3 already contains the other MASF variant")
        state = layer.state_dict()
        return GraftReport(variant, graph.p3_index, layer.p3_masf.channels, len(state), 0)
    if type(layer) is not C3k2:
        raise TypeError(f"P3 Detect input {graph.p3_index} must be C3k2, got {type(layer).__name__}")
    before = tuple(layer.state_dict())
    channels = _output_channels(layer)
    layer.__class__ = expected_class
    layer.add_module("p3_masf", module_class(channels).to(next(layer.parameters()).device))
    after = tuple(layer.state_dict())
    if after[: len(before)] != before:
        raise RuntimeError("graft changed original C3k2 state-dict names")
    return GraftReport(
        variant=variant,
        p3_index=graph.p3_index,
        channels=channels,
        preserved_tensors=len(model.state_dict()) - (len(after) - len(before)),
        new_tensors=len(after) - len(before),
    )
