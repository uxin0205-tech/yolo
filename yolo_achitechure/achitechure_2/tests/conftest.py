from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

import pytest
import torch
from torch import nn
from ultralytics.nn.modules.block import Bottleneck, C3k2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _Normalization(str, Enum):
    FLOAT = "piecewise_linear"
    BITTRUE = "bit_true_pwl"


class _AttentionConfig:
    def __init__(self, normalization: str) -> None:
        self.normalization = _Normalization(normalization)


class HardwareFriendlyAttention(nn.Module):
    def __init__(self, normalization: str = "piecewise_linear") -> None:
        super().__init__()
        self.config = _AttentionConfig(normalization)
        self.conv = nn.Conv2d(8, 8, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class P3MASFFull35(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.alpha * self.conv(value)


class _AttentionBlock(nn.Module):
    def __init__(self, normalization: str) -> None:
        super().__init__()
        self.attn = HardwareFriendlyAttention(normalization)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.attn(value)


class _C2PSA(nn.Module):
    def __init__(self, normalization: str) -> None:
        super().__init__()
        self.m = nn.ModuleList([_AttentionBlock(normalization)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class FlowModel(nn.Module):
    def forward(self, value):
        return value


class _Detect(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = [16, 19, 22]
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.end2end = True

    def forward(self, value):
        return value


class Pose26(_Detect):
    def __init__(self) -> None:
        super().__init__()
        self.flow_model = FlowModel()


class ToyYolo(nn.Module):
    def __init__(
        self,
        normalization: str = "piecewise_linear",
        masf: bool = True,
        task: str = "detect",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Identity() for _ in range(24)]
        for index, channels in ((2, 16), (4, 24), (6, 32), (8, 32), (13, 32), (16, 24), (19, 32)):
            layers[index] = C3k2(channels, channels, n=1, c3k=True, e=0.5)
        if masf:
            layers[16].add_module("p3_masf", P3MASFFull35())
        layers[10] = _C2PSA(normalization)
        layer22 = C3k2(32, 32, n=1, c3k=True, e=0.5)
        layer22.m = nn.ModuleList(
            [nn.Sequential(Bottleneck(layer22.c, layer22.c), _AttentionBlock(normalization))]
        )
        layers[22] = layer22
        layers[23] = Pose26() if task == "pose" else _Detect()
        self.model = nn.Sequential(*layers)
        self.end2end = True


class CombinedToy(nn.Module):
    """小型 shared winner fixture，公開介面與 yolo_combine 相同。"""

    model_kind = "shared_dual_head"

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.trunk = nn.Module()
        self.trunk.layers = nn.ModuleList(
            [
                C3k2(8, 8, n=1, c3k=True, e=0.5),
                C3k2(8, 8, n=1, c3k=True, e=0.5),
                nn.BatchNorm2d(8),
            ]
        )
        self.detect_head = nn.Conv2d(8, 80, 1)
        self.pose_head = nn.Conv2d(8, 8, 1)

    def forward(self, images: torch.Tensor, tasks=None):
        selected = {"detect", "pose"} if tasks in (None, "both") else (
            {tasks} if isinstance(tasks, str) else set(tasks)
        )
        value = self.stem(images)
        for layer in self.trunk.layers:
            value = layer(value)
        outputs = {}
        if "detect" in selected:
            outputs["detect"] = self.detect_head(value)
        if "pose" in selected:
            outputs["pose"] = self.pose_head(value)
        return outputs

    def contract(self) -> dict:
        names = {index: f"class-{index}" for index in range(80)}
        names[0] = "person"
        return {
            "interface": "model(images, tasks=detect|pose|both)",
            "model_kind": self.model_kind,
            "head_inputs": [16, 19, 22],
            "detect_nc": 80,
            "pose_nc": 2,
            "kpt_shape": [2, 3],
            "detect_names": names,
            "pose_names": {0: "ball", 1: "bat"},
        }


@pytest.fixture
def toy_parent() -> ToyYolo:
    return ToyYolo()


@pytest.fixture
def bittrue_parent() -> ToyYolo:
    return ToyYolo("bit_true_pwl")


@pytest.fixture
def pose_parent() -> ToyYolo:
    return ToyYolo(task="pose")


@pytest.fixture
def combined_parent() -> CombinedToy:
    return CombinedToy()
