"""Explicit Lite-C3k2 blocks composed from upstream Ultralytics modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch import nn
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2


class KernelMode(str, Enum):
    """Supported inner Bottleneck kernel pairs."""

    K3_K3 = "3x3_3x3"
    K1_K3 = "1x1_3x3"

    @property
    def kernels(self) -> tuple[int, int]:
        return (3, 3) if self is KernelMode.K3_K3 else (1, 3)


@dataclass(frozen=True)
class LiteC3k2Config:
    """One-factor configuration for a Lite-C3k2 replacement."""

    e: float = 0.5
    inner_n: int = 2
    kernel_mode: KernelMode | str = KernelMode.K3_K3
    use_rep: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_mode", KernelMode(self.kernel_mode))
        if not 0.0 < self.e <= 1.0:
            raise ValueError("e must be in (0, 1]")
        if self.inner_n < 1:
            raise ValueError("inner_n must be positive")
        if self.use_rep:
            raise ValueError("R1/RepBottleneck 尚未核准納入本 revision")


class LiteC3k(C3k):
    """C3k with explicit inner depth, kernels, and optional RepBottleneck."""

    def __init__(
        self,
        channels: int,
        *,
        config: LiteC3k2Config,
        shortcut: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__(channels, channels, config.inner_n, shortcut, groups, e=0.5, k=3)
        hidden = int(channels * 0.5)
        kernels = config.kernel_mode.kernels
        self.m = nn.Sequential(
            *(
                Bottleneck(hidden, hidden, shortcut, groups, k=kernels, e=1.0)
                for _ in range(config.inner_n)
            )
        )
        self.inner_n = config.inner_n
        self.kernel_mode = config.kernel_mode.value
        self.use_rep = config.use_rep


class LiteC3k2(C3k2):
    """Drop-in C3k2 whose research factors are visible and assertable."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        *,
        config: LiteC3k2Config,
        shortcut: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__(c1, c2, n, c3k=True, e=config.e, attn=False, g=groups, shortcut=shortcut)
        self.m = nn.ModuleList(
            LiteC3k(self.c, config=config, shortcut=shortcut, groups=groups) for _ in range(n)
        )
        self.lite_config = config
        self.e = config.e
        self.inner_n = config.inner_n
        self.kernel_mode = config.kernel_mode.value
        self.use_rep = config.use_rep
        self.outer_n = n
