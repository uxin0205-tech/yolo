"""Hardware-friendly MFAM variants composed from Ultralytics primitives."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from ultralytics.nn.modules import Conv


class MFAM(nn.Module):
    """Parallel depthwise multiscale sum followed by pointwise fusion."""

    def __init__(self, channels: int, kernels: Sequence[int] = (3, 5, 7, 9)) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        normalized = tuple(int(kernel) for kernel in kernels)
        if not normalized or any(kernel not in (3, 5, 7, 9) for kernel in normalized):
            raise ValueError("MFAM kernels must be a non-empty subset of 3, 5, 7, 9")
        if len(set(normalized)) != len(normalized):
            raise ValueError("MFAM kernels must be unique")
        self.channels = channels
        self.kernels = normalized
        self.branches = nn.ModuleList(self._branch(channels, kernel) for kernel in normalized)
        self.fuse = Conv(channels, channels, 1, 1)

    @staticmethod
    def _branch(channels: int, kernel: int) -> nn.Module:
        if kernel in (3, 5):
            return Conv(channels, channels, kernel, 1, g=channels)
        return nn.Sequential(
            Conv(channels, channels, (1, kernel), 1, g=channels),
            Conv(channels, channels, (kernel, 1), 1, g=channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        summed = value
        for branch in self.branches:
            summed = summed + branch(value)
        return self.fuse(summed)


class PaperFormulaMFAM(nn.Module):
    """MFAM following the paper's equations (1)--(6).

    The legacy :class:`MFAM` is intentionally left unchanged for Phase 1
    compatibility.  This class keeps the four depthwise branches explicit,
    sums them with the identity, then applies the two 1x1 residual fusions
    described by equation (6).  It has no learnable branch weighting or gate.
    """

    def __init__(self, channels: int, kernels: Sequence[int] = (3, 5, 7, 9)) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        normalized = tuple(int(kernel) for kernel in kernels)
        if not normalized or any(kernel not in (3, 5, 7, 9) for kernel in normalized):
            raise ValueError("paper MFAM kernels must be a non-empty subset of 3, 5, 7, 9")
        if len(set(normalized)) != len(normalized):
            raise ValueError("paper MFAM kernels must be unique")
        self.channels = channels
        self.kernels = normalized
        self.branches = nn.ModuleList(self._branch(channels, kernel) for kernel in normalized)
        self.pre_fuse = Conv(channels, channels, 1, 1)
        self.post_fuse = Conv(channels, channels, 1, 1)

    @staticmethod
    def _branch(channels: int, kernel: int) -> nn.Module:
        if kernel in (3, 5):
            return Conv(channels, channels, kernel, 1, g=channels)
        return nn.Sequential(
            Conv(channels, channels, (1, kernel), 1, g=channels),
            Conv(channels, channels, (kernel, 1), 1, g=channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        summed = value
        for branch in self.branches:
            summed = summed + branch(value)
        return self.post_fuse(self.pre_fuse(summed) + value)


class PartialPaperFormulaMFAM(nn.Module):
    """PaperFormulaMFAM on a leading channel slice with exact bypass."""

    def __init__(
        self,
        channels: int,
        processed_ratio: float,
        kernels: Sequence[int] = (3, 5),
    ) -> None:
        super().__init__()
        if processed_ratio not in (0.5, 0.25):
            raise ValueError("processed_ratio must be 0.5 or 0.25")
        processed_channels = int(channels * processed_ratio)
        if processed_channels < 1 or processed_channels + int(channels * (1 - processed_ratio)) != channels:
            raise ValueError("channels must divide exactly at the processed ratio")
        self.channels = channels
        self.processed_ratio = processed_ratio
        self.processed_channels = processed_channels
        self.process = PaperFormulaMFAM(processed_channels, kernels=kernels)

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {value.shape[1]}")
        processed, bypass = torch.split(
            value, (self.processed_channels, self.channels - self.processed_channels), dim=1
        )
        return torch.cat((self.process(processed), bypass), dim=1)


class PartialMFAM(nn.Module):
    """Apply M1 to leading channels and concatenate an exact bypass."""

    def __init__(
        self,
        channels: int,
        processed_ratio: float,
        kernels: Sequence[int] = (3, 5),
    ) -> None:
        super().__init__()
        if processed_ratio not in (0.5, 0.25):
            raise ValueError("processed_ratio must be 0.5 or 0.25")
        processed_channels = int(channels * processed_ratio)
        if processed_channels < 1 or processed_channels + int(channels * (1 - processed_ratio)) != channels:
            raise ValueError("channels must divide exactly at the processed ratio")
        self.channels = channels
        self.processed_ratio = processed_ratio
        self.processed_channels = processed_channels
        self.process = MFAM(processed_channels, kernels=kernels)

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {value.shape[1]}")
        processed, bypass = torch.split(
            value, (self.processed_channels, self.channels - self.processed_channels), dim=1
        )
        return torch.cat((self.process(processed), bypass), dim=1)
