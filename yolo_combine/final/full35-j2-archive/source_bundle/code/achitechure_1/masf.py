"""Identity-safe P3 multi-scale spatial feature modules."""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv


class _P3MASFContext(nn.Module):
    """DW3/DW5 context followed by the required channel-mixing projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.dw3 = Conv(channels, channels, 3, 1, g=channels)
        self.dw5 = Conv(channels, channels, 5, 1, g=channels)
        self.project = Conv(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.dw3(x) + self.dw5(x))


class P3MASFFull35(nn.Module):
    """Apply MASF to every P3 channel with an identity-safe scalar residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.context = _P3MASFContext(channels)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected BCHW input with {self.channels} channels, got {tuple(x.shape)}")
        return x + self.alpha * self.context(x)


class P3MASFPartial75(nn.Module):
    """Apply MASF to 25% of P3 channels and bypass the remaining 75% exactly."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("Partial75 requires at least two channels")
        self.channels = channels
        self.context_channels = max(1, round(channels * 0.25))
        self.bypass_channels = channels - self.context_channels
        self.context = _P3MASFContext(self.context_channels)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected BCHW input with {self.channels} channels, got {tuple(x.shape)}")
        context, bypass = torch.split(
            x,
            (self.context_channels, self.bypass_channels),
            dim=1,
        )
        enhanced = context + self.alpha * self.context(context)
        return torch.cat((enhanced, bypass), dim=1)
