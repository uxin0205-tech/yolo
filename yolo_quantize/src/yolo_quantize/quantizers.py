"""Quantizer primitives shared by activation policies and later adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def round_to_nearest_even_ste(value: Tensor) -> Tensor:
    """Round-to-nearest-even in forward, identity straight-through backward."""

    rounded = torch.round(value)
    return value + (rounded - value).detach()


def grad_scale(value: Tensor, factor: float) -> Tensor:
    """Preserve the forward value while multiplying its backward gradient."""

    scaled = value * factor
    return value.detach() + scaled - scaled.detach()


@dataclass(frozen=True)
class LSQPlusSpec:
    """Per-tensor asymmetric LSQ+ activation-output code range."""

    bits: int
    signed_codes: bool = False
    rounding_mode: ClassVar[str] = "nearest_even"

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise ValueError("bits must be between 2 and 16")

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed_codes else 0

    @property
    def qmax(self) -> int:
        if self.signed_codes:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1


def _inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class LSQPlusActivationQuantizer(nn.Module):
    """Learnable per-tensor LSQ+ fake quantizer for skewed activations."""

    def __init__(
        self,
        spec: LSQPlusSpec,
        *,
        initial_scale: float,
        initial_offset: float,
        enabled: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if initial_scale <= eps:
            raise ValueError("initial_scale must be greater than eps")
        self.spec = spec
        self.enabled = enabled
        self.eps = float(eps)
        raw_scale = _inverse_softplus(float(initial_scale) - self.eps)
        self._scale_unconstrained = nn.Parameter(torch.tensor(raw_scale))
        self.offset = nn.Parameter(torch.tensor(float(initial_offset)))

    @property
    def scale(self) -> Tensor:
        return F.softplus(self._scale_unconstrained) + self.eps

    @torch.no_grad()
    def initialize_from_range(
        self,
        observed_min: float,
        observed_max: float,
    ) -> None:
        """Map observed endpoints to the configured integer code endpoints."""

        if not math.isfinite(observed_min) or not math.isfinite(observed_max):
            raise ValueError("observed range endpoints must be finite")
        if observed_max <= observed_min:
            raise ValueError("observed_max must be greater than observed_min")
        code_span = self.spec.qmax - self.spec.qmin
        scale = (observed_max - observed_min) / code_span
        if scale <= self.eps:
            raise ValueError("observed range is too narrow for the configured eps")
        offset = observed_min - self.spec.qmin * scale
        raw_scale = _inverse_softplus(scale - self.eps)
        self._scale_unconstrained.copy_(self._scale_unconstrained.new_tensor(raw_scale))
        self.offset.copy_(self.offset.new_tensor(offset))

    def forward(self, value: Tensor) -> Tensor:
        if not self.enabled:
            return value
        gradient_factor = 1.0 / math.sqrt(
            max(1, value.numel()) * max(1, self.spec.qmax)
        )
        scale = grad_scale(self.scale, gradient_factor)
        codes = round_to_nearest_even_ste((value - self.offset) / scale)
        codes = codes.clamp(self.spec.qmin, self.spec.qmax)
        return codes * scale + self.offset

    @torch.no_grad()
    def encode(self, value: Tensor) -> Tensor:
        """Return deployment integer codes using the same RNE endpoints."""

        codes = torch.round((value - self.offset) / self.scale)
        return codes.clamp(self.spec.qmin, self.spec.qmax).to(torch.int64)

    def extra_repr(self) -> str:
        return (
            f"bits={self.spec.bits}, qrange=[{self.spec.qmin}, "
            f"{self.spec.qmax}], enabled={self.enabled}"
        )
