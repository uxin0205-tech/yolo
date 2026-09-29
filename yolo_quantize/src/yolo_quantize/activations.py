"""Reviewed activation functions admitted by the quantization project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QSiLUPQProfile:
    """Immutable qSiLU contract imported from ``yolo_activation``.

    This profile describes the activation function approximation only.  It
    deliberately does not prescribe an activation-output bit width.
    """

    name: str = "qsilu_pq"
    threshold: float = 8.0
    knots: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
    quadratic_coefficients: tuple[tuple[float, float, float], ...] = (
        (57.0 / 256.0, 0.0, 0.0),
        (23.0 / 256.0, 17.0 / 64.0, -17.0 / 128.0),
        (-11.0 / 512.0, 91.0 / 128.0, -37.0 / 64.0),
        (-5.0 / 1024.0, 37.0 / 64.0, -5.0 / 16.0),
    )


@dataclass(frozen=True)
class QSiLUFixedPointConfig:
    """Legacy qSiLU Q-format used by the upstream BitTrue evaluator.

    Its signed half-away rounding is retained only for upstream parity.  The
    later activation-output quantizer owns a separate round-to-nearest-even
    contract.
    """

    total_bits: int = 16
    fraction_bits: int = 10
    rounding_mode: ClassVar[str] = "legacy_signed_half_away_from_zero"

    def __post_init__(self) -> None:
        if not 8 <= self.total_bits <= 32:
            raise ValueError("total_bits must be between 8 and 32")
        if self.fraction_bits < 10:
            raise ValueError("qsilu_pq coefficients require at least Q*.10")
        if self.fraction_bits > min(self.total_bits - 2, 20):
            raise ValueError("fraction_bits leaves no signed integer range")
        if self.qmax < 8 * self.scale:
            raise ValueError("fixed-point range must represent threshold T=8")

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    @property
    def qmin(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << (self.total_bits - 1)) - 1


def _legacy_round_shift_signed(value: Tensor, shift: int) -> Tensor:
    if shift == 0:
        return value
    magnitude = value.abs()
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return torch.where(value < 0, -rounded, rounded)


def emulate_qsilu_pq_legacy(
    input_q: Tensor,
    config: QSiLUFixedPointConfig,
) -> Tensor:
    """Evaluate qSiLU with the reviewed upstream integer arithmetic."""

    x_q = input_q.to(torch.int64).clamp(config.qmin, config.qmax)
    u_q = x_q.abs()
    u2_q = _legacy_round_shift_signed(
        u_q * u_q,
        config.fraction_bits,
    )
    scale = config.scale

    h0_q = _legacy_round_shift_signed(57 * u2_q, 8)
    h1_q = (
        _legacy_round_shift_signed(23 * u2_q, 8)
        + _legacy_round_shift_signed(17 * u_q, 6)
        - 17 * (scale >> 7)
    )
    h2_q = (
        -_legacy_round_shift_signed(11 * u2_q, 9)
        + _legacy_round_shift_signed(91 * u_q, 7)
        - 37 * (scale >> 6)
    )
    h3_q = (
        -_legacy_round_shift_signed(5 * u2_q, 10)
        + _legacy_round_shift_signed(37 * u_q, 6)
        - 5 * (scale >> 4)
    )
    interior_q = torch.where(
        u_q < scale,
        h0_q,
        torch.where(
            u_q < 2 * scale,
            h1_q,
            torch.where(u_q < 4 * scale, h2_q, h3_q),
        ),
    )
    even_q = torch.where(u_q >= 8 * scale, (u_q + 1) >> 1, interior_q)
    return ((x_q >> 1) + even_q).clamp(config.qmin, config.qmax)


class QSiLUPQ(nn.Module):
    """Parameter-free C1 piecewise-quadratic SiLU approximation.

    The module is the differentiable activation surrogate.  Output fake
    quantization remains a separate boundary owned by a quantization policy.
    """

    output_quantizer = None

    def __init__(self, profile: QSiLUPQProfile | None = None) -> None:
        super().__init__()
        self.profile = profile or QSiLUPQProfile()

    def forward(self, x: Tensor) -> Tensor:
        u = x.abs()
        u2 = u * u
        (a0, b0, c0), (a1, b1, c1), (a2, b2, c2), (a3, b3, c3) = (
            self.profile.quadratic_coefficients
        )
        h0 = a0 * u2 + b0 * u + c0
        h1 = a1 * u2 + b1 * u + c1
        h2 = a2 * u2 + b2 * u + c2
        h3 = a3 * u2 + b3 * u + c3
        interior = torch.where(
            u < 1.0,
            h0,
            torch.where(u < 2.0, h1, torch.where(u < 4.0, h2, h3)),
        )
        even_part = torch.where(u >= self.profile.threshold, 0.5 * u, interior)
        return 0.5 * x + even_part

    def extra_repr(self) -> str:
        return f"profile={self.profile.name}, T={self.profile.threshold}"
