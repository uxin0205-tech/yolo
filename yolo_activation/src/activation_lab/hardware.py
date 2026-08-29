"""Hardware proxies and a bit-accurate emulator for the shift-friendly profile."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .activations import ActivationName


@dataclass(frozen=True)
class HardwareProxy:
    variable_multiplications: int
    constant_multiplications: int
    add_subtract: int
    range_operations: int
    transcendental_operations: int
    coefficient_count: int
    power_of_two_threshold: bool
    dyadic_or_apot_constants: bool
    fixed_point_emulator: bool
    standard_fused_operator: bool
    interpretation: str


@dataclass(frozen=True)
class FixedPointConfig:
    total_bits: int = 16
    fraction_bits: int = 10

    def __post_init__(self) -> None:
        if not 8 <= self.total_bits <= 32:
            raise ValueError("total_bits must be between 8 and 32")
        if not 0 <= self.fraction_bits <= min(self.total_bits - 2, 20):
            raise ValueError(
                "fraction_bits leaves no signed integer range or exceeds 20"
            )
        if self.qmax < 8 * self.scale:
            raise ValueError(
                "fixed-point range must represent the poly_shift threshold T=8"
            )

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    @property
    def qmin(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << (self.total_bits - 1)) - 1


def hardware_proxy(name: ActivationName) -> HardwareProxy:
    """Return an operator proxy, never a latency, energy, or area claim."""

    if name == "silu":
        return HardwareProxy(
            1,
            0,
            0,
            0,
            1,
            0,
            False,
            False,
            False,
            True,
            "Sigmoid + data multiply, or backend-fused SiLU",
        )
    if name == "hardswish":
        return HardwareProxy(
            1,
            1,
            1,
            1,
            0,
            0,
            False,
            False,
            False,
            True,
            "HardSwish op, or add/clip/multiply/scale decomposition",
        )
    if name == "relu":
        return HardwareProxy(
            0, 0, 0, 1, 0, 0, False, True, False, True, "Compare/max primitive floor"
        )
    if name == "qsilu_pq":
        return HardwareProxy(
            1,
            8,
            10,
            5,
            0,
            10,
            True,
            True,
            True,
            False,
            "One shared square; dyadic branch coefficients and static coefficient selection",
        )
    if name == "poly_shift":
        return HardwareProxy(
            4,
            6,
            5,
            3,
            0,
            4,
            True,
            True,
            True,
            False,
            "Four data-power multiplies; all fixed constants map to shifts/adds",
        )
    if name == "poly_quality":
        return HardwareProxy(
            4,
            6,
            5,
            3,
            0,
            4,
            False,
            False,
            False,
            False,
            "Four data-power multiplies plus general constant multipliers",
        )
    raise ValueError(f"unsupported activation: {name}")


def _round_shift_signed(value: Tensor, shift: int) -> Tensor:
    if shift == 0:
        return value
    magnitude = value.abs()
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return torch.where(value < 0, -rounded, rounded)


def _multiply_q(left: Tensor, right: Tensor, config: FixedPointConfig) -> Tensor:
    return _round_shift_signed(left * right, config.fraction_bits)


def emulate_qsilu_pq_int(input_q: Tensor, config: FixedPointConfig) -> Tensor:
    """Evaluate the dyadic qSiLU spline with signed Q-format arithmetic."""

    if config.fraction_bits < 10:
        raise ValueError("qsilu_pq bit-true coefficients require at least Q*.10")
    x_q = input_q.to(torch.int64).clamp(config.qmin, config.qmax)
    u_q = x_q.abs()
    u2_q = _multiply_q(u_q, u_q, config)
    scale = config.scale

    h0_q = _round_shift_signed(57 * u2_q, 8)
    h1_q = (
        _round_shift_signed(23 * u2_q, 8)
        + _round_shift_signed(17 * u_q, 6)
        - 17 * (scale >> 7)
    )
    h2_q = (
        -_round_shift_signed(11 * u2_q, 9)
        + _round_shift_signed(91 * u_q, 7)
        - 37 * (scale >> 6)
    )
    h3_q = (
        -_round_shift_signed(5 * u2_q, 10)
        + _round_shift_signed(37 * u_q, 6)
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


def emulate_poly_shift_int(input_q: Tensor, config: FixedPointConfig) -> Tensor:
    """Evaluate ``poly_shift`` with signed integer Q-format arithmetic.

    Polynomial constants are exactly ``9/32, -15/256, 11/2048,
    -3/16384``. Intermediate powers stay in int64 and only the final output is
    saturated. The tail split makes positive/negative ReLU tails bit-exact.
    """

    x_q = input_q.to(torch.int64).clamp(config.qmin, config.qmax)
    u_q = x_q.abs()
    threshold_q = 8 * config.scale
    clipped_u_q = u_q.clamp(max=threshold_q)

    u2_q = _multiply_q(clipped_u_q, clipped_u_q, config)
    u3_q = _multiply_q(u2_q, clipped_u_q, config)
    u4_q = _multiply_q(u2_q, u2_q, config)
    u5_q = _multiply_q(u4_q, clipped_u_q, config)

    interior_h_q = (
        _round_shift_signed(9 * u2_q, 5)
        - _round_shift_signed(15 * u3_q, 8)
        + _round_shift_signed(11 * u4_q, 11)
        - _round_shift_signed(3 * u5_q, 14)
    )
    tail_h_q = (u_q + 1) >> 1
    h_q = torch.where(u_q >= threshold_q, tail_h_q, interior_h_q)

    base_q = x_q >> 1
    return (base_q + h_q).clamp(config.qmin, config.qmax)
