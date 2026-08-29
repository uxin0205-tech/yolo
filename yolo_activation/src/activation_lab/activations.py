"""Activation registry and constrained integral-polynomial reference modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor, nn

ActivationName: TypeAlias = Literal[
    "silu",
    "hardswish",
    "relu",
    "qsilu_pq",
    "poly_quality",
    "poly_shift",
]


@dataclass(frozen=True)
class IntegralPolynomialProfile:
    """Immutable mathematical profile for the proposed activation family."""

    activation_name: ActivationName
    display_name: str
    a: float
    threshold: float
    hardware_class: str
    dyadic_direct_coefficients: bool
    integer_emulator_supported: bool

    @property
    def integral_coefficients(self) -> tuple[float, float, float, float]:
        """Coefficients of z^2..z^5 in the normalized integral R_a(z)."""

        return (
            self.a / 2.0,
            3.0 - 1.5 * self.a,
            1.5 * self.a - 4.0,
            1.5 - 0.5 * self.a,
        )

    @property
    def direct_coefficients(self) -> tuple[float, float, float, float]:
        """Coefficients of u^2..u^5 after folding the constant threshold."""

        return tuple(
            coefficient / (self.threshold ** (degree - 1))
            for degree, coefficient in enumerate(self.integral_coefficients, start=2)
        )


@dataclass(frozen=True)
class ShiftPiecewiseQuadraticProfile:
    """Immutable C1 qSiLU spline with dyadic global coefficients."""

    activation_name: ActivationName = "qsilu_pq"
    display_name: str = "Shift-friendly piecewise-quadratic SiLU"
    threshold: float = 8.0
    knots: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
    quadratic_coefficients: tuple[tuple[float, float, float], ...] = (
        (57.0 / 256.0, 0.0, 0.0),
        (23.0 / 256.0, 17.0 / 64.0, -17.0 / 128.0),
        (-11.0 / 512.0, 91.0 / 128.0, -37.0 / 64.0),
        (-5.0 / 1024.0, 37.0 / 64.0, -5.0 / 16.0),
    )
    hardware_class: str = "dyadic piecewise-quadratic spline"
    dyadic_direct_coefficients: bool = True
    integer_emulator_supported: bool = True


_PROFILES: Mapping[ActivationName, IntegralPolynomialProfile] = MappingProxyType(
    {
        "poly_quality": IntegralPolynomialProfile(
            activation_name="poly_quality",
            display_name="Integral polynomial (quality)",
            a=4.0,
            threshold=109.0 / 16.0,
            hardware_class="constant-multiply polynomial",
            dyadic_direct_coefficients=False,
            integer_emulator_supported=False,
        ),
        "poly_shift": IntegralPolynomialProfile(
            activation_name="poly_shift",
            display_name="Integral polynomial (shift-friendly)",
            a=9.0 / 2.0,
            threshold=8.0,
            hardware_class="dyadic/APoT polynomial",
            dyadic_direct_coefficients=True,
            integer_emulator_supported=True,
        ),
    }
)


class IntegralPolynomialActivation(nn.Module):
    """Parameter-free float reference with exact ReLU tails.

    The direct polynomial form avoids runtime division. Clipping ``u`` before
    evaluating its powers prevents unused tail values from overflowing.
    """

    def __init__(self, profile: IntegralPolynomialProfile) -> None:
        super().__init__()
        self.profile = profile

    def forward(self, x: Tensor) -> Tensor:
        u = x.abs()
        clipped_u = torch.clamp(u, max=self.profile.threshold)
        u2 = clipped_u * clipped_u
        u3 = u2 * clipped_u
        u4 = u2 * u2
        u5 = u4 * clipped_u
        c2, c3, c4, c5 = self.profile.direct_coefficients
        interior_h = c2 * u2 + c3 * u3 + c4 * u4 + c5 * u5
        tail_h = 0.5 * torch.relu(u - self.profile.threshold)
        return 0.5 * x + interior_h + tail_h

    def extra_repr(self) -> str:
        return f"profile={self.profile.activation_name}, a={self.profile.a}, T={self.profile.threshold}"


class ShiftPiecewiseQuadraticSiLU(nn.Module):
    """C1 SiLU approximation with one shared square and exact ReLU tails."""

    def __init__(self, profile: ShiftPiecewiseQuadraticProfile | None = None) -> None:
        super().__init__()
        self.profile = profile or ShiftPiecewiseQuadraticProfile()

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
        return f"profile={self.profile.activation_name}, T={self.profile.threshold}"


def available_activations() -> tuple[ActivationName, ...]:
    """Return the deliberately small Phase 0 registry in reporting order."""

    return (
        "silu",
        "hardswish",
        "relu",
        "qsilu_pq",
        "poly_shift",
        "poly_quality",
    )


def get_profile(
    name: str,
) -> IntegralPolynomialProfile | ShiftPiecewiseQuadraticProfile | None:
    """Return immutable proposed-profile metadata, or ``None`` for controls."""

    if name == "qsilu_pq":
        return ShiftPiecewiseQuadraticProfile()
    return _PROFILES.get(cast(ActivationName, name))


def build_activation(name: str) -> nn.Module:
    """Build a fresh, parameter-free activation module by stable registry name."""

    if name == "silu":
        return nn.SiLU(inplace=False)
    if name == "hardswish":
        return nn.Hardswish(inplace=False)
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "qsilu_pq":
        return ShiftPiecewiseQuadraticSiLU()
    profile = get_profile(name)
    if profile is not None:
        return IntegralPolynomialActivation(profile)
    choices = ", ".join(available_activations())
    raise ValueError(f"unknown activation {name!r}; expected one of: {choices}")
