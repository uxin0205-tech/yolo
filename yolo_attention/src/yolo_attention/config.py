"""Typed configuration at the framework's public seam."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class BasisKind(StringEnum):
    FP = "fp"
    IDENTITY = "identity"
    HADAMARD = "hadamard"
    T5 = "t5"


class BiasKind(StringEnum):
    NONE = "none"
    DENSE_2D = "dense_2d"
    DECOMPOSED_2D = "decomposed_2d"


class ScaleMode(StringEnum):
    DYNAMIC = "dynamic"
    FIXED_HEAD = "fixed_head"
    POWER_OF_TWO = "power_of_two"


class BDCNCodebookKind(StringEnum):
    FIXED_EXP = "fixed_exp"
    LEARNED = "learned"


class BDCNSharing(StringEnum):
    GLOBAL = "global"
    PER_ATTENTION = "per_attention"
    PER_HEAD = "per_head"


class BDCNProjection(StringEnum):
    FLOAT = "float"
    ONE_POT = "one_pot"
    TWO_POT = "two_pot"


class BDCNDenominator(StringEnum):
    EXACT = "exact"
    RECIPROCAL_LUT = "reciprocal_lut"
    POT_SHIFT = "pot_shift"


class NormalizationKind(StringEnum):
    EXACT = "exact"
    LUT = "lut"
    INTEGER_LUT = "integer_lut"
    PIECEWISE_LINEAR = "piecewise_linear"
    POWER_OF_TWO = "power_of_two"
    HARD_SIGMOID = "hard_sigmoid"
    RELU = "relu"
    MULTIMAX = "multimax"
    BDCN = "bdcn"


class RowCorrection(StringEnum):
    NONE = "none"
    MAX_ELEMENT = "max_element"
    LARGEST_REMAINDER = "largest_remainder"


@dataclass(frozen=True)
class VariantConfig:
    """Complete attention variant configuration.

    Callers select behavior here instead of constructing internal modules.
    """

    name: str
    basis: BasisKind = BasisKind.IDENTITY
    bias: BiasKind = BiasKind.NONE
    scale_mode: ScaleMode = ScaleMode.DYNAMIC
    normalization: NormalizationKind = NormalizationKind.EXACT
    row_correction: RowCorrection = RowCorrection.NONE
    use_ste: bool = True
    max_bias_size: int = 32
    score_step: float = 0.125
    score_min: int = -64
    exp_bits: int = 15
    pwl_segments: int = 16
    relu_margin: float = 1.0
    multimax_top_k: int = 5
    normalization_progressive: bool = False
    normalization_transition_epochs: int = 5
    bdcn_codebook: BDCNCodebookKind | None = None
    bdcn_sharing: BDCNSharing | None = None
    bdcn_projection: BDCNProjection | None = None
    bdcn_denominator: BDCNDenominator | None = None
    bdcn_levels: int = 16
    bdcn_step: float = 0.125
    bdcn_distance_max: float | None = None
    bdcn_fused_value: bool = True
    bdcn_reciprocal_newton_steps: int = 0
    p_bits: int | None = None
    v_bits: int | None = None
    projection_weight_bits: int | None = None
    projection_activation_bits: int | None = None
    progressive: bool = False

    def __post_init__(self) -> None:
        for field_name, enum_type in (
            ("basis", BasisKind),
            ("bias", BiasKind),
            ("scale_mode", ScaleMode),
            ("normalization", NormalizationKind),
            ("row_correction", RowCorrection),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, field_name, enum_type(value))
        for field_name, enum_type in (
            ("bdcn_codebook", BDCNCodebookKind),
            ("bdcn_sharing", BDCNSharing),
            ("bdcn_projection", BDCNProjection),
            ("bdcn_denominator", BDCNDenominator),
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, enum_type):
                object.__setattr__(self, field_name, enum_type(value))
        if not self.name.strip():
            raise ValueError("variant name cannot be empty")
        if self.basis is BasisKind.FP and self.normalization is NormalizationKind.INTEGER_LUT:
            raise ValueError("FP attention cannot use integer LUT normalization")
        if self.basis is BasisKind.FP and self.bias is not BiasKind.NONE:
            raise ValueError("FP P0 validation must not add relative bias")
        if (
            self.normalization is not NormalizationKind.INTEGER_LUT
            and self.row_correction is not RowCorrection.NONE
        ):
            raise ValueError("row correction only applies to integer LUT normalization")
        if self.max_bias_size < 1:
            raise ValueError("max_bias_size must be positive")
        if self.score_step <= 0:
            raise ValueError("score_step must be positive")
        if self.score_min >= 0:
            raise ValueError("score_min must be negative")
        if self.exp_bits < 2 or self.exp_bits > 30:
            raise ValueError("exp_bits must be between 2 and 30")
        if self.pwl_segments < 2:
            raise ValueError("pwl_segments must be at least 2")
        if self.relu_margin <= 0:
            raise ValueError("relu_margin must be positive")
        if self.multimax_top_k < 1:
            raise ValueError("multimax_top_k must be positive")
        if self.normalization_transition_epochs < 1:
            raise ValueError("normalization_transition_epochs must be positive")
        bdcn_values = (self.bdcn_codebook, self.bdcn_sharing, self.bdcn_projection, self.bdcn_denominator)
        if self.normalization is NormalizationKind.BDCN:
            if any(value is None for value in bdcn_values):
                raise ValueError("BDCN requires codebook, sharing, projection, and denominator")
            if self.bdcn_levels < 2 or self.bdcn_step <= 0:
                raise ValueError("BDCN levels must be at least 2 and step positive")
            if self.bdcn_distance_max is not None and self.bdcn_distance_max <= 0:
                raise ValueError("BDCN maximum distance must be positive")
            if (
                self.bdcn_codebook is BDCNCodebookKind.FIXED_EXP
                and self.bdcn_projection is not BDCNProjection.FLOAT
            ):
                raise ValueError("fixed exponential BDCN requires float projection")
            if self.bdcn_reciprocal_newton_steps not in {0, 1}:
                raise ValueError("BDCN reciprocal Newton steps must be 0 or 1")
            if (
                self.bdcn_reciprocal_newton_steps
                and self.bdcn_denominator is not BDCNDenominator.RECIPROCAL_LUT
            ):
                raise ValueError("BDCN reciprocal Newton refinement requires reciprocal LUT denominator")
        elif (
            any(value is not None for value in bdcn_values)
            or self.bdcn_levels != 16
            or self.bdcn_step != 0.125
            or self.bdcn_distance_max is not None
            or self.bdcn_reciprocal_newton_steps != 0
        ):
            raise ValueError("BDCN options only apply to BDCN normalization")
        for name, bits in (
            ("p_bits", self.p_bits),
            ("v_bits", self.v_bits),
            ("projection_weight_bits", self.projection_weight_bits),
            ("projection_activation_bits", self.projection_activation_bits),
        ):
            if bits is not None and bits < 2:
                raise ValueError(f"{name} must be at least 2")
        if (self.projection_weight_bits is None) != (self.projection_activation_bits is None):
            raise ValueError("projection weight and activation bits must be configured together")

    @property
    def resolved_bdcn_step(self) -> float:
        """Return the uniform bucket width, preserving legacy step-only configs."""

        if self.bdcn_distance_max is None:
            return self.bdcn_step
        return self.bdcn_distance_max / (self.bdcn_levels - 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in tuple(data.items()):
            if isinstance(value, Enum):
                data[key] = value.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariantConfig:
        return cls(**data)

    def to_yaml(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return destination

    @classmethod
    def from_yaml(cls, path: str | Path) -> VariantConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("variant YAML must contain a mapping")
        return cls.from_dict(data)
