"""Typed configuration at the framework's public seam."""

from __future__ import annotations

import math
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


class FixedScaleFormat(StringEnum):
    Q10 = "q10"
    POWER_OF_TWO = "power_of_two"


@dataclass(frozen=True)
class ScoreAffineConfig:
    """整合研究第 2 階段；與歷史 VariantConfig 的校準模式分開。"""

    format: FixedScaleFormat = FixedScaleFormat.Q10
    num_heads: int = 4
    num_bases: int = 2
    key_dim: int = 32
    max_size: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", FixedScaleFormat(self.format))
        if self.num_heads < 1 or not 1 <= self.num_bases <= 3 or self.key_dim != 32 or self.max_size < 1:
            raise ValueError("unsupported score affine dimensions")

    @property
    def accumulator_bits(self) -> int:
        minimum = -self.num_bases * self.key_dim * 1024 - 2 * 32768
        maximum = self.num_bases * self.key_dim * 1024 + 2 * 32767
        return max((-minimum - 1).bit_length(), maximum.bit_length()) + 1


class ReciprocalBackend(StringEnum):
    LUT256 = "lut256"
    PWL16 = "pwl16"
    NEWTON64 = "newton64"


@dataclass(frozen=True)
class ReciprocalConfig:
    """第 3 階段整數 reciprocal；mantissa Q1.16，另保留 exponent。"""

    backend: ReciprocalBackend = ReciprocalBackend.LUT256
    fraction_bits: int = 16
    probability_fraction_bits: int = 24
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, 'backend', ReciprocalBackend(self.backend))
        if (self.fraction_bits, self.probability_fraction_bits, self.max_tokens) != (16, 24, 1024):
            raise ValueError('unsupported reciprocal fixed-point contract')


@dataclass(frozen=True)
class Stage3NormalizationConfig:
    candidate: str

    def __post_init__(self) -> None:
        if self.candidate not in {*(f'N{i}' for i in range(1, 10)), 'D1', 'D2', 'D3'}:
            raise ValueError('unknown integrated Stage 3 normalization')


class BDCNCodebookKind(StringEnum):
    FIXED_EXP = "fixed_exp"
    LEARNED = "learned"
    LEARNED_ANCHORED = "learned_anchored"


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
    BIT_TRUE_PWL = "bit_true_pwl"
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
    bdcn_log_ratio_bound: float | None = None
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
        if self.normalization is NormalizationKind.BIT_TRUE_PWL and not math.isclose(
            -(self.score_min * self.score_step) / self.pwl_segments,
            0.5,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("bit-true PWL requires uniform segment width 0.5")

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
            if self.bdcn_codebook is BDCNCodebookKind.LEARNED_ANCHORED:
                if self.bdcn_log_ratio_bound is None or self.bdcn_log_ratio_bound <= 0:
                    raise ValueError("anchored BDCN requires a positive log-ratio bound")
            elif self.bdcn_log_ratio_bound is not None:
                raise ValueError("BDCN log-ratio bound only applies to an anchored learned codebook")
        elif (
            any(value is not None for value in bdcn_values)
            or self.bdcn_levels != 16
            or self.bdcn_step != 0.125
            or self.bdcn_distance_max is not None
            or self.bdcn_log_ratio_bound is not None
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


@dataclass(frozen=True)
class Stage3KDConfig:
    """本專案延伸：KD 排除 GT bat 區域，native detection loss 保持原樣。"""

    sites: tuple[int, ...] = (16, 19, 22)
    excluded_class: int = 34
    halo_cells: int = 1
    feature_gradient_target: float = .1
    calibration_batches: int = 4
    min_gain: float = .001
    max_drop: float = .001

    def __post_init__(self):
        if self.sites != (16, 19, 22) or self.excluded_class != 34 or self.halo_cells != 1:
            raise ValueError('Stage 3 KD requires reviewed sites and bat protection')
        if self.feature_gradient_target != .1 or self.calibration_batches != 4:
            raise ValueError('Stage 3 KD calibration budget is fixed')
        if self.min_gain != .001 or self.max_drop != .001:
            raise ValueError('Stage 3 KD acceptance thresholds are fixed')
