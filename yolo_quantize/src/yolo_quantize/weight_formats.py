"""Static reconstruction analysis across Full35 weight format candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .scaled_codebook import optimal_scaled_codebook_scales
from .weight_quantization import (
    Full35WeightRegionCatalog,
    ScaleMethod,
    UniformWeightSpec,
)
from .weight_views import Full35WeightViews

WeightViewName = Literal["master", "deployment"]
FixedSD4ScaleMethod = Literal[
    "max",
    "mse",
    "mse_grid",
    "mse_grid_v1",
    "optimal",
    "optimal_scaled_codebook",
]
WeightGranularity = Literal[
    "per_tensor",
    "per_output_channel",
    "group32",
    "group64",
]

_SUPPORTED_GRANULARITIES = {
    "per_tensor",
    "per_output_channel",
    "group32",
    "group64",
}
_SUPPORTED_UNIFORM_SCALE_METHODS = {
    "max",
    "mse",
    "mse_grid",
    "mse_grid_v1",
    "optimal",
    "optimal_scaled_codebook",
}
_SUPPORTED_FIXED_SD4_SCALE_METHODS = {
    "max",
    "mse",
    "mse_grid",
    "mse_grid_v1",
    "optimal",
    "optimal_scaled_codebook",
}


@dataclass(frozen=True)
class WeightAnalysisPlan:
    """Immutable CPU-only matrix for per-layer weight format analysis."""

    view_names: tuple[WeightViewName, ...] = ("master", "deployment")
    regions: tuple[str, ...] | None = None
    uniform_bits: tuple[int, ...] = (8, 7, 6, 5, 4)
    uniform_granularities: tuple[WeightGranularity, ...] = (
        "per_tensor",
        "per_output_channel",
        "group32",
        "group64",
    )
    uniform_scale_methods: tuple[ScaleMethod, ...] = ("max", "mse_grid_v1")
    fixed_sd4_granularities: tuple[WeightGranularity, ...] = (
        "per_tensor",
        "per_output_channel",
    )
    fixed_sd4_scale_methods: tuple[FixedSD4ScaleMethod, ...] = (
        "max",
        "mse_grid_v1",
        "optimal_scaled_codebook",
    )
    include_fixed_sd4: bool = True
    include_paper_twn: bool = True
    include_filterwise_twn: bool = False
    include_exact_scaled_ternary: bool = False

    def __post_init__(self) -> None:
        if not self.view_names or len(set(self.view_names)) != len(self.view_names):
            raise ValueError("weight analysis views must be non-empty and unique")
        if any(name not in {"master", "deployment"} for name in self.view_names):
            raise ValueError("weight analysis view must be master or deployment")
        if not self.uniform_bits or len(set(self.uniform_bits)) != len(
            self.uniform_bits
        ):
            raise ValueError("uniform weight bits must be non-empty and unique")
        for bits in self.uniform_bits:
            UniformWeightSpec(bits=bits)
        if not self.uniform_granularities or len(
            set(self.uniform_granularities)
        ) != len(self.uniform_granularities):
            raise ValueError("uniform granularities must be non-empty and unique")
        if any(
            granularity not in _SUPPORTED_GRANULARITIES
            for granularity in self.uniform_granularities
        ):
            raise ValueError("unsupported weight granularity")
        if not self.uniform_scale_methods or len(
            set(self.uniform_scale_methods)
        ) != len(self.uniform_scale_methods):
            raise ValueError("uniform scale methods must be non-empty and unique")
        if any(
            method not in _SUPPORTED_UNIFORM_SCALE_METHODS
            for method in self.uniform_scale_methods
        ):
            raise ValueError("unsupported weight scale method")
        if not self.fixed_sd4_granularities or len(
            set(self.fixed_sd4_granularities)
        ) != len(self.fixed_sd4_granularities):
            raise ValueError("Fixed SD4 granularities must be non-empty and unique")
        if any(
            granularity not in _SUPPORTED_GRANULARITIES
            for granularity in self.fixed_sd4_granularities
        ):
            raise ValueError("unsupported weight granularity")
        if not self.fixed_sd4_scale_methods or len(
            set(self.fixed_sd4_scale_methods)
        ) != len(self.fixed_sd4_scale_methods):
            raise ValueError("Fixed SD4 scale methods must be non-empty and unique")
        if any(
            method not in _SUPPORTED_FIXED_SD4_SCALE_METHODS
            for method in self.fixed_sd4_scale_methods
        ):
            raise ValueError("unsupported weight scale method")


@dataclass(frozen=True)
class WeightFormatMeasurement:
    """One layer in one view reconstructed with one candidate format."""

    view: WeightViewName
    path: str
    region: str
    family: str
    format_id: str
    bits: int
    granularity: str
    scale_method: str
    elements: int
    scale_count: int
    code_bytes: int
    metadata_bytes: int
    distribution: dict[str, float]
    numeric: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {
            "view": self.view,
            "path": self.path,
            "region": self.region,
            "family": self.family,
            "format_id": self.format_id,
            "bits": self.bits,
            "granularity": self.granularity,
            "scale_method": self.scale_method,
            "elements": self.elements,
            "scale_count": self.scale_count,
            "code_bytes": self.code_bytes,
            "metadata_bytes": self.metadata_bytes,
            "distribution": self.distribution,
            "numeric": self.numeric,
        }


@dataclass(frozen=True)
class WeightFormatAnalysis:
    """Complete immutable result returned by ``WeightFormatAnalyzer``."""

    schema_version: int
    measurements: tuple[WeightFormatMeasurement, ...]
    view_catalogs: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "formal_training": False,
            "gpu_used": False,
            "view_catalogs": self.view_catalogs,
            "measurements": [item.to_dict() for item in self.measurements],
        }


@dataclass(frozen=True)
class _QuantizedCandidate:
    dequantized: Tensor
    codes: Tensor
    scales: Tensor
    unclamped_codes: Tensor


def _uniform_candidate(
    weight: Tensor,
    spec: UniformWeightSpec,
    granularity: WeightGranularity,
) -> _QuantizedCandidate:
    flat = weight.reshape(weight.shape[0], -1)
    output_channels, inner = flat.shape
    padded_inner = inner
    if granularity == "per_tensor":
        groups = flat.reshape(1, -1)
    elif granularity == "per_output_channel":
        groups = flat
    else:
        group_size = 32 if granularity == "group32" else 64
        padded_inner = math.ceil(inner / group_size) * group_size
        if padded_inner != inner:
            padding = torch.zeros(
                (output_channels, padded_inner - inner),
                dtype=flat.dtype,
                device=flat.device,
            )
            padded = torch.cat((flat, padding), dim=1)
        else:
            padded = flat
        groups = padded.reshape(-1, group_size)

    maximum = groups.abs().amax(dim=1, keepdim=True)
    maximum_scales = torch.where(
        maximum > 0,
        maximum / spec.qmax,
        torch.ones_like(maximum),
    )
    scales = maximum_scales
    if spec.scale_method in {"mse", "mse_grid", "mse_grid_v1"}:
        best_error = torch.full_like(maximum, float("inf"))
        for fraction in (1.0, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
            candidate_scale = torch.where(
                maximum > 0,
                maximum_scales * fraction,
                maximum_scales,
            )
            candidate_codes = torch.round(groups / candidate_scale).clamp(
                spec.qmin,
                spec.qmax,
            )
            candidate_error = (
                (candidate_codes * candidate_scale - groups)
                .square()
                .mean(dim=1, keepdim=True)
            )
            improved = candidate_error < best_error
            best_error = torch.where(improved, candidate_error, best_error)
            scales = torch.where(improved, candidate_scale, scales)
    elif spec.scale_method in {"optimal", "optimal_scaled_codebook"}:
        codebook = torch.arange(
            spec.qmin,
            spec.qmax + 1,
            dtype=groups.dtype,
            device=groups.device,
        )
        scales = optimal_scaled_codebook_scales(groups, codebook=codebook)

    unclamped_groups = torch.round(groups / scales)
    code_groups = unclamped_groups.clamp(spec.qmin, spec.qmax)
    dequantized_groups = code_groups * scales

    def restore(grouped: Tensor) -> Tensor:
        if granularity == "per_tensor":
            restored = grouped.reshape_as(flat)
        elif granularity == "per_output_channel":
            restored = grouped
        else:
            restored = grouped.reshape(output_channels, padded_inner)[:, :inner]
        return restored.reshape_as(weight)

    return _QuantizedCandidate(
        dequantized=restore(dequantized_groups),
        codes=restore(code_groups),
        scales=scales.reshape(-1),
        unclamped_codes=restore(unclamped_groups),
    )


def _sd4_candidate(
    weight: Tensor,
    granularity: WeightGranularity,
    scale_method: FixedSD4ScaleMethod,
) -> tuple[_QuantizedCandidate, int]:
    flat = weight.reshape(weight.shape[0], -1)
    output_channels, inner = flat.shape
    padded_inner = inner
    if granularity == "per_tensor":
        groups = flat.reshape(1, -1)
    elif granularity == "per_output_channel":
        groups = flat
    else:
        group_size = 32 if granularity == "group32" else 64
        padded_inner = math.ceil(inner / group_size) * group_size
        padding = torch.zeros(
            (output_channels, padded_inner - inner),
            dtype=flat.dtype,
            device=flat.device,
        )
        groups = torch.cat((flat, padding), dim=1).reshape(-1, group_size)

    codebook = torch.tensor(
        (
            -1.0,
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.03125,
            -0.015625,
            0.0,
            0.015625,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
        ),
        dtype=groups.dtype,
        device=groups.device,
    )
    code_ids = torch.arange(-7, 8, dtype=torch.int64, device=groups.device)
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5

    def project(scale: Tensor) -> tuple[Tensor, Tensor]:
        indices = torch.bucketize(groups / scale, midpoints)
        return codebook[indices] * scale, code_ids[indices]

    maximum = groups.abs().amax(dim=1, keepdim=True)
    maximum_scales = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    scales = maximum_scales
    if scale_method in {"mse", "mse_grid", "mse_grid_v1"}:
        best_error = torch.full_like(maximum, float("inf"))
        for fraction in (1.0, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
            candidate_scale = torch.where(
                maximum > 0,
                maximum_scales * fraction,
                maximum_scales,
            )
            candidate, _ = project(candidate_scale)
            candidate_error = (candidate - groups).square().mean(dim=1, keepdim=True)
            improved = candidate_error < best_error
            best_error = torch.where(improved, candidate_error, best_error)
            scales = torch.where(improved, candidate_scale, scales)
    elif scale_method in {"optimal", "optimal_scaled_codebook"}:
        scales = optimal_scaled_codebook_scales(
            groups,
            codebook=codebook,
        )
    dequantized_groups, code_groups = project(scales)
    clipping_count = int((groups.abs() > scales).sum().item())

    def restore(grouped: Tensor) -> Tensor:
        if granularity == "per_tensor":
            restored = grouped.reshape_as(flat)
        elif granularity == "per_output_channel":
            restored = grouped
        else:
            restored = grouped.reshape(output_channels, padded_inner)[:, :inner]
        return restored.reshape_as(weight)

    codes = restore(code_groups)
    return (
        _QuantizedCandidate(
            dequantized=restore(dequantized_groups),
            codes=codes,
            scales=scales.reshape(-1),
            unclamped_codes=codes,
        ),
        clipping_count,
    )


def _paper_twn_candidate(
    weight: Tensor,
) -> tuple[_QuantizedCandidate, float, float]:
    value = weight.detach().float()
    threshold = value.abs().mean() * 0.7
    selected = value.abs() > threshold
    if bool(selected.any()):
        alpha = value.abs()[selected].mean()
    else:
        alpha = torch.zeros((), dtype=value.dtype, device=value.device)
    codes = torch.where(selected, torch.sign(value), torch.zeros_like(value)).to(
        torch.int8
    )
    dequantized = codes.to(value.dtype) * alpha
    return (
        _QuantizedCandidate(
            dequantized=dequantized,
            codes=codes,
            scales=alpha.reshape(1),
            unclamped_codes=codes,
        ),
        float(threshold.item()),
        float(alpha.item()),
    )


def _filterwise_twn_candidate(
    weight: Tensor,
    *,
    threshold_multiplier: float = 0.75,
) -> tuple[_QuantizedCandidate, Tensor, Tensor]:
    value = weight.detach().float()
    flat = value.reshape(value.shape[0], -1)
    thresholds = flat.abs().mean(dim=1, keepdim=True) * threshold_multiplier
    selected = flat.abs() > thresholds
    counts = selected.sum(dim=1, keepdim=True)
    selected_sum = torch.where(selected, flat.abs(), torch.zeros_like(flat)).sum(
        dim=1, keepdim=True
    )
    alphas = torch.where(
        counts > 0,
        selected_sum / counts.clamp_min(1),
        torch.zeros_like(selected_sum),
    )
    codes = torch.where(selected, torch.sign(flat), torch.zeros_like(flat)).to(
        torch.int8
    )
    dequantized = codes.to(flat.dtype) * alphas
    return (
        _QuantizedCandidate(
            dequantized=dequantized.reshape_as(weight),
            codes=codes.reshape_as(weight),
            scales=alphas.reshape(-1),
            unclamped_codes=codes.reshape_as(weight),
        ),
        thresholds.reshape(-1),
        alphas.reshape(-1),
    )


def _exact_scaled_ternary_candidate(weight: Tensor) -> _QuantizedCandidate:
    flat = weight.detach().float().reshape(1, -1)
    codebook = flat.new_tensor((-1.0, 0.0, 1.0))
    scales = optimal_scaled_codebook_scales(flat, codebook=codebook)
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5
    indices = torch.bucketize(flat / scales, midpoints)
    code_ids = torch.tensor((-1, 0, 1), dtype=torch.int8, device=flat.device)
    codes = code_ids[indices]
    dequantized = codes.to(flat.dtype) * scales
    return _QuantizedCandidate(
        dequantized=dequantized.reshape_as(weight),
        codes=codes.reshape_as(weight),
        scales=scales.reshape(-1),
        unclamped_codes=codes.reshape_as(weight),
    )


def _distribution(weight: Tensor) -> dict[str, float]:
    value = weight.detach().float().reshape(-1)
    return {
        "minimum": float(value.min().item()),
        "maximum": float(value.max().item()),
        "mean": float(value.mean().item()),
        "standard_deviation": float(value.std(unbiased=False).item()),
        "mean_absolute": float(value.abs().mean().item()),
        "maximum_absolute": float(value.abs().max().item()),
        "zero_ratio": float((value == 0).float().mean().item()),
    }


def _numeric(
    original: Tensor,
    candidate: Tensor,
    codes: Tensor,
    *,
    clipping_count: int,
) -> dict[str, float | int]:
    reference = original.detach().float()
    reconstructed = candidate.detach().float()
    error = reconstructed - reference
    squared_error = float(error.square().sum().item())
    reference_energy = float(reference.square().sum().item())
    candidate_energy = float(reconstructed.square().sum().item())
    dot = float((reference * reconstructed).sum().item())
    elements = reference.numel()
    occupied = int(torch.unique(codes).numel())
    return {
        "mse": squared_error / max(elements, 1),
        "normalized_rmse": math.sqrt(squared_error / max(reference_energy, 1e-24)),
        "sqnr_db": 10.0
        * math.log10(max(reference_energy, 1e-24) / max(squared_error, 1e-24)),
        "cosine": dot / math.sqrt(max(reference_energy * candidate_energy, 1e-24)),
        "maximum_absolute_error": float(error.abs().max().item()),
        "clipping_rate": clipping_count / max(elements, 1),
        "zero_ratio": float((codes == 0).float().mean().item()),
        "occupied_codes": occupied,
        "code_min": int(codes.min().item()),
        "code_max": int(codes.max().item()),
    }


class WeightFormatAnalyzer:
    """Analyze reviewed deployment weights without changing either model view."""

    def analyze(
        self,
        views: Full35WeightViews,
        plan: WeightAnalysisPlan,
    ) -> WeightFormatAnalysis:
        models = {name: getattr(views, name) for name in plan.view_names}
        return self._analyze_models(models, plan)

    def analyze_deployment(
        self,
        model: nn.Module,
        plan: WeightAnalysisPlan,
    ) -> WeightFormatAnalysis:
        """Analyze one already-folded deployment model without rebuilding views."""

        if plan.view_names != ("deployment",):
            raise ValueError(
                "direct deployment analysis requires view_names=('deployment',)"
            )
        return self._analyze_models({"deployment": model}, plan)

    def _analyze_models(
        self,
        models: Mapping[WeightViewName, nn.Module],
        plan: WeightAnalysisPlan,
    ) -> WeightFormatAnalysis:
        measurements: list[WeightFormatMeasurement] = []
        view_catalogs: dict[str, dict[str, object]] = {}
        for view_name in plan.view_names:
            model = models[view_name]
            catalog = Full35WeightRegionCatalog.inspect(model)
            view_catalogs[view_name] = catalog.summary()
            sites = tuple(
                site
                for site in catalog.deployment_sites
                if plan.regions is None or site.region in plan.regions
            )
            if not sites:
                raise ValueError(f"weight analysis selected no {view_name} sites")
            for site in sites:
                module = model.get_submodule(site.path)
                weight = module.weight.detach().float()
                distribution = _distribution(weight)
                for bits in plan.uniform_bits:
                    for granularity in plan.uniform_granularities:
                        for scale_method in plan.uniform_scale_methods:
                            spec = UniformWeightSpec(
                                bits=bits,
                                scale_method=scale_method,
                            )
                            quantized = _uniform_candidate(
                                weight,
                                spec,
                                granularity,
                            )
                            clipping_count = int(
                                (
                                    (quantized.unclamped_codes < spec.qmin)
                                    | (quantized.unclamped_codes > spec.qmax)
                                )
                                .sum()
                                .item()
                            )
                            scale_count = quantized.scales.numel()
                            measurements.append(
                                WeightFormatMeasurement(
                                    view=view_name,
                                    path=site.path,
                                    region=site.region,
                                    family="uniform",
                                    format_id=(
                                        f"uniform-w{bits}-{granularity}-{scale_method}"
                                    ),
                                    bits=bits,
                                    granularity=granularity,
                                    scale_method=scale_method,
                                    elements=weight.numel(),
                                    scale_count=scale_count,
                                    code_bytes=(weight.numel() * bits + 7) // 8,
                                    metadata_bytes=scale_count * 4,
                                    distribution=distribution,
                                    numeric=_numeric(
                                        weight,
                                        quantized.dequantized,
                                        quantized.codes,
                                        clipping_count=clipping_count,
                                    ),
                                )
                            )
                if plan.include_fixed_sd4:
                    for granularity in plan.fixed_sd4_granularities:
                        for scale_method in plan.fixed_sd4_scale_methods:
                            quantized, clipping_count = _sd4_candidate(
                                weight,
                                granularity,
                                scale_method,
                            )
                            scale_count = quantized.scales.numel()
                            measurements.append(
                                WeightFormatMeasurement(
                                    view=view_name,
                                    path=site.path,
                                    region=site.region,
                                    family="fixed_sd4",
                                    format_id=(
                                        f"fixed-sd4-{granularity}-{scale_method}"
                                    ),
                                    bits=4,
                                    granularity=granularity,
                                    scale_method=scale_method,
                                    elements=weight.numel(),
                                    scale_count=scale_count,
                                    code_bytes=(weight.numel() * 4 + 7) // 8,
                                    metadata_bytes=scale_count * 4,
                                    distribution=distribution,
                                    numeric=_numeric(
                                        weight,
                                        quantized.dequantized,
                                        quantized.codes,
                                        clipping_count=clipping_count,
                                    ),
                                )
                            )
                if plan.include_paper_twn:
                    quantized, threshold, alpha = _paper_twn_candidate(weight)
                    numeric = _numeric(
                        weight,
                        quantized.dequantized,
                        quantized.codes,
                        clipping_count=0,
                    )
                    numeric.update({"threshold": threshold, "alpha": alpha})
                    measurements.append(
                        WeightFormatMeasurement(
                            view=view_name,
                            path=site.path,
                            region=site.region,
                            family="paper_twn",
                            format_id="paper-twn-layerwise",
                            bits=2,
                            granularity="per_tensor",
                            scale_method="paper_delta_0.7_mean_abs",
                            elements=weight.numel(),
                            scale_count=1,
                            code_bytes=(weight.numel() * 2 + 7) // 8,
                            metadata_bytes=4,
                            distribution=distribution,
                            numeric=numeric,
                        )
                    )
                if plan.include_filterwise_twn:
                    quantized, thresholds, alphas = _filterwise_twn_candidate(weight)
                    numeric = _numeric(
                        weight,
                        quantized.dequantized,
                        quantized.codes,
                        clipping_count=0,
                    )
                    numeric.update(
                        {
                            "threshold_minimum": float(thresholds.min().item()),
                            "threshold_mean": float(thresholds.mean().item()),
                            "threshold_maximum": float(thresholds.max().item()),
                            "alpha_minimum": float(alphas.min().item()),
                            "alpha_mean": float(alphas.mean().item()),
                            "alpha_maximum": float(alphas.max().item()),
                        }
                    )
                    measurements.append(
                        WeightFormatMeasurement(
                            view=view_name,
                            path=site.path,
                            region=site.region,
                            family="twn_filterwise",
                            format_id="twn-v3-0.75-filterwise",
                            bits=2,
                            granularity="per_output_filter",
                            scale_method="paper_delta_0.75_filter_mean_abs",
                            elements=weight.numel(),
                            scale_count=alphas.numel(),
                            code_bytes=(weight.numel() * 2 + 7) // 8,
                            metadata_bytes=alphas.numel() * 4,
                            distribution=distribution,
                            numeric=numeric,
                        )
                    )
                if plan.include_exact_scaled_ternary:
                    quantized = _exact_scaled_ternary_candidate(weight)
                    numeric = _numeric(
                        weight,
                        quantized.dequantized,
                        quantized.codes,
                        clipping_count=0,
                    )
                    numeric.update({"alpha": float(quantized.scales[0].item())})
                    measurements.append(
                        WeightFormatMeasurement(
                            view=view_name,
                            path=site.path,
                            region=site.region,
                            family="exact_scaled_ternary",
                            format_id="exact-scaled-ternary-per_tensor",
                            bits=2,
                            granularity="per_tensor",
                            scale_method="optimal_scaled_codebook",
                            elements=weight.numel(),
                            scale_count=1,
                            code_bytes=(weight.numel() * 2 + 7) // 8,
                            metadata_bytes=4,
                            distribution=distribution,
                            numeric=numeric,
                        )
                    )
        return WeightFormatAnalysis(
            schema_version=1,
            measurements=tuple(measurements),
            view_catalogs=view_catalogs,
        )
