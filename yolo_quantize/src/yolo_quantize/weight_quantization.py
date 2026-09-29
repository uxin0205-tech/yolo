"""Fail-closed Full35 weight regions and reversible PTQ interfaces."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .scaled_codebook import optimal_scaled_codebook_scales

WeightSiteStatus = Literal["deployment", "training_only", "protected"]
ScaleMethod = Literal[
    "max",
    "mse",
    "mse_grid",
    "mse_grid_v1",
    "optimal",
    "optimal_scaled_codebook",
]


@dataclass(frozen=True)
class UniformWeightSpec:
    """Signed uniform per-output-channel weight format."""

    bits: int
    scale_method: ScaleMethod = "mse_grid_v1"

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise ValueError("weight bits must be between 2 and 16")
        if self.scale_method not in {
            "max",
            "mse",
            "mse_grid",
            "mse_grid_v1",
            "optimal",
            "optimal_scaled_codebook",
        }:
            raise ValueError("unsupported uniform weight scale_method")

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def format_id(self) -> str:
        return f"w{self.bits}"


@dataclass(frozen=True)
class FixedSD4WeightSpec:
    """Four-bit signed dyadic codebook with one scale per output channel."""

    scale_method: ScaleMethod = "optimal_scaled_codebook"
    bits: int = 4

    def __post_init__(self) -> None:
        UniformWeightSpec(bits=4, scale_method=self.scale_method)
        if self.bits != 4:
            raise ValueError("Fixed SD4 is always encoded with four bits")

    @property
    def qmin(self) -> int:
        return -7

    @property
    def qmax(self) -> int:
        return 7

    @property
    def format_id(self) -> str:
        return "fixed-sd4"


@dataclass(frozen=True)
class ExactTernaryWeightSpec:
    """Globally MSE-optimal per-tensor {-scale, 0, +scale} PTQ."""

    scale_method: str = "optimal_scaled_codebook"
    bits: int = 2

    def __post_init__(self) -> None:
        if self.scale_method != "optimal_scaled_codebook":
            raise ValueError("exact ternary requires optimal_scaled_codebook")
        if self.bits != 2:
            raise ValueError("exact ternary is encoded with two bits")

    @property
    def qmin(self) -> int:
        return -1

    @property
    def qmax(self) -> int:
        return 1

    @property
    def format_id(self) -> str:
        return "exact-scaled-ternary"


@dataclass(frozen=True)
class FilterwiseTWNWeightSpec:
    """Paper-TWN threshold and alpha independently fitted per output filter."""

    threshold_multiplier: float = 0.75
    bits: int = 2

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.threshold_multiplier)
            or self.threshold_multiplier <= 0.0
        ):
            raise ValueError("filter-wise TWN threshold_multiplier must be positive")
        if self.bits != 2:
            raise ValueError("filter-wise TWN is encoded with two bits")

    @property
    def qmin(self) -> int:
        return -1

    @property
    def qmax(self) -> int:
        return 1

    @property
    def format_id(self) -> str:
        if math.isclose(self.threshold_multiplier, 0.75):
            return "twn-v3-0.75-filterwise"
        if math.isclose(self.threshold_multiplier, 0.7):
            return "twn-v2-0.70-filterwise"
        return f"twn-{self.threshold_multiplier:g}-filterwise"


@dataclass(frozen=True)
class PaperTWNWeightSpec:
    """Layerwise Paper-TWN PTQ control using {-alpha, 0, +alpha}."""

    threshold_multiplier: float = 0.7
    bits: int = 2

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.threshold_multiplier)
            or self.threshold_multiplier <= 0.0
        ):
            raise ValueError("Paper-TWN threshold_multiplier must be positive")
        if self.bits != 2:
            raise ValueError("Paper-TWN is encoded with two bits")

    @property
    def qmin(self) -> int:
        return -1

    @property
    def qmax(self) -> int:
        return 1

    @property
    def format_id(self) -> str:
        return "paper-twn"


WeightFormatSpec = (
    UniformWeightSpec
    | FixedSD4WeightSpec
    | PaperTWNWeightSpec
    | FilterwiseTWNWeightSpec
    | ExactTernaryWeightSpec
)


@dataclass(frozen=True)
class WeightRegionAssignment:
    """One reviewed region-to-format mapping inside a combined policy."""

    region: str
    spec: WeightFormatSpec
    paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.region.strip():
            raise ValueError("weight region assignment must not be empty")
        if not isinstance(self.paths, tuple):
            raise TypeError("weight assignment paths must be a tuple")
        if any(not path.strip() for path in self.paths):
            raise ValueError("weight assignment paths must not be empty")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("weight assignment paths must be unique")


@dataclass(frozen=True)
class WeightSite:
    """One Conv/Linear weight tensor with an explicit deployment disposition."""

    path: str
    region: str
    status: WeightSiteStatus
    module_type: str
    shape: tuple[int, ...]
    elements: int


@dataclass(frozen=True)
class Full35WeightRegionCatalog:
    """Complete, fail-closed classification of Full35 Conv/Linear weights."""

    sites: tuple[WeightSite, ...]

    @classmethod
    def inspect(cls, model: nn.Module) -> Full35WeightRegionCatalog:
        sites: list[WeightSite] = []
        for path, module in model.named_modules(remove_duplicate=False):
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            disposition = _classify_full35_weight_path(path)
            if disposition is None:
                raise ValueError(f"unmapped Full35 weight module: {path}")
            status, region = disposition
            sites.append(
                WeightSite(
                    path=path,
                    region=region,
                    status=status,
                    module_type=f"{type(module).__module__}.{type(module).__name__}",
                    shape=tuple(module.weight.shape),
                    elements=module.weight.numel(),
                )
            )
        if not sites:
            raise ValueError("model contains no Conv2d or Linear weight modules")
        return cls(tuple(sites))

    @property
    def deployment_sites(self) -> tuple[WeightSite, ...]:
        return tuple(site for site in self.sites if site.status == "deployment")

    @property
    def training_only_sites(self) -> tuple[WeightSite, ...]:
        return tuple(site for site in self.sites if site.status == "training_only")

    @property
    def protected_sites(self) -> tuple[WeightSite, ...]:
        return tuple(site for site in self.sites if site.status == "protected")

    @property
    def deployment_weight_elements(self) -> int:
        return sum(site.elements for site in self.deployment_sites)

    def summary(self) -> dict[str, object]:
        """Return stable totals and per-deployment-region coverage."""

        regions: dict[str, dict[str, int]] = {}
        for site in self.deployment_sites:
            entry = regions.setdefault(
                site.region,
                {"modules": 0, "weight_elements": 0},
            )
            entry["modules"] += 1
            entry["weight_elements"] += site.elements
        return {
            "totals": {
                "modules": len(self.sites),
                "weight_elements": sum(site.elements for site in self.sites),
                "deployment_modules": len(self.deployment_sites),
                "deployment_weight_elements": self.deployment_weight_elements,
                "training_only_modules": len(self.training_only_sites),
                "training_only_weight_elements": sum(
                    site.elements for site in self.training_only_sites
                ),
                "protected_modules": len(self.protected_sites),
                "protected_weight_elements": sum(
                    site.elements for site in self.protected_sites
                ),
            },
            "deployment_regions": regions,
        }


@dataclass(frozen=True)
class AppliedWeightQuantization:
    """One reversible, isolated-region PTQ application."""

    region: str
    spec: WeightFormatSpec
    sites: tuple[WeightSite, ...]
    numeric: dict[str, object]
    site_metrics: tuple[dict[str, object], ...]

    @property
    def quantized_modules(self) -> int:
        return len(self.sites)

    @property
    def weight_elements(self) -> int:
        return sum(site.elements for site in self.sites)

    @property
    def format_id(self) -> str:
        return self.spec.format_id


@dataclass(frozen=True)
class AppliedWeightQuantizationPolicy:
    """All disjoint region quantizers active in one reversible context."""

    assignments: tuple[WeightRegionAssignment, ...]
    applied: tuple[AppliedWeightQuantization, ...]

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(assignment.region for assignment in self.assignments)

    @property
    def quantized_modules(self) -> int:
        return sum(item.quantized_modules for item in self.applied)

    @property
    def weight_elements(self) -> int:
        return sum(item.weight_elements for item in self.applied)

    @property
    def weight_code_bytes(self) -> int:
        return sum(int(item.numeric["weight_code_bytes"]) for item in self.applied)

    @property
    def scale_bytes(self) -> int:
        return sum(int(item.numeric["scale_bytes"]) for item in self.applied)

    @property
    def assignment_records(self) -> tuple[dict[str, object], ...]:
        """Return the exact selectors and formats needed to reproduce the policy."""

        records: list[dict[str, object]] = []
        for assignment in self.assignments:
            format_record: dict[str, object] = {
                "format_id": assignment.spec.format_id,
                "encoded_bits": assignment.spec.bits,
            }
            if isinstance(
                assignment.spec,
                (UniformWeightSpec, FixedSD4WeightSpec, ExactTernaryWeightSpec),
            ):
                format_record["scale_method"] = assignment.spec.scale_method
            else:
                format_record["threshold_multiplier"] = (
                    assignment.spec.threshold_multiplier
                )
                format_record["granularity"] = (
                    "per_output_filter"
                    if isinstance(assignment.spec, FilterwiseTWNWeightSpec)
                    else "per_tensor"
                )
            records.append(
                {
                    "region": assignment.region,
                    "selector": {
                        "kind": "paths" if assignment.paths else "region",
                        "paths": list(assignment.paths),
                    },
                    "format": format_record,
                }
            )
        return tuple(records)


@dataclass(frozen=True)
class _QuantizedWeight:
    dequantized: Tensor
    codes: Tensor
    scales: Tensor
    unclamped_codes: Tensor
    clipping_count: int


class WeightQuantizationAdapter:
    """Temporarily apply weight-only PTQ to one reviewed deployment region."""

    @contextmanager
    def quantized(
        self,
        model: nn.Module,
        *,
        catalog: Full35WeightRegionCatalog,
        region: str,
        spec: WeightFormatSpec,
        paths: tuple[str, ...] = (),
    ) -> Iterator[AppliedWeightQuantization]:
        region_sites = tuple(
            site for site in catalog.deployment_sites if site.region == region
        )
        if not region_sites:
            raise ValueError(f"unknown or empty deployment weight region: {region}")
        if len(set(paths)) != len(paths):
            raise ValueError("weight quantization paths must be unique")
        if paths:
            by_path = {site.path: site for site in region_sites}
            unknown = tuple(path for path in paths if path not in by_path)
            if unknown:
                raise ValueError(
                    f"weight paths are not deployment sites in {region}: "
                    + ",".join(unknown)
                )
            selected_paths = set(paths)
            sites = tuple(site for site in region_sites if site.path in selected_paths)
        else:
            sites = region_sites

        modules: dict[str, nn.Conv2d | nn.Linear] = {}
        originals: dict[str, Tensor] = {}
        for site in sites:
            try:
                module = model.get_submodule(site.path)
            except AttributeError as error:
                raise ValueError(f"stale weight path: {site.path}") from error
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                raise TypeError(f"weight path changed module type: {site.path}")
            if tuple(module.weight.shape) != site.shape:
                raise ValueError(f"weight path changed shape: {site.path}")
            modules[site.path] = module
            originals[site.path] = module.weight.detach().clone()

        total_elements = 0
        squared_error = 0.0
        original_energy = 0.0
        candidate_energy = 0.0
        dot_product = 0.0
        maximum_error = 0.0
        clipping_count = 0
        scale_count = 0
        scale_sum = 0.0
        scale_minimum = math.inf
        scale_maximum = 0.0
        logical_code_count = spec.qmax - spec.qmin + 1
        histogram = torch.zeros(logical_code_count, dtype=torch.int64)
        site_metrics: list[dict[str, object]] = []

        try:
            for site in sites:
                module = modules[site.path]
                original = module.weight.detach().float()
                quantized = _quantize_weight(original, spec)
                error = quantized.dequantized - original
                elements = original.numel()
                site_squared_error = float(error.square().sum().item())
                site_energy = float(original.square().sum().item())
                site_candidate_energy = float(
                    quantized.dequantized.square().sum().item()
                )
                site_dot = float((original * quantized.dequantized).sum().item())
                site_clipping = quantized.clipping_count
                site_histogram = torch.bincount(
                    (quantized.codes.to(torch.int64) - spec.qmin).reshape(-1).cpu(),
                    minlength=logical_code_count,
                )
                histogram += site_histogram
                scales = quantized.scales.float()
                site_metrics.append(
                    {
                        "path": site.path,
                        "region": site.region,
                        "elements": elements,
                        "output_channels": scales.numel(),
                        "mse": site_squared_error / max(elements, 1),
                        "normalized_rmse": math.sqrt(
                            site_squared_error / max(site_energy, 1e-24)
                        ),
                        "sqnr_db": 10.0
                        * math.log10(
                            max(site_energy, 1e-24) / max(site_squared_error, 1e-24)
                        ),
                        "cosine": site_dot
                        / math.sqrt(max(site_energy * site_candidate_energy, 1e-24)),
                        "maximum_absolute_error": float(error.abs().max().item()),
                        "clipping_rate": site_clipping / max(elements, 1),
                        "zero_ratio": int(site_histogram[-spec.qmin].item())
                        / max(elements, 1),
                        "occupied_codes": int((site_histogram > 0).sum().item()),
                        "code_min": int(quantized.codes.min().item()),
                        "code_max": int(quantized.codes.max().item()),
                        "scale_minimum": float(scales.min().item()),
                        "scale_mean": float(scales.mean().item()),
                        "scale_maximum": float(scales.max().item()),
                    }
                )
                total_elements += elements
                squared_error += site_squared_error
                original_energy += site_energy
                candidate_energy += site_candidate_energy
                dot_product += site_dot
                maximum_error = max(
                    maximum_error,
                    float(error.abs().max().item()),
                )
                clipping_count += site_clipping
                scale_count += scales.numel()
                scale_sum += float(scales.sum().item())
                scale_minimum = min(scale_minimum, float(scales.min().item()))
                scale_maximum = max(scale_maximum, float(scales.max().item()))
                with torch.no_grad():
                    module.weight.copy_(
                        quantized.dequantized.to(dtype=module.weight.dtype)
                    )

            occupied = histogram > 0
            occupied_indices = torch.nonzero(occupied).flatten()
            numeric: dict[str, object] = {
                "mse": squared_error / max(total_elements, 1),
                "normalized_rmse": math.sqrt(
                    squared_error / max(original_energy, 1e-24)
                ),
                "sqnr_db": 10.0
                * math.log10(max(original_energy, 1e-24) / max(squared_error, 1e-24)),
                "cosine": dot_product
                / math.sqrt(max(original_energy * candidate_energy, 1e-24)),
                "maximum_absolute_error": maximum_error,
                "clipping_rate": clipping_count / max(total_elements, 1),
                "zero_ratio": int(histogram[-spec.qmin].item())
                / max(total_elements, 1),
                "occupied_codes": int(occupied.sum().item()),
                "code_occupancy_ratio": float(occupied.float().mean().item()),
                "code_min": int(occupied_indices[0].item()) + spec.qmin,
                "code_max": int(occupied_indices[-1].item()) + spec.qmin,
                "code_histogram": histogram.tolist(),
                "encoded_bits": spec.bits,
                "logical_code_count": logical_code_count,
                "scale_count": scale_count,
                "scale_minimum": scale_minimum,
                "scale_mean": scale_sum / max(scale_count, 1),
                "scale_maximum": scale_maximum,
                "weight_code_bytes": (total_elements * spec.bits + 7) // 8,
                "scale_bytes": scale_count * 4,
            }
            yield AppliedWeightQuantization(
                region=region,
                spec=spec,
                sites=sites,
                numeric=numeric,
                site_metrics=tuple(site_metrics),
            )
        finally:
            with torch.no_grad():
                for path, original in originals.items():
                    modules[path].weight.copy_(original)

    @contextmanager
    def quantized_policy(
        self,
        model: nn.Module,
        *,
        catalog: Full35WeightRegionCatalog,
        assignments: tuple[WeightRegionAssignment, ...],
    ) -> Iterator[AppliedWeightQuantizationPolicy]:
        """Apply several disjoint region formats and restore all weights on exit."""

        if not assignments:
            raise ValueError("weight quantization policy requires assignments")
        known_regions = {site.region for site in catalog.deployment_sites}
        unknown = tuple(
            assignment.region
            for assignment in assignments
            if assignment.region not in known_regions
        )
        if unknown:
            raise ValueError(
                "unknown or empty deployment weight regions: " + ",".join(unknown)
            )
        ordered_paths_by_region = {
            region: tuple(
                site.path for site in catalog.deployment_sites if site.region == region
            )
            for region in known_regions
        }
        paths_by_region = {
            region: set(paths) for region, paths in ordered_paths_by_region.items()
        }
        defaults_by_region: dict[str, WeightRegionAssignment] = {}
        explicit_paths_by_region: dict[str, set[str]] = {
            region: set() for region in known_regions
        }
        for assignment in assignments:
            if not assignment.paths:
                if assignment.region in defaults_by_region:
                    raise ValueError("weight quantization policy sites must be unique")
                defaults_by_region[assignment.region] = assignment
                continue
            selected = set(assignment.paths)
            invalid = selected - paths_by_region[assignment.region]
            if invalid:
                raise ValueError(
                    f"weight paths are not deployment sites in {assignment.region}: "
                    + ",".join(sorted(invalid))
                )
            claimed = explicit_paths_by_region[assignment.region]
            if claimed & selected:
                raise ValueError("weight quantization policy sites must be unique")
            claimed.update(selected)

        effective_assignments: list[WeightRegionAssignment] = []
        for assignment in assignments:
            if assignment.paths:
                effective_assignments.append(assignment)
                continue
            overridden = explicit_paths_by_region[assignment.region]
            if not overridden:
                effective_assignments.append(assignment)
                continue
            remaining = tuple(
                path
                for path in ordered_paths_by_region[assignment.region]
                if path not in overridden
            )
            if remaining:
                effective_assignments.append(
                    WeightRegionAssignment(
                        region=assignment.region,
                        spec=assignment.spec,
                        paths=remaining,
                    )
                )

        with ExitStack() as stack:
            applied = tuple(
                stack.enter_context(
                    self.quantized(
                        model,
                        catalog=catalog,
                        region=assignment.region,
                        spec=assignment.spec,
                        paths=assignment.paths,
                    )
                )
                for assignment in effective_assignments
            )
            yield AppliedWeightQuantizationPolicy(
                assignments=tuple(effective_assignments),
                applied=applied,
            )


def _uniform_quantize(
    weight: Tensor,
    spec: UniformWeightSpec,
) -> _QuantizedWeight:
    flat = weight.reshape(weight.shape[0], -1)
    maximum = flat.abs().amax(dim=1, keepdim=True)
    maximum_scales = torch.where(
        maximum > 0,
        maximum / spec.qmax,
        torch.ones_like(maximum),
    )
    scales = maximum_scales
    if spec.scale_method in {"mse", "mse_grid", "mse_grid_v1"}:
        best_error = torch.full_like(maximum, float("inf"))
        for fraction in (
            1.0,
            0.98,
            0.95,
            0.9,
            0.85,
            0.8,
            0.75,
            0.7,
            0.6,
            0.5,
        ):
            candidate = torch.where(
                maximum > 0,
                maximum_scales * fraction,
                maximum_scales,
            )
            candidate_codes = torch.round(flat / candidate).clamp(
                spec.qmin,
                spec.qmax,
            )
            candidate_error = (
                (candidate_codes * candidate - flat).square().mean(dim=1, keepdim=True)
            )
            improved = candidate_error < best_error
            best_error = torch.where(improved, candidate_error, best_error)
            scales = torch.where(improved, candidate, scales)
    elif spec.scale_method in {"optimal", "optimal_scaled_codebook"}:
        codebook = torch.arange(
            spec.qmin,
            spec.qmax + 1,
            dtype=flat.dtype,
            device=flat.device,
        )
        scales = optimal_scaled_codebook_scales(flat, codebook=codebook)
    unclamped = torch.round(flat / scales)
    codes = unclamped.clamp(spec.qmin, spec.qmax)
    dequantized = (codes * scales).reshape_as(weight)
    return _QuantizedWeight(
        dequantized=dequantized,
        codes=codes.reshape_as(weight),
        scales=scales.reshape(-1),
        unclamped_codes=unclamped.reshape_as(weight),
        clipping_count=int(
            ((unclamped < spec.qmin) | (unclamped > spec.qmax)).sum().item()
        ),
    )


def _fixed_sd4_quantize(
    weight: Tensor,
    spec: FixedSD4WeightSpec,
) -> _QuantizedWeight:
    flat = weight.reshape(weight.shape[0], -1)
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
        dtype=flat.dtype,
        device=flat.device,
    )
    logical_codes = torch.arange(-7, 8, dtype=torch.int64, device=flat.device)
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5

    def project(scale: Tensor) -> tuple[Tensor, Tensor]:
        indices = torch.bucketize(flat / scale, midpoints)
        return codebook[indices] * scale, logical_codes[indices]

    maximum = flat.abs().amax(dim=1, keepdim=True)
    maximum_scales = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    scales = maximum_scales
    if spec.scale_method in {"mse", "mse_grid", "mse_grid_v1"}:
        best_error = torch.full_like(maximum, float("inf"))
        for fraction in (1.0, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
            candidate_scale = torch.where(
                maximum > 0,
                maximum_scales * fraction,
                maximum_scales,
            )
            candidate, _ = project(candidate_scale)
            error = (candidate - flat).square().mean(dim=1, keepdim=True)
            improved = error < best_error
            best_error = torch.where(improved, error, best_error)
            scales = torch.where(improved, candidate_scale, scales)
    elif spec.scale_method in {"optimal", "optimal_scaled_codebook"}:
        scales = optimal_scaled_codebook_scales(flat, codebook=codebook)
    dequantized, codes = project(scales)
    return _QuantizedWeight(
        dequantized=dequantized.reshape_as(weight),
        codes=codes.reshape_as(weight),
        scales=scales.reshape(-1),
        unclamped_codes=codes.reshape_as(weight),
        clipping_count=int((flat.abs() > scales).sum().item()),
    )


def _exact_ternary_quantize(
    weight: Tensor,
    spec: ExactTernaryWeightSpec,
) -> _QuantizedWeight:
    flat = weight.detach().float().reshape(1, -1)
    codebook = flat.new_tensor((-1.0, 0.0, 1.0))
    scales = optimal_scaled_codebook_scales(flat, codebook=codebook)
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5
    indices = torch.bucketize(flat / scales, midpoints)
    codes = torch.tensor((-1, 0, 1), device=flat.device, dtype=torch.int8)[indices]
    dequantized = codes.to(flat.dtype) * scales
    return _QuantizedWeight(
        dequantized=dequantized.reshape_as(weight),
        codes=codes.reshape_as(weight),
        scales=scales.reshape(-1),
        unclamped_codes=codes.reshape_as(weight),
        clipping_count=0,
    )


def _filterwise_twn_quantize(
    weight: Tensor,
    spec: FilterwiseTWNWeightSpec,
) -> _QuantizedWeight:
    value = weight.detach().float()
    flat = value.reshape(value.shape[0], -1)
    threshold = flat.abs().mean(dim=1, keepdim=True) * spec.threshold_multiplier
    selected = flat.abs() > threshold
    selected_count = selected.sum(dim=1, keepdim=True)
    selected_sum = torch.where(selected, flat.abs(), torch.zeros_like(flat)).sum(
        dim=1, keepdim=True
    )
    alpha = torch.where(
        selected_count > 0,
        selected_sum / selected_count.clamp_min(1),
        torch.zeros_like(selected_sum),
    )
    codes = torch.where(selected, torch.sign(flat), torch.zeros_like(flat)).to(
        torch.int8
    )
    dequantized = codes.to(flat.dtype) * alpha
    return _QuantizedWeight(
        dequantized=dequantized.reshape_as(weight),
        codes=codes.reshape_as(weight),
        scales=alpha.reshape(-1),
        unclamped_codes=codes.reshape_as(weight),
        clipping_count=0,
    )


def _paper_twn_quantize(
    weight: Tensor,
    spec: PaperTWNWeightSpec,
) -> _QuantizedWeight:
    value = weight.detach().float()
    threshold = value.abs().mean() * spec.threshold_multiplier
    selected = value.abs() > threshold
    alpha = (
        value.abs()[selected].mean() if bool(selected.any()) else value.new_zeros(())
    )
    codes = torch.where(selected, torch.sign(value), torch.zeros_like(value)).to(
        torch.int8
    )
    return _QuantizedWeight(
        dequantized=codes.to(value.dtype) * alpha,
        codes=codes,
        scales=alpha.reshape(1),
        unclamped_codes=codes,
        clipping_count=0,
    )


def _quantize_weight(weight: Tensor, spec: WeightFormatSpec) -> _QuantizedWeight:
    if isinstance(spec, UniformWeightSpec):
        return _uniform_quantize(weight, spec)
    if isinstance(spec, FixedSD4WeightSpec):
        return _fixed_sd4_quantize(weight, spec)
    if isinstance(spec, PaperTWNWeightSpec):
        return _paper_twn_quantize(weight, spec)
    if isinstance(spec, FilterwiseTWNWeightSpec):
        return _filterwise_twn_quantize(weight, spec)
    if isinstance(spec, ExactTernaryWeightSpec):
        return _exact_ternary_quantize(weight, spec)
    raise TypeError(f"unsupported weight format spec: {type(spec).__name__}")


def _classify_full35_weight_path(
    path: str,
) -> tuple[WeightSiteStatus, str] | None:
    if re.search(r"\.attn\.qkv\.[qk]\.conv$", path):
        return "protected", "binary_qk"

    detect = "graph.model.23.detect_head."
    if path.startswith(detect):
        if not path.startswith(detect + "one2one_"):
            return "training_only", "detect_one2many"
        predictor = re.search(r"one2one_cv[23]\.[0-2]\.2$", path)
        return (
            "deployment",
            "detect_one2one_predictor" if predictor else "detect_one2one_tower",
        )

    pose = "graph.model.23.pose_head."
    if path.startswith(pose):
        if not path.startswith(pose + "one2one_"):
            region = "pose_flow" if ".flow_model." in path else "pose_one2many"
            return "training_only", region
        if path.startswith(pose + "one2one_cv4_sigma."):
            return "training_only", "pose_sigma"
        predictor = re.search(
            r"one2one_(?:cv[23]\.[0-2]\.2|cv4_kpts\.[0-2])$",
            path,
        )
        return (
            "deployment",
            "pose_one2one_predictor" if predictor else "pose_one2one_tower",
        )

    if path.startswith("graph.model.16.p3_masf."):
        return "deployment", "masf"
    if path.startswith("graph.model.10."):
        return "deployment", "backbone_attention_safe"
    if path.startswith("graph.model.22.m.0.1."):
        return "deployment", "neck_attention_safe"
    if re.match(r"^graph\.model\.[0-4]\.", path):
        return "deployment", "backbone_early"
    if re.match(r"^graph\.model\.[5-9]\.", path):
        return "deployment", "backbone_deep"
    if re.match(r"^graph\.model\.(13|16|17|19|20|22)\.", path):
        return "deployment", "neck"
    return None
