"""Fold-aware, trainable weight fake quantization for reviewed Full35 policies."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .quantizers import grad_scale, round_to_nearest_even_ste
from .weight_quantization import (
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    Full35WeightRegionCatalog,
    PaperTWNWeightSpec,
    UniformWeightSpec,
    WeightFormatSpec,
    WeightRegionAssignment,
    WeightSite,
    _quantize_weight,
)


def _inverse_softplus(value: Tensor) -> Tensor:
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value)))


@dataclass(frozen=True)
class ProgressiveQuantizationSchedule:
    """Linear fake-quant blend schedule with explicit FP and full-QAT epochs."""

    start_epoch: int
    full_epoch: int

    def __post_init__(self) -> None:
        if self.start_epoch < 0:
            raise ValueError("start_epoch cannot be negative")
        if self.full_epoch <= self.start_epoch:
            raise ValueError("full_epoch must be greater than start_epoch")

    def ratio(self, epoch: int) -> float:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if epoch <= self.start_epoch:
            return 0.0
        if epoch >= self.full_epoch:
            return 1.0
        return (epoch - self.start_epoch) / (self.full_epoch - self.start_epoch)


class TrainableWeightFakeQuantizer(nn.Module):
    """STE projector with a positive learned scale and progressive blend."""

    def __init__(
        self,
        spec: WeightFormatSpec,
        *,
        initial_scales: Tensor,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if initial_scales.ndim != 1 or initial_scales.numel() < 1:
            raise ValueError("initial_scales must be a non-empty rank-one tensor")
        if not torch.isfinite(initial_scales).all() or bool(
            (initial_scales <= eps).any()
        ):
            raise ValueError("initial_scales must be finite and greater than eps")
        self.spec = spec
        self.eps = float(eps)
        unconstrained = _inverse_softplus(initial_scales.detach().float() - self.eps)
        self._scale_unconstrained = nn.Parameter(unconstrained)
        self.register_buffer("_blend_ratio", torch.tensor(0.0))

    @classmethod
    def from_weight(
        cls,
        weight: Tensor,
        spec: WeightFormatSpec,
    ) -> Self:
        quantized = _quantize_weight(weight.detach().float(), spec)
        scales = quantized.scales.detach().float().reshape(-1)
        scales = torch.where(
            scales > 1e-8,
            scales,
            torch.full_like(scales, 1e-8 * 2.0),
        )
        return cls(spec, initial_scales=scales).to(device=weight.device)

    def reinitialize_from_weight(self, weight: Tensor) -> None:
        """Refit scale in place after loading a different FP32 shadow weight."""

        quantized = _quantize_weight(weight.detach().float(), self.spec)
        scales = quantized.scales.detach().float().reshape(-1)
        scales = torch.where(
            scales > self.eps,
            scales,
            torch.full_like(scales, self.eps * 2.0),
        )
        if scales.shape != self._scale_unconstrained.shape:
            raise ValueError(
                "warm-start weight scale count differs from the QAT format"
            )
        unconstrained = _inverse_softplus(scales - self.eps).to(
            device=self._scale_unconstrained.device,
            dtype=self._scale_unconstrained.dtype,
        )
        with torch.no_grad():
            self._scale_unconstrained.copy_(unconstrained)
            self._blend_ratio.zero_()

    @property
    def scale(self) -> Tensor:
        return F.softplus(self._scale_unconstrained) + self.eps

    @property
    def scale_parameter(self) -> nn.Parameter:
        return self._scale_unconstrained

    @property
    def blend_ratio(self) -> float:
        return float(self._blend_ratio.item())

    def set_blend_ratio(self, ratio: float) -> None:
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("blend ratio must be finite and in [0, 1]")
        self._blend_ratio.fill_(ratio)

    def _channel_scales(self, weight: Tensor) -> Tensor:
        if self.scale.numel() == 1:
            shape = (1,) * weight.ndim
        elif self.scale.numel() == weight.shape[0]:
            shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
        else:
            raise RuntimeError(
                "weight output channels differ from initialized QAT scale count"
            )
        values_per_scale = weight.numel() // self.scale.numel()
        factor = 1.0 / math.sqrt(max(1, values_per_scale) * max(1, int(self.spec.qmax)))
        return grad_scale(self.scale, factor).reshape(shape).to(dtype=weight.dtype)

    def _uniform(self, weight: Tensor, scale: Tensor) -> Tensor:
        spec = self.spec
        if not isinstance(spec, UniformWeightSpec):
            raise TypeError(type(spec).__name__)
        normalized = weight / scale
        codes = round_to_nearest_even_ste(normalized).clamp(spec.qmin, spec.qmax)
        return codes * scale

    def _fixed_sd4(self, weight: Tensor, scale: Tensor) -> Tensor:
        normalized = weight / scale
        codebook = weight.new_tensor(
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
            )
        )
        indices = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
        nearest = codebook[indices]
        codes = normalized + (nearest - normalized).detach()
        return codes * scale

    def _paper_twn(self, weight: Tensor, scale: Tensor) -> Tensor:
        spec = self.spec
        if not isinstance(spec, PaperTWNWeightSpec):
            raise TypeError(type(spec).__name__)
        normalized = weight / scale
        threshold = weight.detach().abs().mean() * spec.threshold_multiplier
        nearest = torch.where(
            weight.detach().abs() > threshold,
            torch.sign(weight.detach()),
            torch.zeros_like(weight),
        )
        codes = normalized + (nearest - normalized).detach()
        return codes * scale

    def _exact_ternary(self, weight: Tensor, scale: Tensor) -> Tensor:
        normalized = weight / scale
        detached = normalized.detach()
        nearest = torch.where(
            detached > 0.5,
            torch.ones_like(detached),
            torch.where(
                detached < -0.5,
                -torch.ones_like(detached),
                torch.zeros_like(detached),
            ),
        )
        codes = normalized + (nearest - normalized).detach()
        return codes * scale

    def _filterwise_twn(self, weight: Tensor, scale: Tensor) -> Tensor:
        spec = self.spec
        if not isinstance(spec, FilterwiseTWNWeightSpec):
            raise TypeError(type(spec).__name__)
        normalized = weight / scale
        flat = weight.detach().reshape(weight.shape[0], -1)
        threshold = flat.abs().mean(dim=1, keepdim=True) * spec.threshold_multiplier
        threshold = threshold.reshape((weight.shape[0],) + (1,) * (weight.ndim - 1))
        nearest = torch.where(
            weight.detach().abs() > threshold,
            torch.sign(weight.detach()),
            torch.zeros_like(weight),
        )
        codes = normalized + (nearest - normalized).detach()
        return codes * scale

    def forward(self, weight: Tensor) -> Tensor:
        scale = self._channel_scales(weight)
        if isinstance(self.spec, UniformWeightSpec):
            quantized = self._uniform(weight, scale)
        elif isinstance(self.spec, FixedSD4WeightSpec):
            quantized = self._fixed_sd4(weight, scale)
        elif isinstance(self.spec, PaperTWNWeightSpec):
            quantized = self._paper_twn(weight, scale)
        elif isinstance(self.spec, ExactTernaryWeightSpec):
            quantized = self._exact_ternary(weight, scale)
        elif isinstance(self.spec, FilterwiseTWNWeightSpec):
            quantized = self._filterwise_twn(weight, scale)
        else:  # pragma: no cover - closed by WeightFormatSpec
            raise TypeError(
                f"unsupported QAT weight format: {type(self.spec).__name__}"
            )
        ratio = self._blend_ratio.to(dtype=weight.dtype)
        return weight + ratio * (quantized - weight)


class QATConv2d(nn.Conv2d):
    """Conv2d retaining an FP32 master weight while fake-quantizing forward."""

    @classmethod
    def from_module(cls, module: nn.Conv2d, spec: WeightFormatSpec) -> QATConv2d:
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError("QAT refuses Conv2d modules with runtime hooks")
        result = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        result.weight = module.weight
        result.bias = module.bias
        result.weight_quantizer = TrainableWeightFakeQuantizer.from_weight(
            module.weight,
            spec,
        )
        result.train(module.training)
        return result

    def forward(self, value: Tensor) -> Tensor:
        return self._conv_forward(
            value,
            self.weight_quantizer(self.weight),
            self.bias,
        )


class QATLinear(nn.Linear):
    """Linear retaining an FP32 master weight while fake-quantizing forward."""

    @classmethod
    def from_module(cls, module: nn.Linear, spec: WeightFormatSpec) -> QATLinear:
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError("QAT refuses Linear modules with runtime hooks")
        result = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        result.weight = module.weight
        result.bias = module.bias
        result.weight_quantizer = TrainableWeightFakeQuantizer.from_weight(
            module.weight,
            spec,
        )
        result.train(module.training)
        return result

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, self.weight_quantizer(self.weight), self.bias)


def _set_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = replacement
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = replacement
    else:
        setattr(parent, child_name, replacement)


def _resolve_sites(
    catalog: Full35WeightRegionCatalog,
    assignments: tuple[WeightRegionAssignment, ...],
) -> tuple[tuple[WeightSite, WeightFormatSpec], ...]:
    if not assignments:
        raise ValueError("QAT weight policy requires assignments")
    sites_by_region: dict[str, tuple[WeightSite, ...]] = {}
    for site in catalog.deployment_sites:
        sites_by_region.setdefault(site.region, ())
        sites_by_region[site.region] += (site,)
    claimed: dict[str, WeightFormatSpec] = {}
    defaults: dict[str, WeightFormatSpec] = {}
    for assignment in assignments:
        region_sites = sites_by_region.get(assignment.region)
        if not region_sites:
            raise ValueError(
                f"unknown or empty deployment weight region: {assignment.region}"
            )
        if assignment.paths:
            known = {site.path for site in region_sites}
            invalid = set(assignment.paths) - known
            if invalid:
                raise ValueError(
                    f"weight paths are not deployment sites in {assignment.region}: "
                    + ",".join(sorted(invalid))
                )
            for path in assignment.paths:
                if path in claimed:
                    raise ValueError("QAT weight policy sites must be unique")
                claimed[path] = assignment.spec
        else:
            if assignment.region in defaults:
                raise ValueError("QAT weight policy region defaults must be unique")
            defaults[assignment.region] = assignment.spec
    resolved: list[tuple[WeightSite, WeightFormatSpec]] = []
    for site in catalog.deployment_sites:
        spec = claimed.get(site.path, defaults.get(site.region))
        if spec is not None:
            resolved.append((site, spec))
    return tuple(resolved)


@dataclass(frozen=True)
class AppliedFoldedQATWeightPolicy:
    """Persistent reviewed QAT wrappers and their export/materialization seam."""

    model: nn.Module
    assignments: tuple[WeightRegionAssignment, ...]
    paths: tuple[str, ...]
    site_records: tuple[dict[str, object], ...]

    @property
    def quantized_modules(self) -> int:
        return len(self.paths)

    def _modules(
        self, model: nn.Module | None = None
    ) -> tuple[QATConv2d | QATLinear, ...]:
        source = self.model if model is None else model
        modules = tuple(source.get_submodule(path) for path in self.paths)
        if not all(isinstance(module, (QATConv2d, QATLinear)) for module in modules):
            raise RuntimeError("QAT weight paths became stale")
        return modules  # type: ignore[return-value]

    def set_blend_ratio(self, ratio: float) -> None:
        for module in self._modules():
            module.weight_quantizer.set_blend_ratio(ratio)

    def rebind(self, model: nn.Module) -> AppliedFoldedQATWeightPolicy:
        """Bind controls to an equivalent deepcopy such as a ModelEMA graph."""

        rebound = AppliedFoldedQATWeightPolicy(
            model=model,
            assignments=self.assignments,
            paths=self.paths,
            site_records=self.site_records,
        )
        mismatched = tuple(
            path
            for path, module, record in zip(
                rebound.paths,
                rebound._modules(),
                rebound.site_records,
                strict=True,
            )
            if module.weight_quantizer.spec.format_id != record["format_id"]
            or module.weight_quantizer.spec.bits != record["encoded_bits"]
        )
        if mismatched:
            raise RuntimeError(
                "rebound QAT weight policy differs at: " + ", ".join(mismatched)
            )
        return rebound

    @property
    def blend_ratio(self) -> float:
        ratios = {module.weight_quantizer.blend_ratio for module in self._modules()}
        if len(ratios) != 1:
            raise RuntimeError("QAT weight modules have inconsistent blend ratios")
        return ratios.pop()

    def materialize(
        self,
        *,
        clone_model: bool = True,
        expected_blend_ratio: float = 1.0,
    ) -> nn.Module:
        if not math.isclose(
            self.blend_ratio,
            expected_blend_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "QAT materialization blend ratio differs: "
                f"{self.blend_ratio} != {expected_blend_ratio}"
            )
        target = copy.deepcopy(self.model) if clone_model else self.model
        for path, module in zip(self.paths, self._modules(target), strict=True):
            with torch.no_grad():
                quantized_weight = module.weight_quantizer(module.weight).detach()
            if isinstance(module, QATConv2d):
                plain: nn.Conv2d | nn.Linear = nn.Conv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None,
                    module.padding_mode,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            else:
                plain = nn.Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            plain.weight = nn.Parameter(
                quantized_weight.clone(),
                requires_grad=module.weight.requires_grad,
            )
            if module.bias is not None:
                plain.bias = nn.Parameter(
                    module.bias.detach().clone(),
                    requires_grad=module.bias.requires_grad,
                )
            plain.train(module.training)
            _set_submodule(target, path, plain)
        return target


class FoldedQATWeightAdapter:
    """Apply persistent QAT modules only to a reviewed BN-folded graph."""

    def apply(
        self,
        model: nn.Module,
        *,
        catalog: Full35WeightRegionCatalog,
        assignments: tuple[WeightRegionAssignment, ...],
    ) -> AppliedFoldedQATWeightPolicy:
        retained_bn = sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in model.modules()
        )
        if retained_bn:
            raise ValueError(
                f"QAT requires a BN-folded graph; found {retained_bn} BatchNorm modules"
            )
        resolved = _resolve_sites(catalog, assignments)
        if not resolved:
            raise ValueError("QAT policy resolves to no deployment weight sites")
        paths: list[str] = []
        records: list[dict[str, object]] = []
        seen_module_ids: set[int] = set()
        for site, spec in resolved:
            module = model.get_submodule(site.path)
            if id(module) in seen_module_ids:
                raise ValueError("QAT refuses shared Conv/Linear module aliases")
            seen_module_ids.add(id(module))
            if tuple(module.weight.shape) != site.shape:
                raise ValueError(f"weight path changed shape: {site.path}")
            if isinstance(module, nn.Conv2d) and not isinstance(module, QATConv2d):
                replacement: nn.Module = QATConv2d.from_module(module, spec)
            elif isinstance(module, nn.Linear) and not isinstance(module, QATLinear):
                replacement = QATLinear.from_module(module, spec)
            else:
                raise TypeError(f"QAT weight path changed module type: {site.path}")
            _set_submodule(model, site.path, replacement)
            paths.append(site.path)
            records.append(
                {
                    "path": site.path,
                    "region": site.region,
                    "format_id": spec.format_id,
                    "encoded_bits": spec.bits,
                    "elements": site.elements,
                }
            )
        return AppliedFoldedQATWeightPolicy(
            model=model,
            assignments=assignments,
            paths=tuple(paths),
            site_records=tuple(records),
        )


__all__ = (
    "AppliedFoldedQATWeightPolicy",
    "FoldedQATWeightAdapter",
    "ProgressiveQuantizationSchedule",
    "QATConv2d",
    "QATLinear",
    "TrainableWeightFakeQuantizer",
)
