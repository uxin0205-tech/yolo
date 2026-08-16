"""Streaming diagnostics for Exact, float-PWL and project bit-true PWL."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Self

import torch
from torch import nn
from torch.nn import functional as F

from .attention import HardwareFriendlyAttention
from .config import BasisKind, BiasKind, NormalizationKind, ScaleMode
from .normalization import BitTruePiecewiseLinearSoftmax, PiecewiseLinearSoftmax

_THRESHOLDS = (4.0, 6.0, 8.0, 10.0)
_PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)


@dataclass
class _MetricState:
    histogram_min: float
    histogram_step: float
    histogram_bins: int
    count: int = 0
    row_count: int = 0
    value_count: int = 0
    minimum: float = math.inf
    total: float = 0.0
    total_sq: float = 0.0
    threshold_counts: dict[float, int] = field(
        default_factory=lambda: {threshold: 0 for threshold in _THRESHOLDS}
    )
    exact_tail_mass: float = 0.0
    float_exp_abs_sum: float = 0.0
    float_exp_abs_max: float = 0.0
    bit_exp_abs_sum: float = 0.0
    bit_exp_abs_max: float = 0.0
    float_probability_abs_sum: float = 0.0
    float_probability_abs_max: float = 0.0
    bit_probability_abs_sum: float = 0.0
    bit_probability_abs_max: float = 0.0
    float_pv_abs_sum: float = 0.0
    float_pv_abs_max: float = 0.0
    bit_pv_abs_sum: float = 0.0
    bit_pv_abs_max: float = 0.0
    float_pv_cosine_sum: float = 0.0
    bit_pv_cosine_sum: float = 0.0
    cosine_count: int = 0
    histogram: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.histogram = torch.zeros(self.histogram_bins, dtype=torch.int64)

    @staticmethod
    def _error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
        error = (candidate.float() - reference.float()).abs()
        if not torch.isfinite(error).all():
            raise FloatingPointError("PWL diagnostics observed a non-finite approximation error")
        return float(error.sum(dtype=torch.float64).item()), float(error.max().item())

    def update(
        self,
        *,
        centered: torch.Tensor,
        exact_weights: torch.Tensor,
        float_weights: torch.Tensor,
        bit_weights: torch.Tensor,
        exact_probability: torch.Tensor,
        float_probability: torch.Tensor,
        bit_probability: torch.Tensor,
        exact_pv: torch.Tensor,
        float_pv: torch.Tensor,
        bit_pv: torch.Tensor,
    ) -> None:
        values = centered.detach().float()
        self.count += values.numel()
        self.row_count += values.numel() // values.shape[-1]
        self.minimum = min(self.minimum, float(values.min().item()))
        self.total += float(values.sum(dtype=torch.float64).item())
        self.total_sq += float(values.double().square().sum().item())
        for threshold in _THRESHOLDS:
            self.threshold_counts[threshold] += int((values < -threshold).sum().item())
        tail = values < -8.0
        self.exact_tail_mass += float(exact_probability.masked_fill(~tail, 0.0).sum().item())

        clipped = values.clamp(self.histogram_min, 0.0)
        histogram = torch.histc(
            clipped,
            bins=self.histogram_bins,
            min=self.histogram_min,
            max=0.0,
        ).to(torch.int64)
        self.histogram += histogram.cpu()

        exp_sum, exp_max = self._error(exact_weights, float_weights)
        self.float_exp_abs_sum += exp_sum
        self.float_exp_abs_max = max(self.float_exp_abs_max, exp_max)
        exp_sum, exp_max = self._error(exact_weights, bit_weights)
        self.bit_exp_abs_sum += exp_sum
        self.bit_exp_abs_max = max(self.bit_exp_abs_max, exp_max)

        probability_sum, probability_max = self._error(exact_probability, float_probability)
        self.float_probability_abs_sum += probability_sum
        self.float_probability_abs_max = max(self.float_probability_abs_max, probability_max)
        probability_sum, probability_max = self._error(exact_probability, bit_probability)
        self.bit_probability_abs_sum += probability_sum
        self.bit_probability_abs_max = max(self.bit_probability_abs_max, probability_max)

        pv_sum, pv_max = self._error(exact_pv, float_pv)
        self.float_pv_abs_sum += pv_sum
        self.float_pv_abs_max = max(self.float_pv_abs_max, pv_max)
        pv_sum, pv_max = self._error(exact_pv, bit_pv)
        self.bit_pv_abs_sum += pv_sum
        self.bit_pv_abs_max = max(self.bit_pv_abs_max, pv_max)
        self.value_count += exact_pv.numel()

        exact_vectors = exact_pv.float().transpose(-2, -1)
        float_vectors = float_pv.float().transpose(-2, -1)
        bit_vectors = bit_pv.float().transpose(-2, -1)
        device_type = exact_vectors.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            float_cosine = F.cosine_similarity(
                exact_vectors, float_vectors, dim=-1, eps=1e-12
            )
            bit_cosine = F.cosine_similarity(exact_vectors, bit_vectors, dim=-1, eps=1e-12)
        if not torch.isfinite(float_cosine).all() or not torch.isfinite(bit_cosine).all():
            raise FloatingPointError("PWL diagnostics observed a non-finite PV cosine")
        self.float_pv_cosine_sum += float(float_cosine.sum(dtype=torch.float64).item())
        self.bit_pv_cosine_sum += float(bit_cosine.sum(dtype=torch.float64).item())
        self.cosine_count += exact_vectors.numel() // exact_vectors.shape[-1]

    def _percentile(self, percentile: float) -> float:
        if not self.count:
            raise RuntimeError("PWL diagnostics observed no scores")
        target = max(1, math.ceil(percentile / 100.0 * self.count))
        index = int(torch.searchsorted(self.histogram.cumsum(0), torch.tensor(target)).item())
        index = min(index, self.histogram_bins - 1)
        return self.histogram_min + (index + 0.5) * self.histogram_step

    def summary(self) -> dict[str, float | int | dict[str, float]]:
        if not self.count or not self.row_count or not self.value_count or not self.cosine_count:
            raise RuntimeError("PWL diagnostics observed no scores")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        result: dict[str, float | int | dict[str, float]] = {
            "count": self.count,
            "row_count": self.row_count,
            "value_count": self.value_count,
            "min": self.minimum,
            "mean": mean,
            "std": math.sqrt(variance),
            "percentiles": {str(value): self._percentile(value) for value in _PERCENTILES},
            "exact_tail_probability_mass_mean": self.exact_tail_mass / self.row_count,
            "float_exp_mae": self.float_exp_abs_sum / self.count,
            "float_exp_max_error": self.float_exp_abs_max,
            "bit_true_exp_mae": self.bit_exp_abs_sum / self.count,
            "bit_true_exp_max_error": self.bit_exp_abs_max,
            "float_probability_mae": self.float_probability_abs_sum / self.count,
            "float_probability_max_error": self.float_probability_abs_max,
            "bit_true_probability_mae": self.bit_probability_abs_sum / self.count,
            "bit_true_probability_max_error": self.bit_probability_abs_max,
            "float_pv_mae": self.float_pv_abs_sum / self.value_count,
            "float_pv_max_error": self.float_pv_abs_max,
            "float_pv_cosine_similarity": self.float_pv_cosine_sum / self.cosine_count,
            "bit_true_pv_mae": self.bit_pv_abs_sum / self.value_count,
            "bit_true_pv_max_error": self.bit_pv_abs_max,
            "bit_true_pv_cosine_similarity": self.bit_pv_cosine_sum / self.cosine_count,
        }
        for threshold in _THRESHOLDS:
            result[f"ratio_lt_neg{int(threshold)}"] = self.threshold_counts[threshold] / self.count
        return result


class PWLValidationAccumulator:
    """Compare two PWL paths against Exact Softmax without storing score matrices."""

    def __init__(
        self,
        *,
        site: str,
        heads: int,
        score_floor: float = -8.0,
        segments: int = 16,
        histogram_min: float = -64.0,
        histogram_step: float = 0.25,
    ) -> None:
        if heads < 1:
            raise ValueError("heads must be positive")
        if histogram_min >= 0 or histogram_step <= 0:
            raise ValueError("histogram range must be negative-to-zero with positive step")
        histogram_bins = round(-histogram_min / histogram_step)
        self.site = site
        self.heads = heads
        self.score_floor = score_floor
        self.segments = segments
        self.float_pwl = PiecewiseLinearSoftmax(score_floor=score_floor, segments=segments)
        self.bit_true_pwl = BitTruePiecewiseLinearSoftmax(
            score_floor=score_floor,
            segments=segments,
        )
        make_state = lambda: _MetricState(histogram_min, histogram_step, histogram_bins)
        self._aggregate = make_state()
        self._heads = [make_state() for _ in range(heads)]

    @staticmethod
    def _pv(values: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        device_type = values.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            result = values.float() @ probability.float().transpose(-2, -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("PWL diagnostics observed a non-finite FP32 PV reference")
        return result

    def update(self, scores: torch.Tensor, values: torch.Tensor) -> None:
        if scores.ndim != 4 or scores.shape[1] != self.heads:
            raise ValueError(f"scores must have shape [batch, heads={self.heads}, query, key]")
        if values.ndim != 4 or values.shape[:2] != scores.shape[:2]:
            raise ValueError("values must match scores batch and heads")
        if values.shape[-1] != scores.shape[-1]:
            raise ValueError("values and scores must have the same key tokens")
        centered = scores.detach().float() - scores.detach().float().amax(dim=-1, keepdim=True)
        exact_weights = centered.exp()
        exact_probability = exact_weights / exact_weights.sum(dim=-1, keepdim=True)
        self.float_pwl.to(centered.device)
        self.bit_true_pwl.to(centered.device)
        float_weights = self.float_pwl.approximate_weights(centered)
        bit_weights = self.bit_true_pwl.approximate_weights(centered)
        float_probability = float_weights / float_weights.sum(dim=-1, keepdim=True)
        bit_probability = bit_weights / bit_weights.sum(dim=-1, keepdim=True)
        exact_pv = self._pv(values, exact_probability)
        float_pv = self._pv(values, float_probability)
        bit_pv = self._pv(values, bit_probability)
        common = {
            "centered": centered,
            "exact_weights": exact_weights,
            "float_weights": float_weights,
            "bit_weights": bit_weights,
            "exact_probability": exact_probability,
            "float_probability": float_probability,
            "bit_probability": bit_probability,
            "exact_pv": exact_pv,
            "float_pv": float_pv,
            "bit_pv": bit_pv,
        }
        self._aggregate.update(**common)
        for head, state in enumerate(self._heads):
            state.update(**{name: tensor[:, head : head + 1] for name, tensor in common.items()})

    def summary(self) -> dict[str, object]:
        return {
            "site": self.site,
            "score_floor": self.score_floor,
            "segments": self.segments,
            "segment_width": -self.score_floor / self.segments,
            "aggregate": self._aggregate.summary(),
            "heads": [
                {"head": head, **state.summary()} for head, state in enumerate(self._heads)
            ],
        }

    def histogram_rows(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        states = [("all", self._aggregate), *[(str(i), state) for i, state in enumerate(self._heads)]]
        for head, state in states:
            for index, count in enumerate(state.histogram.tolist()):
                rows.append(
                    {
                        "site": self.site,
                        "head": head,
                        "bin_left": state.histogram_min + index * state.histogram_step,
                        "bin_right": state.histogram_min + (index + 1) * state.histogram_step,
                        "count": count,
                    }
                )
        return rows


class PWLModelDiagnosticsCollector:
    """Fail-closed hooks for the two formal YOLO26m Attention sites."""

    def __init__(
        self,
        model: nn.Module,
        *,
        expected_paths: tuple[str, ...] | None = None,
        score_floor: float = -8.0,
        segments: int = 16,
    ) -> None:
        if expected_paths is None:
            from .integration import YOLO26M_ATTENTION_PATHS

            expected_paths = YOLO26M_ATTENTION_PATHS
        modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, HardwareFriendlyAttention)
        }
        if set(modules) != set(expected_paths):
            raise ValueError(
                f"expected PWL diagnostic paths {list(expected_paths)}, found {sorted(modules)}"
            )
        required = (
            BasisKind.HADAMARD,
            ScaleMode.POWER_OF_TWO,
            BiasKind.DECOMPOSED_2D,
            NormalizationKind.EXACT,
        )
        for path in expected_paths:
            config = modules[path].config
            actual = (config.basis, config.scale_mode, config.bias, config.normalization)
            if actual != required:
                raise ValueError(f"PWL score analysis requires fixed H/PoT/decomposed/exact at {path}")
        self._modules = [(path, modules[path]) for path in expected_paths]
        self._accumulators = {
            path: PWLValidationAccumulator(
                site=path,
                heads=module.num_heads,
                score_floor=score_floor,
                segments=segments,
            )
            for path, module in self._modules
        }
        self._handles: list[object] = []

    def _hook(self, path: str):
        def collect(module: HardwareFriendlyAttention, _inputs: object, _output: object) -> None:
            if module.last_scores is None or module.last_values is None:
                raise RuntimeError(f"Attention diagnostics are unavailable at {path}")
            self._accumulators[path].update(module.last_scores, module.last_values)

        return collect

    def __enter__(self) -> Self:
        self._handles = [module.register_forward_hook(self._hook(path)) for path, module in self._modules]
        return self

    def __exit__(self, *_args: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summaries(self) -> list[dict[str, object]]:
        return [self._accumulators[path].summary() for path, _ in self._modules]

    def histogram_rows(self) -> list[dict[str, float | int | str]]:
        return [
            row
            for path, _ in self._modules
            for row in self._accumulators[path].histogram_rows()
        ]
