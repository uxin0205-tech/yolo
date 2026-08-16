"""Binary distance-codebook normalization and fused value reference."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import BDCNCodebookKind, BDCNDenominator, BDCNProjection


def project_one_pot(codebook: torch.Tensor, *, max_exponent: int = 15) -> torch.Tensor:
    candidates = torch.pow(
        torch.tensor(2.0, device=codebook.device, dtype=codebook.dtype),
        -torch.arange(max_exponent + 1, device=codebook.device, dtype=codebook.dtype),
    )
    index = (codebook.unsqueeze(-1) - candidates).abs().argmin(dim=-1)
    return candidates[index]


def project_two_pot(codebook: torch.Tensor, *, max_exponent: int = 15) -> torch.Tensor:
    powers = torch.pow(
        torch.tensor(2.0, device=codebook.device, dtype=codebook.dtype),
        -torch.arange(max_exponent + 1, device=codebook.device, dtype=codebook.dtype),
    )
    terms = torch.cat((torch.zeros_like(powers[:1]), powers))
    candidates = (terms[:, None] + terms[None, :]).flatten()
    index = (codebook.unsqueeze(-1) - candidates).abs().argmin(dim=-1)
    return candidates[index]


class BDCNCodebookBank(nn.Module):
    """Monotonic codebook bank shared according to the configured table assignment."""

    def __init__(
        self,
        num_tables: int,
        levels: int,
        step: float,
        kind: BDCNCodebookKind,
        projection: BDCNProjection,
        *,
        log_ratio_bound: float | None = None,
    ) -> None:
        super().__init__()
        if num_tables < 1 or levels < 2 or step <= 0:
            raise ValueError("num_tables/levels must be positive and levels at least 2")
        self.num_tables = int(num_tables)
        self.levels = int(levels)
        self.step = float(step)
        self.kind = BDCNCodebookKind(kind)
        self.projection = BDCNProjection(projection)
        self.log_ratio_bound = log_ratio_bound
        fixed = torch.exp(-torch.arange(levels, dtype=torch.float32) * step)
        if self.kind is BDCNCodebookKind.FIXED_EXP:
            if log_ratio_bound is not None:
                raise ValueError("fixed codebook does not accept a log-ratio bound")
            self.register_buffer("fixed", fixed.expand(num_tables, -1).clone(), persistent=True)
        elif self.kind is BDCNCodebookKind.LEARNED_ANCHORED:
            if log_ratio_bound is None or log_ratio_bound <= 0:
                raise ValueError("anchored codebook requires a positive log-ratio bound")
            self.raw_deltas = nn.Parameter(torch.zeros(num_tables, levels - 1))
        else:
            if log_ratio_bound is not None:
                raise ValueError("unbounded learned codebook does not accept a log-ratio bound")
            ratio = math.exp(-step)
            raw = math.log(ratio / (1.0 - ratio))
            self.raw_ratios = nn.Parameter(torch.full((num_tables, levels - 1), raw))

    def codebook(self) -> torch.Tensor:
        if self.kind is BDCNCodebookKind.FIXED_EXP:
            base = self.fixed
        elif self.kind is BDCNCodebookKind.LEARNED_ANCHORED:
            log_ratio = -self.step + self.log_ratio_bound * torch.tanh(self.raw_deltas)
            tail = torch.exp(log_ratio.cumsum(dim=-1))
            base = torch.cat((torch.ones_like(tail[:, :1]), tail), dim=-1)
        else:
            ratio = torch.sigmoid(self.raw_ratios).clamp_max(1.0 - 1e-6)
            tail = ratio.cumprod(dim=-1)
            base = torch.cat((torch.ones_like(tail[:, :1]), tail), dim=-1)
        if self.projection is BDCNProjection.FLOAT:
            return base
        projected = (
            project_one_pot(base) if self.projection is BDCNProjection.ONE_POT else project_two_pot(base)
        )
        if self.training and self.kind is not BDCNCodebookKind.FIXED_EXP:
            return base + (projected - base).detach()
        return projected


class BDCNNormalizer(nn.Module):
    """Map final scores to P, or aggregate V without materializing P."""

    def __init__(
        self,
        bank: BDCNCodebookBank,
        table_indices: torch.Tensor,
        step: float,
        denominator: BDCNDenominator,
        reciprocal_lut_size: int = 256,
        reciprocal_newton_steps: int = 0,
    ) -> None:
        super().__init__()
        if table_indices.ndim != 1 or step <= 0 or reciprocal_lut_size < 2:
            raise ValueError("table_indices must be 1D and steps/LUT size positive")
        self.bank = bank
        self.step = float(step)
        self.denominator = BDCNDenominator(denominator)
        if reciprocal_newton_steps not in {0, 1}:
            raise ValueError("reciprocal_newton_steps must be 0 or 1")
        if reciprocal_newton_steps and self.denominator is not BDCNDenominator.RECIPROCAL_LUT:
            raise ValueError("Newton refinement only applies to reciprocal LUT")
        self.reciprocal_newton_steps = int(reciprocal_newton_steps)
        self.register_buffer("table_indices", table_indices.to(torch.long), persistent=True)
        mantissa = torch.linspace(1.0, 2.0, reciprocal_lut_size)
        self.register_buffer("reciprocal_lut", mantissa.reciprocal(), persistent=True)
        self.last_row_sums: torch.Tensor | None = None
        self.last_bucket_saturation: torch.Tensor | None = None
        self.last_bucket_histogram: torch.Tensor | None = None
        self.last_bucket_overflow_rate: torch.Tensor | None = None
        self.last_distance_max: torch.Tensor | None = None
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        """Reset validation-wide counters used by the result contract."""

        self.diagnostic_bucket_histogram: torch.Tensor | None = None
        self.diagnostic_overflow_count = 0.0
        self.diagnostic_total_count = 0
        self.diagnostic_distance_max: float | None = None

    def _weights(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if scores.ndim != 4 or scores.shape[1] != self.table_indices.numel():
            raise ValueError("scores must be [B,H,N,N] and match table_indices")
        distance = scores.amax(dim=-1, keepdim=True) - scores
        raw_bucket = torch.round(distance / self.step).to(torch.long)
        bucket = raw_bucket.clamp(0, self.bank.levels - 1)
        tables = self.bank.codebook()[self.table_indices]
        source = tables.view(1, scores.shape[1], 1, -1).expand(scores.shape[0], -1, scores.shape[2], -1)
        weights = source.gather(-1, bucket)
        self.last_bucket_histogram = torch.bincount(
            bucket.reshape(-1), minlength=self.bank.levels
        ).detach()
        self.last_bucket_overflow_rate = (
            distance > self.step * (self.bank.levels - 1)
        ).float().mean().detach()
        self.last_distance_max = distance.max().detach()
        self.last_bucket_saturation = (
            bucket == self.bank.levels - 1
        ).float().mean().detach()
        # Diagnostics are checkpoint metadata, not part of the model dataflow.
        # Keep their accumulator on CPU so a checkpoint restored on CPU can be
        # moved to CUDA without combining stale CPU state with a CUDA batch.
        histogram = self.last_bucket_histogram.cpu()
        self.diagnostic_bucket_histogram = (
            histogram.clone()
            if self.diagnostic_bucket_histogram is None
            else self.diagnostic_bucket_histogram + histogram
        )
        total = int(histogram.sum().item())
        self.diagnostic_total_count += total
        self.diagnostic_overflow_count += float(self.last_bucket_overflow_rate.item()) * total
        maximum = float(self.last_distance_max.item())
        self.diagnostic_distance_max = (
            maximum
            if self.diagnostic_distance_max is None
            else max(self.diagnostic_distance_max, maximum)
        )
        return weights, bucket, tables

    def _reciprocal(self, denominator: torch.Tensor) -> torch.Tensor:
        if self.denominator is BDCNDenominator.EXACT:
            return denominator.reciprocal()
        if self.denominator is BDCNDenominator.POT_SHIFT:
            exponent = torch.round(torch.log2(denominator))
            return torch.pow(2.0, -exponent)
        mantissa, exponent = torch.frexp(denominator)
        normalized = mantissa * 2.0
        normalized_exponent = exponent - 1
        index = torch.round((normalized - 1.0) * (self.reciprocal_lut.numel() - 1)).to(torch.long)
        reciprocal = self.reciprocal_lut[index.clamp(0, self.reciprocal_lut.numel() - 1)]
        reciprocal = torch.ldexp(reciprocal.to(denominator.dtype), -normalized_exponent)
        if self.reciprocal_newton_steps:
            reciprocal = reciprocal * (2.0 - denominator * reciprocal)
        return reciprocal

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        weights, _, _ = self._weights(scores)
        denominator = weights.sum(dim=-1, keepdim=True)
        probability = weights * self._reciprocal(denominator)
        self.last_row_sums = probability.sum(dim=-1).detach()
        return probability

    def aggregate(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Compute grouped weighted V and apply one reciprocal per query row."""

        weights, bucket, tables = self._weights(scores)
        if value.ndim != 4 or value.shape[:2] != scores.shape[:2] or value.shape[-1] != scores.shape[-1]:
            raise ValueError("value must be [B,H,D,N] and align with scores")
        # Bucket sums can exceed the FP16 range before normalization even when
        # the final weighted average is finite. Accumulate the fused PV path in
        # FP32 under AMP, then return to the value-path dtype at the boundary.
        accumulation_dtype = (
            torch.float32 if value.dtype in {torch.float16, torch.bfloat16} else value.dtype
        )
        accumulated_value = value.to(accumulation_dtype)
        accumulated_weights = weights.to(accumulation_dtype)
        accumulated_tables = tables.to(accumulation_dtype)
        with torch.autocast(device_type=value.device.type, enabled=False):
            output = torch.zeros(
                value.shape[0],
                value.shape[1],
                value.shape[2],
                scores.shape[2],
                dtype=accumulation_dtype,
                device=value.device,
            )
            for level in range(self.bank.levels):
                mask = (bucket == level).to(accumulation_dtype)
                grouped = torch.einsum("bhqk,bhdk->bhdq", mask, accumulated_value)
                output = output + grouped * accumulated_tables[:, level].view(1, -1, 1, 1)
            denominator = accumulated_weights.sum(dim=-1, keepdim=True)
            reciprocal_rows = self._reciprocal(denominator)
            self.last_row_sums = (denominator * reciprocal_rows).squeeze(-1).detach()
            reciprocal = reciprocal_rows.transpose(-2, -1)
            return (output * reciprocal).to(value.dtype)
