"""統一 scores-to-P 介面背後的 probability normalization 策略。"""

from __future__ import annotations

import math

import torch
from torch import nn

from .bdcn import BDCNCodebookBank, BDCNNormalizer
from .config import NormalizationKind, RowCorrection, VariantConfig


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    return weights / denominator


class ExactSoftmax(nn.Module):
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return scores.softmax(dim=-1)


class ScoreIndexedLUTSoftmax(nn.Module):
    """量化 exp lookup 的 score offsets，但讓 P 保持 floating 且精確 normalization。"""

    def __init__(self, *, score_step: float = 0.125, score_min: int = -64) -> None:
        super().__init__()
        if score_step <= 0 or score_min >= 0:
            raise ValueError("score_step must be positive and score_min negative")
        self.score_step = float(score_step)
        self.score_min = int(score_min)
        offsets = torch.arange(0, -score_min + 1, dtype=torch.float32)
        self.register_buffer("exp_lut", torch.exp(-offsets * score_step), persistent=True)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scores_int = torch.round(scores / self.score_step).to(torch.int64)
        centered = (scores_int - scores_int.amax(dim=-1, keepdim=True)).clamp(self.score_min, 0)
        weights = self.exp_lut[(-centered).to(torch.long)].to(scores.dtype)
        return _normalize_weights(weights)


class IntegerLUTSoftmax(nn.Module):
    """量化 scores、套用 exponential LUT，並輸出 U8 probabilities。"""

    def __init__(
        self,
        *,
        score_step: float = 0.125,
        score_min: int = -64,
        exp_bits: int = 15,
        correction: RowCorrection = RowCorrection.NONE,
    ) -> None:
        super().__init__()
        if score_step <= 0 or score_min >= 0:
            raise ValueError("score_step must be positive and score_min negative")
        self.score_step = score_step
        self.score_min = score_min
        self.exp_bits = exp_bits
        self.correction = RowCorrection(correction)
        offsets = torch.arange(0, -score_min + 1, dtype=torch.float64)
        scale = 2**exp_bits - 1
        lut = torch.round(torch.exp(-offsets * score_step) * scale).clamp_min(1).to(torch.int64)
        self.register_buffer("exp_lut", lut, persistent=True)
        self.last_u8: torch.Tensor | None = None
        self.last_row_sums: torch.Tensor | None = None

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scores_int = torch.round(scores / self.score_step).to(torch.int64)
        centered = scores_int - scores_int.amax(dim=-1, keepdim=True)
        centered = centered.clamp(self.score_min, 0)
        exponent = self.exp_lut[(-centered).to(torch.long)]
        denominator = exponent.sum(dim=-1, keepdim=True)
        numerator = 255 * exponent
        rounded = torch.div(numerator + denominator // 2, denominator, rounding_mode="floor")
        if self.correction is RowCorrection.MAX_ELEMENT:
            rounded = self._max_element_correction(rounded, exponent)
        elif self.correction is RowCorrection.LARGEST_REMAINDER:
            rounded = self._largest_remainder(numerator, denominator)
        u8 = rounded.clamp(0, 255).to(torch.uint8)
        self.last_u8 = u8.detach()
        self.last_row_sums = u8.to(torch.int32).sum(dim=-1).detach()
        quantized = u8.to(scores.dtype) / 255.0
        if self.training:
            exact = scores.softmax(dim=-1)
            return exact + (quantized - exact).detach()
        return quantized

    @staticmethod
    def _max_element_correction(values: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        corrected = values.clone()
        delta = 255 - corrected.sum(dim=-1, keepdim=True)
        index = exponent.argmax(dim=-1, keepdim=True)
        selected = corrected.gather(-1, index)
        corrected.scatter_(-1, index, (selected + delta).clamp(0, 255))
        return corrected

    @staticmethod
    def _largest_remainder(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        base = torch.div(numerator, denominator, rounding_mode="floor")
        remainder = numerator.remainder(denominator)
        missing = 255 - base.sum(dim=-1)
        order = remainder.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(order.shape[-1], device=order.device).expand_as(order)
        ranks.scatter_(-1, order, rank_values)
        return base + (ranks < missing.unsqueeze(-1)).to(base.dtype)


class PiecewiseLinearSoftmax(nn.Module):
    """用等距 linear segments 近似 exp，再進行 normalization。"""

    def __init__(self, *, score_floor: float = -8.0, segments: int = 16) -> None:
        super().__init__()
        if score_floor >= 0 or segments < 2:
            raise ValueError("score_floor must be negative and segments at least 2")
        self.score_floor = float(score_floor)
        self.segments = int(segments)
        knots = torch.linspace(self.score_floor, 0.0, self.segments + 1)
        self.register_buffer("knots", knots, persistent=True)
        self.register_buffer("values", knots.exp(), persistent=True)

    def approximate_weights(self, centered: torch.Tensor) -> torch.Tensor:
        centered = centered.clamp(self.score_floor, 0.0)
        position = (centered - self.score_floor) * self.segments / -self.score_floor
        lower = position.floor().to(torch.long).clamp(0, self.segments - 1)
        fraction = position - lower.to(position.dtype)
        y0 = self.values[lower]
        y1 = self.values[lower + 1]
        return y0 + fraction * (y1 - y0)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        centered = scores - scores.amax(dim=-1, keepdim=True)
        return _normalize_weights(self.approximate_weights(centered))


class BitTruePiecewiseLinearSoftmax(nn.Module):
    """專案的 Bit-True Q8.8/UQ1.15 PWL-exp 路徑，搭配 exact float normalization。

    Endpoint lookup、indexing、interpolation 與 saturation 會重現預定的 integer
    datapath。最後 reciprocal 仍是 exact software reference，不宣稱為 integer
    或 division-free implementation。
    """

    score_fraction_bits = 8
    endpoint_fraction_bits = 15
    endpoint_bits = 16
    fraction_bits = 7

    def __init__(self, *, score_floor: float = -8.0, segments: int = 16) -> None:
        super().__init__()
        if score_floor >= 0 or segments < 2:
            raise ValueError("score_floor must be negative and segments at least 2")
        width = -float(score_floor) / int(segments)
        if not math.isclose(width, 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("bit-true PWL requires uniform segment width 0.5")
        self.score_floor = float(score_floor)
        self.segments = int(segments)
        knots = torch.linspace(self.score_floor, 0.0, self.segments + 1, dtype=torch.float64)
        scale = 1 << self.endpoint_fraction_bits
        endpoints = torch.floor(knots.exp() * scale + 0.5).clamp(0, 2**self.endpoint_bits - 1)
        self.register_buffer("endpoint_table", endpoints.to(torch.int64), persistent=True)
        self.last_centered_q: torch.Tensor | None = None
        self.last_weights_int: torch.Tensor | None = None

    @property
    def endpoint_storage_bits(self) -> int:
        return self.endpoint_table.numel() * self.endpoint_bits

    def approximate_weights(self, centered: torch.Tensor) -> torch.Tensor:
        score_scale = 1 << self.score_fraction_bits
        centered_q = torch.floor(centered * score_scale + 0.5).to(torch.int64)
        signed_min = -(1 << 15)
        signed_max = (1 << 15) - 1
        floor_q = round(self.score_floor * score_scale)
        centered_q = centered_q.clamp(signed_min, signed_max).clamp(floor_q, 0)
        z = centered_q - floor_q
        endpoint_mask = z == -floor_q
        segment_index = torch.bitwise_right_shift(z, self.fraction_bits).clamp(0, self.segments - 1)
        fraction = torch.bitwise_and(z, (1 << self.fraction_bits) - 1)
        y0 = self.endpoint_table[segment_index]
        y1 = self.endpoint_table[segment_index + 1]
        interpolated = y0 + torch.bitwise_right_shift(fraction * (y1 - y0), self.fraction_bits)
        weights_int = torch.where(endpoint_mask, self.endpoint_table[-1], interpolated)
        self.last_centered_q = centered_q.detach()
        self.last_weights_int = weights_int.detach()
        return weights_int.to(centered.dtype) / (1 << self.endpoint_fraction_bits)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        centered = scores - scores.amax(dim=-1, keepdim=True)
        return _normalize_weights(self.approximate_weights(centered))


class PowerOfTwoSoftmax(nn.Module):
    """把 exp weights 投影到 powers of two；這是專案 reference，不是 Bit-True Shiftmax。"""

    def __init__(self, *, score_floor: float = -8.0) -> None:
        super().__init__()
        if score_floor >= 0:
            raise ValueError("score_floor must be negative")
        self.score_floor = float(score_floor)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        centered = (scores - scores.amax(dim=-1, keepdim=True)).clamp(self.score_floor, 0.0)
        exponent = torch.round(centered / torch.log(torch.tensor(2.0, device=scores.device)))
        return _normalize_weights(torch.pow(2.0, exponent))


class NormalizedHardSigmoid(nn.Module):
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        centered = scores - scores.amax(dim=-1, keepdim=True)
        return _normalize_weights(((centered + 3.0) / 6.0).clamp(0.0, 1.0))


class ReluNormalize(nn.Module):
    def __init__(self, *, margin: float = 1.0) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.margin = float(margin)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        centered = scores - scores.amax(dim=-1, keepdim=True)
        return _normalize_weights(torch.relu(centered + self.margin))


class Multimax(nn.Module):
    """受 Multimax 啟發的 sparse top-k stick-breaking normalization。"""

    def __init__(self, *, top_k: int) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = int(top_k)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.top_k, scores.shape[-1])
        values, indices = scores.topk(k, dim=-1)
        if k == 1:
            selected = torch.ones_like(values)
        else:
            gates = torch.sigmoid(values[..., :-1] - values[..., 1:])
            remaining = torch.ones_like(values[..., :1])
            pieces: list[torch.Tensor] = []
            for gate in gates.unbind(dim=-1):
                gate = gate.unsqueeze(-1)
                pieces.append(remaining * gate)
                remaining = remaining * (1.0 - gate)
            pieces.append(remaining)
            selected = torch.cat(pieces, dim=-1)
        probability = torch.zeros_like(scores)
        return probability.scatter(-1, indices, selected)


class ProgressiveNormalizer(nn.Module):
    """Normalization-level PMP：從 exact P 漸進到 candidate P。"""

    def __init__(self, candidate: nn.Module, *, transition_epochs: int = 5) -> None:
        super().__init__()
        if transition_epochs < 1:
            raise ValueError("transition_epochs must be positive")
        self.exact = ExactSoftmax()
        self.candidate = candidate
        self.transition_epochs = int(transition_epochs)
        self.register_buffer("current_epoch", torch.zeros((), dtype=torch.long), persistent=True)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.current_epoch.fill_(epoch)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        rho = min(float(self.current_epoch.item()) / self.transition_epochs, 1.0)
        exact = self.exact(scores)
        candidate = self.candidate(scores)
        return exact * (1.0 - rho) + candidate * rho


def build_normalizer(
    config: VariantConfig,
    *,
    bdcn_bank: BDCNCodebookBank | None = None,
    bdcn_table_indices: torch.Tensor | None = None,
) -> nn.Module:
    """只依公開 variant configuration 建立 normalization。"""

    if config.normalization is NormalizationKind.BDCN:
        if bdcn_bank is None or bdcn_table_indices is None:
            raise ValueError("BDCN normalization requires a shared bank and table indices")
        return BDCNNormalizer(
            bank=bdcn_bank,
            table_indices=bdcn_table_indices,
            step=config.resolved_bdcn_step,
            denominator=config.bdcn_denominator,
            reciprocal_newton_steps=config.bdcn_reciprocal_newton_steps,
        )
    score_floor = config.score_min * config.score_step
    builders = {
        NormalizationKind.EXACT: lambda: ExactSoftmax(),
        NormalizationKind.LUT: lambda: ScoreIndexedLUTSoftmax(
            score_step=config.score_step,
            score_min=config.score_min,
        ),
        NormalizationKind.INTEGER_LUT: lambda: IntegerLUTSoftmax(
            score_step=config.score_step,
            score_min=config.score_min,
            exp_bits=config.exp_bits,
            correction=config.row_correction,
        ),
        NormalizationKind.PIECEWISE_LINEAR: lambda: PiecewiseLinearSoftmax(
            score_floor=score_floor,
            segments=config.pwl_segments,
        ),
        NormalizationKind.BIT_TRUE_PWL: lambda: BitTruePiecewiseLinearSoftmax(
            score_floor=score_floor,
            segments=config.pwl_segments,
        ),
        NormalizationKind.POWER_OF_TWO: lambda: PowerOfTwoSoftmax(score_floor=score_floor),
        NormalizationKind.HARD_SIGMOID: lambda: NormalizedHardSigmoid(),
        NormalizationKind.RELU: lambda: ReluNormalize(margin=config.relu_margin),
        NormalizationKind.MULTIMAX: lambda: Multimax(top_k=config.multimax_top_k),
    }
    candidate = builders[config.normalization]()
    if config.normalization_progressive and config.normalization is not NormalizationKind.EXACT:
        return ProgressiveNormalizer(
            candidate,
            transition_epochs=config.normalization_transition_epochs,
        )
    return candidate
