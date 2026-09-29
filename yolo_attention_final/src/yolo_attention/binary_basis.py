"""以單一小型介面提供 I／H／T5 binary score 實作。"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import BasisKind, ScaleMode


class _ClippedSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        return gradient * (value.abs() <= 1).to(gradient.dtype)


def clipped_ste_sign(value: torch.Tensor) -> torch.Tensor:
    return _ClippedSign.apply(value)


def deterministic_sign(value: torch.Tensor) -> torch.Tensor:
    """把零映射為 +1，讓 float 與 Bit-True reference 共用同一規則。"""

    return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))


def xnor_popcount_dot(q_sign: torch.Tensor, k_sign: torch.Tensor) -> torch.Tensor:
    """處理 `[B, H, D, N]` tensors 的 XNOR＋popcount reference。"""

    if q_sign.shape[:-1] != k_sign.shape[:-1]:
        raise ValueError(f"Q/K shape mismatch: {tuple(q_sign.shape)} vs {tuple(k_sign.shape)}")
    q_bits = q_sign >= 0
    k_bits = k_sign >= 0
    matches = (q_bits.transpose(-2, -1).unsqueeze(-2) == k_bits.transpose(-2, -1).unsqueeze(-3)).sum(dim=-1)
    return (2 * matches - q_sign.shape[-2]).to(q_sign.dtype)


def fast_hadamard_transform(
    value: torch.Tensor,
    *,
    dim: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    """沿 power-of-two axis 執行可微分 Walsh-Hadamard transform。"""

    dim = dim if dim >= 0 else value.ndim + dim
    size = value.shape[dim]
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {size}")
    result = value.movedim(dim, -1)
    prefix = result.shape[:-1]
    width = 1
    while width < size:
        blocks = result.reshape(*prefix, -1, 2, width)
        left, right = blocks[..., 0, :], blocks[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape(*prefix, size)
        width *= 2
    if normalize:
        result = result / math.sqrt(size)
    return result.movedim(-1, dim)


class BinaryScore(nn.Module):
    """為 FP、I、H 或 T5 bases 產生全域 `[B,H,N,N]` scores。"""

    def __init__(
        self,
        *,
        num_heads: int,
        basis: BasisKind,
        scale_mode: ScaleMode = ScaleMode.DYNAMIC,
        use_ste: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.basis = BasisKind(basis)
        self.scale_mode = ScaleMode(scale_mode)
        self.use_ste = use_ste
        self.num_bases = 2 if self.basis in {BasisKind.HADAMARD, BasisKind.T5} else 1
        gamma = torch.full((self.num_bases,), 1.0 / self.num_bases)
        self.gamma = nn.Parameter(gamma)
        self.register_buffer(
            "fixed_coefficients",
            torch.ones(num_heads, self.num_bases),
            persistent=True,
        )
        self.register_buffer("fixed_coefficients_ready", torch.tensor(False), persistent=True)
        self.register_buffer("calibration_sums", torch.zeros(num_heads, self.num_bases), persistent=False)
        self.register_buffer("calibration_counts", torch.zeros(self.num_bases), persistent=False)
        self.register_buffer("calibration_enabled", torch.tensor(False), persistent=False)

    def _sign(self, value: torch.Tensor) -> torch.Tensor:
        return clipped_ste_sign(value) if self.use_ste and self.training else deterministic_sign(value)

    def set_fixed_coefficients(self, coefficients: torch.Tensor) -> None:
        expected = (self.num_heads, self.num_bases)
        if tuple(coefficients.shape) != expected:
            raise ValueError(f"fixed coefficients must have shape {expected}")
        values = coefficients.detach().to(self.fixed_coefficients)
        if self.scale_mode is ScaleMode.POWER_OF_TWO:
            values = torch.sign(values) * torch.pow(
                2.0, torch.round(torch.log2(values.abs().clamp_min(1e-12)))
            )
        self.fixed_coefficients.copy_(values)
        self.fixed_coefficients_ready.fill_(True)

    @property
    def needs_calibration(self) -> bool:
        return self.scale_mode is not ScaleMode.DYNAMIC and not bool(self.fixed_coefficients_ready)

    def begin_calibration(self) -> None:
        if self.scale_mode is ScaleMode.DYNAMIC:
            raise ValueError("dynamic scale does not require calibration")
        self.calibration_sums.zero_()
        self.calibration_counts.zero_()
        self.calibration_enabled.fill_(True)

    def _observe(self, index: int, coefficient: torch.Tensor) -> None:
        values = coefficient.detach().float().movedim(1, 0).reshape(self.num_heads, -1)
        self.calibration_sums[:, index].add_(values.sum(dim=1))
        self.calibration_counts[index].add_(values.shape[1])

    def finish_calibration(self) -> torch.Tensor:
        if not bool(self.calibration_enabled):
            raise RuntimeError("fixed scale calibration has not been started")
        if bool((self.calibration_counts == 0).any()):
            raise RuntimeError("fixed scale calibration observed no coefficients")
        with torch.inference_mode():
            coefficients = self.calibration_sums / self.calibration_counts.view(1, -1)
            self.set_fixed_coefficients(coefficients)
            self.calibration_enabled.fill_(False)
            frozen = self.fixed_coefficients.detach().clone()
        return frozen

    def _fixed(self, index: int) -> torch.Tensor:
        if not bool(self.fixed_coefficients_ready):
            raise RuntimeError("fixed scale mode requires set_fixed_coefficients() before forward")
        return self.fixed_coefficients[:, index].view(1, self.num_heads, 1, 1)

    @staticmethod
    def _magnitude(value: torch.Tensor) -> torch.Tensor:
        return value.abs().mean(dim=(-2, -1), keepdim=True)

    def _dynamic_coefficient(
        self,
        index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_scale: float,
    ) -> torch.Tensor:
        return attention_scale * self.gamma[index] * self._magnitude(q) * self._magnitude(k)

    def _coefficient(
        self,
        index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_scale: float,
    ) -> torch.Tensor:
        coefficient = self._dynamic_coefficient(index, q, k, attention_scale)
        if self.scale_mode is ScaleMode.DYNAMIC:
            return coefficient
        if bool(self.calibration_enabled):
            self._observe(index, coefficient)
            return coefficient
        return self._fixed(index)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if q.shape != k.shape:
            raise ValueError(f"Q/K shape mismatch: {tuple(q.shape)} vs {tuple(k.shape)}")
        if q.ndim != 4 or q.shape[1] != self.num_heads:
            raise ValueError("Q/K must have shape [batch, heads, channels, tokens]")
        attention_scale = q.shape[-2] ** -0.5
        if self.basis is BasisKind.FP:
            return (q * attention_scale).transpose(-2, -1) @ k
        if self.basis is BasisKind.IDENTITY:
            z = xnor_popcount_dot(self._sign(q), self._sign(k))
            return self._coefficient(0, q, k, attention_scale) * z
        if self.basis is BasisKind.HADAMARD:
            # Dynamic magnitude estimation 在 train 與 eval 都需要 normalized values。
            # Fixed-scale inference 只使用 sign，可省略正的 normalization constant，
            # 以符合硬體規劃。
            normalize_hadamard = (
                self.training or self.scale_mode is ScaleMode.DYNAMIC or bool(self.calibration_enabled)
            )
            q_h = fast_hadamard_transform(q, dim=-2, normalize=normalize_hadamard)
            k_h = fast_hadamard_transform(k, dim=-2, normalize=normalize_hadamard)
            z_i = xnor_popcount_dot(self._sign(q), self._sign(k))
            z_h = xnor_popcount_dot(self._sign(q_h), self._sign(k_h))
            return (
                self._coefficient(0, q, k, attention_scale) * z_i
                + self._coefficient(1, q_h, k_h, attention_scale) * z_h
            )
        if self.basis is BasisKind.T5:
            return self._t5(q, k, attention_scale)
        raise AssertionError(f"unsupported basis: {self.basis}")

    def _t5(self, q: torch.Tensor, k: torch.Tensor, attention_scale: float) -> torch.Tensor:
        q_scale_1 = q.abs().mean(dim=-2, keepdim=True)
        k_scale_1 = k.abs().mean(dim=-2, keepdim=True)
        q_sign_1, k_sign_1 = self._sign(q), self._sign(k)
        q_residual = q - q_scale_1 * q_sign_1
        k_residual = k - k_scale_1 * k_sign_1
        q_scale_2 = q_residual.abs().mean(dim=-2, keepdim=True)
        k_scale_2 = k_residual.abs().mean(dim=-2, keepdim=True)
        z_1 = xnor_popcount_dot(q_sign_1, k_sign_1)
        z_2 = xnor_popcount_dot(self._sign(q_residual), self._sign(k_residual))
        if self.scale_mode is ScaleMode.DYNAMIC:
            coefficient_1 = attention_scale * self.gamma[0] * q_scale_1.transpose(-2, -1) * k_scale_1
            coefficient_2 = attention_scale * self.gamma[1] * q_scale_2.transpose(-2, -1) * k_scale_2
        elif bool(self.calibration_enabled):
            coefficient_1 = attention_scale * self.gamma[0] * q_scale_1.transpose(-2, -1) * k_scale_1
            coefficient_2 = attention_scale * self.gamma[1] * q_scale_2.transpose(-2, -1) * k_scale_2
            self._observe(0, coefficient_1)
            self._observe(1, coefficient_2)
        else:
            coefficient_1, coefficient_2 = self._fixed(0), self._fixed(1)
        return coefficient_1 * z_1 + coefficient_2 * z_2
