"""Differentiable quantizers used by the BinaryAttention variants.

The operators in this file are deliberately small and deterministic.  They
implement fake quantization for accuracy experiments; they do not claim to be
bitwise kernels or a hardware speed measurement.
"""
from __future__ import annotations

import torch


class _ClippedSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        return grad_output * (value.abs() <= 1).to(grad_output.dtype)


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def clipped_ste_sign(value: torch.Tensor) -> torch.Tensor:
    """Sign in the forward pass with a clipped straight-through derivative."""

    return _ClippedSign.apply(value)


def round_ste(value: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(value)


def fake_quant_p8(probabilities: torch.Tensor) -> torch.Tensor:
    """P8 fake quantization.

    P8 is intentionally *not* renormalized after rounding.  This preserves
    the exact experiment definition and lets tests catch accidental softmax
    re-normalization.
    """

    return round_ste(probabilities * 255.0) / 255.0


def fake_quant_symmetric(value: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample symmetric fake quantization with an STE round operation."""

    if bits < 2:
        raise ValueError("symmetric fake quantization requires at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    scale = value.detach().abs().amax().clamp_min(eps) / qmax
    return scale * round_ste((value / scale).clamp(-qmax, qmax))


def fake_quant_v(value: torch.Tensor, bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Per-channel symmetric fake quantization for V across tokens."""

    if bits < 2:
        raise ValueError("V quantization requires at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    # YOLO attention stores V as [B, head, channel, token].  The released
    # BinaryAttention implementation derives one scale per V channel by
    # reducing the token axis only.
    scale = value.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    return scale * round_ste((value / scale).clamp(-qmax, qmax))


def fake_quant_magnitude(value: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    """Unsigned fake quantization for the non-negative N4 magnitude channel."""

    if bits < 2:
        raise ValueError("magnitude quantization requires at least 2 bits")
    qmax = 2**bits - 1
    scale = value.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(eps) / qmax
    return scale * round_ste((value / scale).clamp(0, qmax))
