"""Deterministic fake-quantization primitives for Q0/Q2 simulation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.conv import Conv


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value.round()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        return gradient


def round_ste(value: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(value)


def fake_quant_probability(value: torch.Tensor, *, bits: int = 8) -> torch.Tensor:
    if bits < 2:
        raise ValueError("probability quantization requires at least 2 bits")
    qmax = 2**bits - 1
    return round_ste(value.clamp(0, 1) * qmax) / qmax


def fake_quant_symmetric(
    value: torch.Tensor,
    *,
    bits: int = 8,
    dim: int | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Symmetric signed fake quantization with a detached dynamic scale."""

    if bits < 2:
        raise ValueError("symmetric quantization requires at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        maximum = value.detach().abs().amax()
    else:
        maximum = value.detach().abs().amax(dim=dim, keepdim=True)
    scale = maximum.clamp_min(eps) / qmax
    return scale * round_ste((value / scale).clamp(-qmax, qmax))


class FakeQuantConvBN(nn.Module):
    """QAT-style W/A fake quant around one Ultralytics Conv+BN wrapper."""

    def __init__(self, module: Conv, *, weight_bits: int = 8, activation_bits: int = 8) -> None:
        super().__init__()
        self.module = module
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        convolution = self.module.conv
        activation = fake_quant_symmetric(value, bits=self.activation_bits)
        weight = fake_quant_symmetric(
            convolution.weight,
            bits=self.weight_bits,
            dim=(1, 2, 3),
        )
        output = F.conv2d(
            activation,
            weight,
            convolution.bias,
            convolution.stride,
            convolution.padding,
            convolution.dilation,
            convolution.groups,
        )
        return self.module.act(self.module.bn(output))


def integer_pv_accumulate(probability_u8: torch.Tensor, value_s8: torch.Tensor) -> torch.Tensor:
    """Raw S32 accumulator for V[S8] @ P[U8]^T, without dequantization."""

    if probability_u8.dtype is not torch.uint8:
        raise TypeError("probability must be torch.uint8")
    if value_s8.dtype is not torch.int8:
        raise TypeError("value must be torch.int8")
    if probability_u8.shape[:-2] != value_s8.shape[:-2]:
        raise ValueError("P/V batch and head dimensions must match")
    if probability_u8.shape[-1] != value_s8.shape[-1]:
        raise ValueError("P key tokens and V tokens must match")
    return value_s8.to(torch.int32) @ probability_u8.to(torch.int32).transpose(-2, -1)
