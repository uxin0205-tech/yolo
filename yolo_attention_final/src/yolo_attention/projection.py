"""No-Attention-inspired modular Q/K/V projection and BN-fold references."""

from __future__ import annotations

import copy

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv


def qkv_channel_indices(
    *,
    num_heads: int,
    key_dim: int,
    head_dim: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return channel gathers for official per-head [Q,K,V] fused layout."""

    q_indices: list[int] = []
    k_indices: list[int] = []
    v_indices: list[int] = []
    stride = 2 * key_dim + head_dim
    for head in range(num_heads):
        base = head * stride
        q_indices.extend(range(base, base + key_dim))
        k_indices.extend(range(base + key_dim, base + 2 * key_dim))
        v_indices.extend(range(base + 2 * key_dim, base + stride))
    return tuple(
        torch.tensor(indices, dtype=torch.long, device=device)
        for indices in (q_indices, k_indices, v_indices)
    )


def interleave_qkv_maps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    """Reconstruct the official fused per-head QKV channel layout."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V maps must be BCHW tensors")
    batch, _, height, width = q.shape
    if (
        k.shape[0] != batch
        or v.shape[0] != batch
        or k.shape[-2:] != (height, width)
        or v.shape[-2:]
        != (
            height,
            width,
        )
    ):
        raise ValueError("Q/K/V map shapes are incompatible")
    if any(tensor.shape[1] % num_heads for tensor in (q, k, v)):
        raise ValueError("Q/K/V channels must be divisible by num_heads")
    q = q.view(batch, num_heads, q.shape[1] // num_heads, height, width)
    k = k.view(batch, num_heads, k.shape[1] // num_heads, height, width)
    v = v.view(batch, num_heads, v.shape[1] // num_heads, height, width)
    return torch.cat((q, k, v), dim=2).reshape(batch, -1, height, width)


def _new_conv_like(source: Conv, out_channels: int) -> Conv:
    convolution = source.conv
    dilation = convolution.dilation[0]
    if any(value != dilation for value in convolution.dilation):
        raise ValueError("Ultralytics Conv adapter requires equal spatial dilation")
    module = Conv(
        convolution.in_channels,
        out_channels,
        k=convolution.kernel_size,
        s=convolution.stride,
        p=convolution.padding,
        g=convolution.groups,
        d=dilation,
        act=copy.deepcopy(source.act),
    )
    module.bn.eps = source.bn.eps
    module.bn.momentum = source.bn.momentum
    return module.to(device=convolution.weight.device, dtype=convolution.weight.dtype)


def _slice_conv_bn(source: Conv, indices: torch.Tensor) -> Conv:
    target = _new_conv_like(source, len(indices))
    indices = indices.to(source.conv.weight.device)
    with torch.no_grad():
        target.conv.weight.copy_(source.conv.weight.index_select(0, indices))
        for name in ("weight", "bias", "running_mean", "running_var"):
            source_value = getattr(source.bn, name)
            target_value = getattr(target.bn, name)
            target_value.copy_(source_value.index_select(0, indices))
        target.bn.num_batches_tracked.copy_(source.bn.num_batches_tracked)
    target.train(source.training)
    return target


class ModularQKVProjection(nn.Module):
    """Three Conv+BN paths initialized exactly from official fused QKV."""

    def __init__(self, q: Conv, k: Conv, v: Conv, *, num_heads: int) -> None:
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.num_heads = num_heads

    @classmethod
    def from_fused(
        cls,
        source: Conv,
        *,
        q_channels: int,
        k_channels: int,
        v_channels: int,
        num_heads: int,
    ) -> ModularQKVProjection:
        if q_channels != k_channels:
            raise ValueError("Q and K channel counts must match")
        if q_channels % num_heads or v_channels % num_heads:
            raise ValueError("Q/K/V channels must be divisible by num_heads")
        if 2 * q_channels + v_channels != source.conv.out_channels:
            raise ValueError("Q/K/V channels do not cover fused projection output")
        key_dim, head_dim = q_channels // num_heads, v_channels // num_heads
        q_idx, k_idx, v_idx = qkv_channel_indices(
            num_heads=num_heads,
            key_dim=key_dim,
            head_dim=head_dim,
            device=source.conv.weight.device,
        )
        return cls(
            _slice_conv_bn(source, q_idx),
            _slice_conv_bn(source, k_idx),
            _slice_conv_bn(source, v_idx),
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(x), self.k(x), self.v(x)


def fold_conv_bn(source: Conv) -> nn.Conv2d:
    """Fold an eval Conv+BN wrapper into a biased Conv2d."""

    if not isinstance(source.act, nn.Identity):
        raise TypeError("fold_conv_bn currently requires Identity activation")
    convolution, bn = source.conv, source.bn
    folded = nn.Conv2d(
        convolution.in_channels,
        convolution.out_channels,
        convolution.kernel_size,
        convolution.stride,
        convolution.padding,
        convolution.dilation,
        convolution.groups,
        bias=True,
        padding_mode=convolution.padding_mode,
        device=convolution.weight.device,
        dtype=convolution.weight.dtype,
    )
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    with torch.no_grad():
        folded.weight.copy_(convolution.weight * scale[:, None, None, None])
        folded.bias.copy_(bn.bias - scale * bn.running_mean)
    folded.train(source.training)
    return folded


def _slice_folded(source: nn.Conv2d, indices: torch.Tensor) -> nn.Conv2d:
    target = nn.Conv2d(
        source.in_channels,
        len(indices),
        source.kernel_size,
        source.stride,
        source.padding,
        source.dilation,
        source.groups,
        bias=True,
        padding_mode=source.padding_mode,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    indices = indices.to(source.weight.device)
    with torch.no_grad():
        target.weight.copy_(source.weight.index_select(0, indices))
        if source.bias is None:
            target.bias.zero_()
        else:
            target.bias.copy_(source.bias.index_select(0, indices))
    target.train(source.training)
    return target


def split_folded_qkv(
    source: nn.Conv2d,
    *,
    num_heads: int,
    key_dim: int,
    head_dim: int,
) -> tuple[nn.Conv2d, nn.Conv2d, nn.Conv2d]:
    indices = qkv_channel_indices(
        num_heads=num_heads,
        key_dim=key_dim,
        head_dim=head_dim,
        device=source.weight.device,
    )
    return tuple(_slice_folded(source, index) for index in indices)
