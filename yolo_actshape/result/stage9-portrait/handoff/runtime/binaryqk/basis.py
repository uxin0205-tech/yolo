"""精確二值前向、signed-dot 反向代理；不使用浮點 QK 補償。"""

import math

import torch
from torch import nn

BASES = {
    "B1": ("sign", "residual"),
    "B2": ("sign", "hadamard"),
    "B3": ("sign", "residual", "hadamard"),
}


class ClippedSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return torch.where(value >= 0, 1.0, -1.0).to(value.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        return gradient * (value.abs() <= 1)


def hadamard(value, *, normalize=False):
    """只沿 key channel 軸變換；token 數不必是二次冪。"""
    size = value.shape[-2]
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard key dimension must be a power of two")
    result = value.transpose(-2, -1)
    prefix = result.shape[:-1]
    width = 1
    while width < size:
        blocks = result.reshape(*prefix, -1, 2, width)
        left, right = blocks[..., 0, :], blocks[..., 1, :]
        result = torch.cat((left + right, left - right), -1).reshape(*prefix, size)
        width *= 2
    if normalize:
        result = result * (1 / math.sqrt(size))
    return result.transpose(-2, -1)


def xnor_dot(q, k, tile=32):
    """布林 XNOR/popcount reference；分塊限制暫存，尚非 packed kernel。"""
    if q.ndim != 4 or q.shape[:-1] != k.shape[:-1] or tile <= 0:
        raise ValueError("expected compatible [B,H,D,N] Q/K and positive tile")
    qb, kb = (q >= 0).transpose(-2, -1), (k >= 0).transpose(-2, -1)
    out = q.new_empty((*q.shape[:2], q.shape[-1], k.shape[-1]))
    for start in range(0, q.shape[-1], tile):
        count = (qb[..., start:start + tile, None, :] == kb[..., None, :, :]).sum(-1)
        out[..., start:start + tile, :] = (2 * count - q.shape[-2]).to(q.dtype)
    return out


class ExactDotSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        ctx.save_for_backward(q, k)
        return xnor_dot(q, k)

    @staticmethod
    def backward(ctx, gradient):
        q, k = ctx.saved_tensors
        return k @ gradient.transpose(-2, -1), q @ gradient


def branch_dots(q, k, bases, *, training):
    """共用二值分支生成；Stage 1 固定尺度與 Stage 2 定點尺度使用相同 dots。"""
    qs, ks = ClippedSign.apply(q), ClippedSign.apply(k)
    for basis in bases:
        if basis == "sign":
            a, b = qs, ks
        elif basis == "residual":
            a = ClippedSign.apply(q - q.abs().mean(-2, keepdim=True) * qs)
            b = ClippedSign.apply(k - k.abs().mean(-2, keepdim=True) * ks)
        elif basis == "hadamard":
            a = ClippedSign.apply(hadamard(q, normalize=training))
            b = ClippedSign.apply(hadamard(k, normalize=training))
        else:
            raise ValueError(f"unknown binary basis: {basis}")
        yield ExactDotSurrogate.apply(a, b)


class BranchScore(nn.Module):
    def __init__(self, variant, coefficients):
        super().__init__()
        if variant not in BASES:
            raise ValueError(f"unknown branch variant: {variant}")
        values = torch.as_tensor(coefficients, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != len(BASES[variant]):
            raise ValueError("coefficients must be [heads, bases]")
        if not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError("fixed coefficients must be finite and positive")
        self.variant, self.bases = variant, BASES[variant]
        self.register_buffer("coefficients", values.clone())

    def forward(self, q, k, *, height=None, width=None):
        if q.ndim != 4 or q.shape != k.shape or q.shape[1] != len(self.coefficients):
            raise ValueError("incompatible Q/K/head dimensions")
        result = None
        for index, score in enumerate(branch_dots(q, k, self.bases, training=self.training)):
            term = score * self.coefficients[:, index].view(1, -1, 1, 1)
            result = term if result is None else result + term
        return result


def calibration_moments(q, k):
    """共同 parent 上取每個 head 的幅度乘積，不建立 N×N 校準矩陣。"""
    sign = lambda x: torch.where(x >= 0, 1.0, -1.0)
    qr = q - q.abs().mean(-2, keepdim=True) * sign(q)
    kr = k - k.abs().mean(-2, keepdim=True) * sign(k)
    pairs = [(q, k), (qr, kr), (hadamard(q, normalize=True), hadamard(k, normalize=True))]
    # 對全 token pair 的 mean(absQ * absK) 等於兩個 mean 的乘積。
    return torch.stack([
        (a.abs().mean((-2, -1)) * b.abs().mean((-2, -1))).mean(0)
        * q.shape[-2] ** -0.5 for a, b in pairs
    ], -1)
