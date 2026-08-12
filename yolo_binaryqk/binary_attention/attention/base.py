"""YOLO11-compatible FP and BinaryAttention implementations.

The binary classes are concrete modules.  The model parser constructs them
inside C2PSA, so the training entry point never creates an FP C2PSA and then
swaps its attention after the fact.
"""
from __future__ import annotations

import copy
from typing import Literal

import torch
from torch import nn

from .quantizers import clipped_ste_sign, fake_quant_magnitude, fake_quant_p8, fake_quant_v

QKMode = Literal["fp", "sign", "scaled_sign", "dual"]


def _deepcopy_without_forward_graph(module: nn.Module, memo: dict):
    """Deep-copy an attention module while dropping transient graph state."""

    result = module.__class__.__new__(module.__class__)
    memo[id(module)] = result
    state = module.__dict__.copy()
    for name in ("last_scores", "last_probabilities", "kd_probabilities"):
        if name in state:
            state[name] = None
    result.__dict__ = copy.deepcopy(state, memo)
    return result


class FPAttention(nn.Module):
    """Dependency-light YOLO11-style attention used by numerical tests."""

    def __init__(self, dim: int, num_heads: int = 1, attn_ratio: float = 0.5) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.key_dim = max(1, int(self.head_dim * attn_ratio))
        self.scale = self.key_dim**-0.5
        self.qkv = nn.Conv2d(dim, dim + 2 * self.key_dim * num_heads, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.pe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)

    def _split_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x).view(batch, self.num_heads, self.key_dim * 2 + self.head_dim, tokens)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        return q, k, v, height, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, height, width = self._split_qkv(x)
        probability = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        batch, channels = x.shape[:2]
        out = (v @ probability.transpose(-2, -1)).reshape(batch, channels, height, width)
        return self.proj(out + self.pe(v.reshape(batch, channels, height, width)))


class BinaryAttention(FPAttention):
    """Dependency-light binary attention with persistent forward diagnostics."""

    qk_mode: QKMode = "sign"

    def __init__(
        self,
        *args,
        use_qat: bool = False,
        p_bits: int | None = None,
        v_bits: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_qat, self.p_bits, self.v_bits = use_qat, p_bits, v_bits
        self.register_buffer("binary_forward_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("binary_qk_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("softmax_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("pv_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.last_scores: torch.Tensor | None = None
        self.last_probabilities: torch.Tensor | None = None
        self.kd_probabilities: torch.Tensor | None = None

    def __deepcopy__(self, memo):
        return _deepcopy_without_forward_graph(self, memo)

    def _binary(self, value: torch.Tensor) -> torch.Tensor:
        if self.use_qat and self.training:
            return clipped_ste_sign(value)
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    def binary_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._binary(q), self._binary(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.binary_forward_count.add_(1)
            self.binary_qk_count.add_(1)
            self.softmax_count.add_(1)
            self.pv_count.add_(1)
        q, k, v, height, width = self._split_qkv(x)
        q, k = self.binary_qk(q, k)
        scores = (q * self.scale).transpose(-2, -1) @ k
        probability = scores.softmax(dim=-1)
        if self.p_bits == 8:
            probability = fake_quant_p8(probability)
        if self.v_bits:
            v = fake_quant_v(v, self.v_bits)
        self.last_scores, self.last_probabilities = scores.detach(), probability.detach()
        self.kd_probabilities = probability
        batch, channels = x.shape[:2]
        out = (v @ probability.transpose(-2, -1)).reshape(batch, channels, height, width)
        return self.proj(out + self.pe(v.reshape(batch, channels, height, width)))


class SignOnlyBinaryAttention(BinaryAttention):
    qk_mode: QKMode = "sign"


class ScaledBinaryAttention(BinaryAttention):
    qk_mode: QKMode = "scaled_sign"

    def binary_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The released BinaryAttention implementation uses one scale per
        # sample/head, averaged over both channel and token dimensions.
        q_scale = q.abs().mean(dim=(2, 3), keepdim=True)
        k_scale = k.abs().mean(dim=(2, 3), keepdim=True)
        return q_scale * self._binary(q), k_scale * self._binary(k)


try:  # Keep numerical tests importable even if Ultralytics is absent.
    from ultralytics.nn.modules.block import Attention as _ULAttention
    from ultralytics.nn.modules.block import Conv, PSABlock
except ImportError:  # pragma: no cover
    _ULAttention = Conv = PSABlock = None


class UltralyticsBinaryAttention(_ULAttention if _ULAttention else nn.Module):
    """Concrete Ultralytics attention with sign, QAT, bias and fake quantization."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_ratio: float = 0.5,
        *,
        qk_mode: QKMode = "sign",
        use_qat: bool = False,
        p_bits: int | None = None,
        v_bits: int | None = None,
        bias_type: str = "none",
        magnitude_bits: int | None = None,
        max_bias_size: int = 32,
    ) -> None:
        if _ULAttention is None:
            raise RuntimeError("Ultralytics is required for a YOLO BinaryAttention model")
        super().__init__(dim, num_heads, attn_ratio)
        self.qk_mode = qk_mode
        self.use_qat, self.p_bits, self.v_bits = use_qat, p_bits, v_bits
        self.bias_type, self.magnitude_bits = bias_type, magnitude_bits
        self.max_bias_size = max_bias_size
        self.register_buffer("binary_forward_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("binary_qk_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("softmax_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("pv_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.last_scores: torch.Tensor | None = None
        self.last_probabilities: torch.Tensor | None = None
        self.kd_probabilities: torch.Tensor | None = None
        if qk_mode == "dual":
            shape = (1, num_heads, self.key_dim, 1)
            self.threshold_q1 = nn.Parameter(torch.zeros(shape))
            self.threshold_q2 = nn.Parameter(torch.zeros(shape))
            self.threshold_k1 = nn.Parameter(torch.zeros(shape))
            self.threshold_k2 = nn.Parameter(torch.zeros(shape))

        if bias_type == "dense_2d":
            size = 2 * max_bias_size - 1
            self.relative_bias = nn.Parameter(torch.zeros(num_heads, size, size))
            nn.init.trunc_normal_(self.relative_bias, std=0.02)
        elif bias_type == "decomposed_2d":
            size = 2 * max_bias_size - 1
            self.relative_bias_h = nn.Parameter(torch.zeros(num_heads, size))
            self.relative_bias_w = nn.Parameter(torch.zeros(num_heads, size))

    def __deepcopy__(self, memo):
        """Copy the module without retaining forward-only autograd tensors.

        ``kd_probabilities`` intentionally keeps the student's autograd graph
        alive until the KD loss consumes it.  Ultralytics creates its EMA by
        deep-copying the model after the AMP probe, so PyTorch 2.11 rejects a
        normal copy when that transient graph is still attached.  The EMA
        needs parameters, buffers, and configuration only; diagnostics are
        reconstructed by the next forward and must not be copied into it.
        """

        return _deepcopy_without_forward_graph(self, memo)

    def _sign(self, value: torch.Tensor) -> torch.Tensor:
        if self.use_qat and self.training:
            return clipped_ste_sign(value)
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    def _qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qk_mode == "fp":
            return q, k
        if self.qk_mode == "scaled_sign":
            q_scale = q.abs().mean(dim=(2, 3), keepdim=True)
            k_scale = k.abs().mean(dim=(2, 3), keepdim=True)
            return q_scale * self._sign(q), k_scale * self._sign(k)
        return self._sign(q), self._sign(k)

    def _dual_basis(self, value: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two residual binary bases U1/U2 for one latent tensor."""

        threshold1 = getattr(self, f"threshold_{prefix}1")
        threshold2 = getattr(self, f"threshold_{prefix}2")
        alpha1 = value.abs().mean(dim=2, keepdim=True)
        b1 = self._sign(value - threshold1)
        residual = value - alpha1 * b1
        alpha2 = residual.abs().mean(dim=2, keepdim=True)
        b2 = self._sign(residual - threshold2)
        return alpha1 * b1, alpha2 * b2

    def _bias(self, height: int, width: int, device: torch.device) -> torch.Tensor | int:
        if self.bias_type == "none":
            return 0
        if max(height, width) > self.max_bias_size:
            raise ValueError(f"relative bias supports H/W <= {self.max_bias_size}, got {height}x{width}")
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        y, x = yy.flatten(), xx.flatten()
        offset = self.max_bias_size - 1
        dy = y[:, None] - y[None, :] + offset
        dx = x[:, None] - x[None, :] + offset
        if self.bias_type == "dense_2d":
            return self.relative_bias[:, dy, dx]
        return self.relative_bias_h[:, dy] + self.relative_bias_w[:, dx]

    def _dot(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return (q * self.scale).transpose(-2, -1) @ k

    def _finish(
        self,
        scores: torch.Tensor,
        value: torch.Tensor,
        height: int,
        width: int,
        *,
        probability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if probability is None:
            with torch.no_grad():
                self.softmax_count.add_(1)
            probability = scores.softmax(dim=-1)
            if self.p_bits == 8:
                probability = fake_quant_p8(probability)
        if self.v_bits:
            value = fake_quant_v(value, self.v_bits)
        with torch.no_grad():
            self.pv_count.add_(1)
        self.last_scores = scores.detach()
        self.last_probabilities = probability.detach()
        self.kd_probabilities = probability
        batch, channels = value.shape[0], value.shape[1] * value.shape[2]
        out = (value @ probability.transpose(-2, -1)).reshape(batch, channels, height, width)
        return self.proj(out + self.pe(value.reshape(batch, channels, height, width)))

    def _qkv_from_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x).view(batch, self.num_heads, self.key_dim * 2 + self.head_dim, tokens)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        return q, k, v, height, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.binary_forward_count.add_(1)
            self.binary_qk_count.add_(1)
        q, k, v, height, width = self._qkv_from_input(x)
        q, k = self._qk(q, k)
        scores = self._dot(q, k) + self._bias(height, width, x.device)
        return self._finish(scores, v, height, width)


class SignOnlyUltralyticsBinaryAttention(UltralyticsBinaryAttention):
    """Named concrete class for parser manifests and checkpoint diagnostics."""


class ScaledUltralyticsBinaryAttention(UltralyticsBinaryAttention):
    """Named concrete class for scaled-sign parser manifests."""


class _DualBasisAttention(UltralyticsBinaryAttention):
    """Shared implementation for N1/N2/N3/N4 residual basis variants."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["qk_mode"] = "dual"
        super().__init__(*args, **kwargs)
        self.basis_lambda = nn.Parameter(torch.ones(2, 2))

    def _basis_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        return self._dual_basis(q, "q"), self._dual_basis(k, "k")

    def _begin_dual_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        q, k, v, height, width = self._qkv_from_input(x)
        return q, k, v, height, width


class ParallelDualBinaryAttention(_DualBasisAttention):
    """N1: two branches, two softmax operations and two PV operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.binary_forward_count.add_(1)
            self.binary_qk_count.add_(2)
            self.softmax_count.add_(2)
            self.pv_count.add_(2)
        q, k, v, height, width = self._begin_dual_forward(x)
        (q1, q2), (k1, k2) = self._basis_qk(q, k)
        s1 = self._dot(q1, k1) + self._bias(height, width, x.device)
        s2 = self._dot(q2, k2) + self._bias(height, width, x.device)
        p1, p2 = s1.softmax(-1), s2.softmax(-1)
        if self.p_bits == 8:
            p1, p2 = fake_quant_p8(p1), fake_quant_p8(p2)
        if self.v_bits:
            v = fake_quant_v(v, self.v_bits)
        self.last_scores = ((s1 + s2) / 2).detach()
        self.last_probabilities = ((p1 + p2) / 2).detach()
        self.kd_probabilities = (p1 + p2) / 2
        batch, channels = x.shape[0], x.shape[1]
        out = ((v @ p1.transpose(-2, -1)) + (v @ p2.transpose(-2, -1)))
        out = (out / 2).reshape(batch, channels, height, width)
        return self.proj(out + self.pe(v.reshape(batch, channels, height, width)))


class ResidualDualFullBasisAttention(_DualBasisAttention):
    """N2: all four Uqa U*kb cross terms, one softmax and one PV."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.binary_forward_count.add_(1)
            self.binary_qk_count.add_(4)
        q, k, v, height, width = self._begin_dual_forward(x)
        (q1, q2), (k1, k2) = self._basis_qk(q, k)
        score = (
            self.basis_lambda[0, 0] * self._dot(q1, k1)
            + self.basis_lambda[0, 1] * self._dot(q1, k2)
            + self.basis_lambda[1, 0] * self._dot(q2, k1)
            + self.basis_lambda[1, 1] * self._dot(q2, k2)
            + self._bias(height, width, x.device)
        )
        return self._finish(score, v, height, width)


class ResidualDualMatchedBasisAttention(_DualBasisAttention):
    """N3: only D11 and D22 are evaluated; cross terms are not computed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.binary_forward_count.add_(1)
            self.binary_qk_count.add_(2)
        q, k, v, height, width = self._begin_dual_forward(x)
        (q1, q2), (k1, k2) = self._basis_qk(q, k)
        score = (
            self.basis_lambda[0, 0] * self._dot(q1, k1)
            + self.basis_lambda[1, 1] * self._dot(q2, k2)
            + self._bias(height, width, x.device)
        )
        return self._finish(score, v, height, width)


class MagnitudeSideChannelAttention(ResidualDualMatchedBasisAttention):
    """N4: N3 score plus a Q/K magnitude rank-1 score correction."""

    def __init__(self, *args, magnitude_bits: int | None = None, **kwargs) -> None:
        super().__init__(*args, magnitude_bits=magnitude_bits, **kwargs)
        self.magnitude_weight = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.binary_forward_count.add_(1)
            self.binary_qk_count.add_(2)
        q, k, v, height, width = self._begin_dual_forward(x)
        (q1, q2), (k1, k2) = self._basis_qk(q, k)
        score = (
            self.basis_lambda[0, 0] * self._dot(q1, k1)
            + self.basis_lambda[1, 1] * self._dot(q2, k2)
        )
        mq = q.abs().mean(dim=2)
        mk = k.abs().mean(dim=2)
        if self.magnitude_bits is not None:
            mq = fake_quant_magnitude(mq.unsqueeze(2), self.magnitude_bits).squeeze(2)
            mk = fake_quant_magnitude(mk.unsqueeze(2), self.magnitude_bits).squeeze(2)
        magnitude_score = mq.unsqueeze(-1) * mk.unsqueeze(-2)
        score = score + self.scale * self.magnitude_weight * magnitude_score + self._bias(height, width, x.device)
        return self._finish(score, v, height, width)


class BinaryPSABlock(nn.Module):
    """PSABlock that constructs the requested attention class directly."""

    def __init__(
        self,
        c: int,
        attn_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True,
        attention_cls: type[nn.Module] = UltralyticsBinaryAttention,
        attention_kwargs: dict | None = None,
    ) -> None:
        if Conv is None:
            raise RuntimeError("Ultralytics is required")
        super().__init__()
        self.attn = attention_cls(c, attn_ratio=attn_ratio, num_heads=max(num_heads, 1), **(attention_kwargs or {}))
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class BinaryC2PSA(nn.Module):
    """C2PSA whose ``m`` sequence is built with the concrete attention class."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        *,
        attention_cls: type[nn.Module] = UltralyticsBinaryAttention,
        attention_kwargs: dict | None = None,
    ) -> None:
        if Conv is None:
            raise RuntimeError("Ultralytics is required")
        super().__init__()
        if c1 != c2:
            raise ValueError("BinaryC2PSA requires c1 == c2")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(
                BinaryPSABlock(
                    self.c,
                    attn_ratio=0.5,
                    num_heads=max(self.c // 64, 1),
                    attention_cls=attention_cls,
                    attention_kwargs=attention_kwargs,
                )
                for _ in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), dim=1))


# The qk mode is an immutable constructor argument, so one concrete module
# class safely covers sign-only and scaled-sign T variants.  These aliases
# preserve the historical public class name and keep checkpoint manifests
# stable while the resolved variant still records the exact qk mode.
SignOnlyUltralyticsBinaryAttention = UltralyticsBinaryAttention
ScaledUltralyticsBinaryAttention = UltralyticsBinaryAttention
SignOnlyAttention = UltralyticsBinaryAttention
ScaledAttention = UltralyticsBinaryAttention
