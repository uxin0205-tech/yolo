"""與 YOLO26 相容、保留官方 value path 的 Attention。"""

from __future__ import annotations

import copy

import torch
from torch import nn
from ultralytics.nn.modules.block import Attention

from .binary_basis import BinaryScore
from .config import BasisKind, NormalizationKind, ScaleMode, VariantConfig
from .normalization import build_normalizer
from .projection import ModularQKVProjection
from .quantization import FakeQuantConvBN, fake_quant_probability, fake_quant_symmetric
from .relative_bias import RelativePositionBias
from .schedule import ProgressiveBlend


class HardwareFriendlyAttention(nn.Module):
    """統一支援 P0、I、H、T5 與 V2 的可設定 Attention 介面。"""

    def __init__(
        self,
        source: Attention,
        config: VariantConfig,
        *,
        normalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.key_dim = source.key_dim
        self.dim = self.num_heads * self.head_dim
        q_channels = self.num_heads * self.key_dim
        self.qkv = ModularQKVProjection.from_fused(
            source.qkv,
            q_channels=q_channels,
            k_channels=q_channels,
            v_channels=self.dim,
            num_heads=self.num_heads,
        )
        self.pe = copy.deepcopy(source.pe)
        self.proj = copy.deepcopy(source.proj)
        if config.projection_weight_bits is not None:
            quantize = lambda module: FakeQuantConvBN(
                module,
                weight_bits=config.projection_weight_bits,
                activation_bits=config.projection_activation_bits,
            )
            self.qkv.q = quantize(self.qkv.q)
            self.qkv.k = quantize(self.qkv.k)
            self.qkv.v = quantize(self.qkv.v)
            self.proj = quantize(self.proj)
        self.score = BinaryScore(
            num_heads=self.num_heads,
            basis=config.basis,
            scale_mode=config.scale_mode,
            use_ste=config.use_ste,
        )
        self.bias = RelativePositionBias(
            num_heads=self.num_heads,
            kind=config.bias,
            max_size=config.max_bias_size,
        )
        self.normalize = normalizer if normalizer is not None else build_normalizer(config)
        self.progressive = ProgressiveBlend(transition_epochs=10) if config.progressive else None
        self.register_buffer("current_epoch", torch.zeros((), dtype=torch.long), persistent=True)
        self.last_scores: torch.Tensor | None = None
        self.last_probabilities: torch.Tensor | None = None
        self.last_values: torch.Tensor | None = None

    @classmethod
    def from_ultralytics(
        cls,
        source: Attention,
        config: VariantConfig,
        *,
        normalizer: nn.Module | None = None,
    ) -> HardwareFriendlyAttention:
        module = cls(source, config, normalizer=normalizer)
        module.train(source.training)
        return module

    @classmethod
    def from_hardware(
        cls,
        source: HardwareFriendlyAttention,
        config: VariantConfig,
        *,
        normalizer: nn.Module | None = None,
    ) -> HardwareFriendlyAttention:
        """重新設定既有 checkpoint，同時保留已訓練的 value path。"""

        module = copy.deepcopy(source)
        module.config = config
        score = BinaryScore(
            num_heads=source.num_heads,
            basis=config.basis,
            scale_mode=config.scale_mode,
            use_ste=config.use_ste,
        ).to(source.score.gamma.device)
        if score.gamma.shape == source.score.gamma.shape:
            score.gamma.data.copy_(source.score.gamma.detach())
        if (
            config.scale_mode is not ScaleMode.DYNAMIC
            and source.score.scale_mode is not ScaleMode.DYNAMIC
            and not source.score.needs_calibration
        ):
            score.set_fixed_coefficients(source.score.fixed_coefficients)
        module.score = score
        bias = RelativePositionBias(
            num_heads=source.num_heads,
            kind=config.bias,
            max_size=config.max_bias_size,
        ).to(source.score.gamma.device)
        if config.bias is source.config.bias:
            bias.load_state_dict(source.bias.state_dict())
        module.bias = bias
        module.normalize = normalizer if normalizer is not None else build_normalizer(config)
        module.normalize.to(source.score.gamma.device)
        module.progressive = ProgressiveBlend(transition_epochs=10) if config.progressive else None
        module.last_scores = None
        module.last_values = None
        module.last_probabilities = None
        module.train(source.training)
        return module

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.current_epoch.fill_(epoch)
        set_normalization_epoch = getattr(self.normalize, "set_epoch", None)
        if set_normalization_epoch is not None:
            set_normalization_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {channels}")
        tokens = height * width
        q_map, k_map, v_map = self.qkv(x)
        q = q_map.view(batch, self.num_heads, self.key_dim, tokens)
        k = k_map.view(batch, self.num_heads, self.key_dim, tokens)
        v = v_map.view(batch, self.num_heads, self.head_dim, tokens)
        scores = self.score(q, k)
        if self.progressive is not None and self.config.basis is not BasisKind.FP:
            fp_scores = (q * (self.key_dim**-0.5)).transpose(-2, -1) @ k
            scores = self.progressive(
                fp_scores,
                scores,
                epoch=int(self.current_epoch.item()),
            )
        scores = self.bias(scores, height=height, width=width)
        if self.config.v_bits:
            v = fake_quant_symmetric(v, bits=self.config.v_bits, dim=-1)
        self.last_values = v.detach()
        self.last_scores = scores.detach()
        aggregate = getattr(self.normalize, "aggregate", None)
        if self.config.bdcn_fused_value and aggregate is not None:
            aggregated = aggregate(scores, v).reshape(batch, channels, height, width)
            self.last_probabilities = None
        else:
            probability = self.normalize(scores)
            if self.config.p_bits and self.config.normalization is NormalizationKind.EXACT:
                probability = fake_quant_probability(probability, bits=self.config.p_bits)
            self.last_probabilities = probability.detach()
            aggregated = (v @ probability.transpose(-2, -1)).reshape(batch, channels, height, width)
        return self.proj(aggregated + self.pe(v.reshape(batch, channels, height, width)))
