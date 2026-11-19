"""Q10／PoT 固定尺度與 signed16 相對 bias 的整數 reference。

沿用 optimize StaticDyadicScore／DyadicRelativeBias；PoT 訓練與整數合併為專案延伸。
訓練前向與部署整數路徑一致，反向使用 bounded-amplitude STE 浮點代理。
"""

from __future__ import annotations

import torch
from torch import nn

from .config import FixedScaleFormat, ScoreAffineConfig


class QuantizedScoreAffine(nn.Module):
    def __init__(self, config: ScoreAffineConfig, coefficients: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        values = torch.as_tensor(coefficients, dtype=torch.float32).detach().clone()
        if values.shape != (config.num_heads, config.num_bases) or not torch.isfinite(values).all():
            raise ValueError("invalid coefficients")
        self.coefficients = nn.Parameter(values)
        self.table_x = nn.Parameter(torch.zeros(config.num_heads, 2 * config.max_size - 1))
        self.table_y = nn.Parameter(torch.zeros_like(self.table_x))
        scale, bx, by = self._codes()
        self.register_buffer("cached_scale_codes", scale.clone(), persistent=False)
        self.register_buffer("cached_bias_x", bx.clone(), persistent=False)
        self.register_buffer("cached_bias_y", by.clone(), persistent=False)

    def _codes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.coefficients.detach().clamp(1 / 1024, 1)
        if self.config.format is FixedScaleFormat.Q10:
            scale = (c * 1024).round().to(torch.int32)
        else:
            scale = c.log2().round().to(torch.int32)
        tables = [(p.detach().clamp(-32, 32767 / 1024) * 1024).round().to(torch.int32)
                  for p in (self.table_x, self.table_y)]
        return scale, *tables

    def train(self, mode: bool = True) -> QuantizedScoreAffine:
        super().train(mode)
        if not mode:
            for cached, codes in zip((self.cached_scale_codes, self.cached_bias_x, self.cached_bias_y),
                                     self._codes()):
                cached.copy_(codes)
        return self

    def _indices(self, counts: list[torch.Tensor], height: int, width: int):
        cfg = self.config
        if not 1 <= height <= cfg.max_size or not 1 <= width <= cfg.max_size:
            raise ValueError("relative bias size exceeds explicit contract")
        n = height * width
        if (len(counts) != cfg.num_bases or any(x.ndim != 4 or x.shape[1:] != (cfg.num_heads, n, n)
                                              for x in counts)
                or any(x.shape != counts[0].shape for x in counts)):
            raise ValueError("binary score shape mismatch")
        y, x = torch.meshgrid(torch.arange(height, device=counts[0].device),
                              torch.arange(width, device=counts[0].device), indexing="ij")
        y, x = y.flatten(), x.flatten()
        return y[:, None] - y[None, :] + cfg.max_size - 1, x[:, None] - x[None, :] + cfg.max_size - 1

    def integer_score(self, counts: list[torch.Tensor], *, height: int, width: int) -> torch.Tensor:
        """counts 為合法 [-32,32] binary dots；int32 容器、B2 所需 signed18。"""
        dy, dx = self._indices(counts, height, width)
        if any(x.dtype != torch.int32 for x in counts):
            raise ValueError("integer score requires int32 counts")
        scale, bx, by = self._codes() if self.training else (
            self.cached_scale_codes, self.cached_bias_x, self.cached_bias_y)
        result = bx[:, dx].unsqueeze(0) + by[:, dy].unsqueeze(0)
        for index, count in enumerate(counts):
            code = scale[:, index].view(1, -1, 1, 1)
            term = count * code if self.config.format is FixedScaleFormat.Q10 else torch.bitwise_left_shift(
                count, code + 10)
            result = result + term
        return result

    def forward(self, counts: list[torch.Tensor], *, height: int, width: int) -> torch.Tensor:
        codes = self.integer_score([x.detach().to(torch.int32) for x in counts], height=height, width=width)
        exact = codes.to(torch.float32) * (1 / 1024)
        if not self.training or not torch.is_grad_enabled():
            return exact
        dy, dx = self._indices(counts, height, width)
        scale, bx, by = self._codes()
        quantized = scale.float() / 1024 if self.config.format is FixedScaleFormat.Q10 else torch.pow(
            2., scale.float())
        bounded = self.coefficients.clamp(1 / 1024, 1)
        c = quantized + (bounded - bounded.detach())

        def proxy_table(parameter, integer):
            bounded = parameter.clamp(-32, 32767 / 1024)
            return integer.float() / 1024 + (bounded - bounded.detach())

        proxy = proxy_table(self.table_x, bx)[:, dx].unsqueeze(0) + proxy_table(self.table_y, by)[:, dy].unsqueeze(0)
        for index, count in enumerate(counts):
            proxy = proxy + count * c[:, index].view(1, -1, 1, 1)
        return exact + (proxy - proxy.detach())

    def export_constants(self) -> dict:
        scale, bx, by = self._codes()
        return {"format": self.config.format.value, "scale_codes": scale.cpu().tolist(),
                "bias_x_codes": bx.cpu().tolist(), "bias_y_codes": by.cpu().tolist(),
                "fraction_bits": 10, "bias_signed_bits": 16,
                "accumulator_signed_bits": self.config.accumulator_bits,
                "rounding": "nearest-ties-to-even", "bias_orientation": "query-minus-key",
                "scale_code_meaning": "m/1024" if self.config.format is FixedScaleFormat.Q10 else "2^k",
                "max_height": self.config.max_size, "max_width": self.config.max_size}
