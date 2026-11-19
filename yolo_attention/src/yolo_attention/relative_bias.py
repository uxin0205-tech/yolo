"""Dense and decomposed two-dimensional relative-position bias."""

from __future__ import annotations

import torch
from torch import nn

from .config import BiasKind


class RelativePositionBias(nn.Module):
    def __init__(self, *, num_heads: int, kind: BiasKind, max_size: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kind = BiasKind(kind)
        self.max_size = max_size
        size = 2 * max_size - 1
        if self.kind is BiasKind.DENSE_2D:
            self.table = nn.Parameter(torch.zeros(num_heads, size, size))
        elif self.kind is BiasKind.DECOMPOSED_2D:
            self.table_y = nn.Parameter(torch.zeros(num_heads, size))
            self.table_x = nn.Parameter(torch.zeros(num_heads, size))
        elif self.kind is not BiasKind.NONE:
            raise ValueError(f"unsupported bias kind: {kind}")

    def forward(self, scores: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        tokens = height * width
        if scores.shape[-2:] != (tokens, tokens):
            raise ValueError(
                f"score token dimensions {tuple(scores.shape[-2:])} do not match {height}x{width}"
            )
        if self.kind is BiasKind.NONE:
            return scores
        if height > self.max_size or width > self.max_size:
            raise ValueError(f"relative bias supports at most {self.max_size}x{self.max_size}")
        y, x = torch.meshgrid(
            torch.arange(height, device=scores.device),
            torch.arange(width, device=scores.device),
            indexing="ij",
        )
        y, x = y.flatten(), x.flatten()
        offset = self.max_size - 1
        dy = y[:, None] - y[None, :] + offset
        dx = x[:, None] - x[None, :] + offset
        if self.kind is BiasKind.DENSE_2D:
            bias = self.table[:, dy, dx]
        else:
            bias = self.table_y[:, dy] + self.table_x[:, dx]
        return scores + bias.unsqueeze(0).to(scores.dtype)
