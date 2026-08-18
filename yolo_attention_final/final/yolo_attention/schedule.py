"""Training schedules that are independent from the Ultralytics trainer."""

from __future__ import annotations

import torch
from torch import nn


class ProgressiveBlend(nn.Module):
    def __init__(self, transition_epochs: int = 10) -> None:
        super().__init__()
        if transition_epochs < 1:
            raise ValueError("transition_epochs must be positive")
        self.transition_epochs = transition_epochs

    def lambda_at(self, epoch: int) -> float:
        return min(max(epoch, 0) / self.transition_epochs, 1.0)

    def forward(
        self,
        fp_scores: torch.Tensor,
        binary_scores: torch.Tensor,
        *,
        epoch: int,
    ) -> torch.Tensor:
        weight = self.lambda_at(epoch)
        return (1.0 - weight) * fp_scores + weight * binary_scores
