from __future__ import annotations

import pytest
import torch
from torch import nn

from achitechure_2.lite_c3k2 import LiteC3k2, LiteC3k2Config


class _ThreeScaleTask(nn.Module):
    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__()
        self.block = LiteC3k2(3, channels, config=LiteC3k2Config(inner_n=1))
        self.heads = nn.ModuleList(nn.Conv2d(channels, outputs, 1) for _ in range(3))

    def forward(self, value: torch.Tensor):
        feature = self.block(value)
        sizes = (80, 40, 20)
        return tuple(
            head(nn.functional.adaptive_avg_pool2d(feature, size)) for head, size in zip(self.heads, sizes)
        )


@pytest.mark.parametrize("outputs", (8, 12))
def test_640_detect_and_pose_contract_forward_loss_and_gradient(outputs: int) -> None:
    model = _ThreeScaleTask(8, outputs)
    predictions = model(torch.randn(1, 3, 640, 640))
    assert tuple(value.shape[-2:] for value in predictions) == ((80, 80), (40, 40), (20, 20))
    loss = sum(value.square().mean() for value in predictions)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
    )
