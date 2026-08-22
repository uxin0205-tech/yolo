from __future__ import annotations

import pytest
import torch

from achitechure_2.lite_c3k2 import KernelMode, LiteC3k2, LiteC3k2Config


def test_config_validation() -> None:
    assert LiteC3k2Config(kernel_mode="1x1_3x3").kernel_mode is KernelMode.K1_K3
    with pytest.raises(ValueError, match="e must"):
        LiteC3k2Config(e=0.0)
    with pytest.raises(ValueError, match="inner_n"):
        LiteC3k2Config(inner_n=0)
    with pytest.raises(ValueError, match="尚未核准"):
        LiteC3k2Config(use_rep=True)


@pytest.mark.parametrize(
    ("config", "inner_n", "kernel"),
    [
        (LiteC3k2Config(e=0.375), 2, (3, 3)),
        (LiteC3k2Config(inner_n=1), 1, (3, 3)),
        (LiteC3k2Config(kernel_mode="1x1_3x3"), 2, (1, 3)),
    ],
)
def test_shape_and_explicit_factor(config: LiteC3k2Config, inner_n: int, kernel: tuple[int, int]) -> None:
    layer = LiteC3k2(32, 48, n=2, config=config)
    output = layer(torch.randn(2, 32, 12, 12))
    first = layer.m[0].m[0]
    assert output.shape == (2, 48, 12, 12)
    assert len(layer.m) == 2
    assert len(layer.m[0].m) == inner_n
    assert first.cv1.conv.kernel_size[0] == kernel[0]
    assert first.cv2.conv.kernel_size[0] == kernel[1]
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in layer.parameters()
    )
