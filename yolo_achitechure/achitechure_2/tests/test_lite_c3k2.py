from __future__ import annotations

import pytest
import torch
from ultralytics.nn.modules.block import RepBottleneck

from achitechure_2.lite_c3k2 import KernelMode, LiteC3k2, LiteC3k2Config
from achitechure_2.rep import assert_rep_fuse


def test_config_validation() -> None:
    assert LiteC3k2Config(kernel_mode="1x1_3x3").kernel_mode is KernelMode.K1_K3
    with pytest.raises(ValueError, match="e must"):
        LiteC3k2Config(e=0.0)
    with pytest.raises(ValueError, match="inner_n"):
        LiteC3k2Config(inner_n=0)
    with pytest.raises(ValueError, match="Rep recovery"):
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


def test_r1_uses_only_rep_bottleneck() -> None:
    layer = LiteC3k2(32, 32, config=LiteC3k2Config(inner_n=1, use_rep=True))
    assert isinstance(layer.m[0].m[0], RepBottleneck)


def test_r1_rep_fuse_is_numerically_equivalent() -> None:
    layer = LiteC3k2(32, 32, config=LiteC3k2Config(inner_n=1, use_rep=True)).eval()
    report = assert_rep_fuse(layer, torch.randn(1, 32, 8, 8))
    assert report.passed
    assert report.max_abs_diff <= 1e-4
