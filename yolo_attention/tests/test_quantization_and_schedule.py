from __future__ import annotations

import torch
from ultralytics.nn.modules.conv import Conv

from yolo_attention.quantization import (
    FakeQuantConvBN,
    fake_quant_probability,
    fake_quant_symmetric,
    integer_pv_accumulate,
)
from yolo_attention.schedule import ProgressiveBlend


def test_probability_quantizer_uses_unsigned_fixed_grid() -> None:
    value = torch.tensor([0.0, 0.1, 0.5, 1.0])

    quantized = fake_quant_probability(value, bits=8)

    assert torch.all((quantized >= 0) & (quantized <= 1))
    torch.testing.assert_close(quantized * 255, torch.round(quantized * 255))


def test_symmetric_quantizer_is_bounded_and_has_ste_gradient() -> None:
    value = torch.tensor([-3.0, -0.2, 0.3, 2.0], requires_grad=True)

    quantized = fake_quant_symmetric(value, bits=8)
    quantized.sum().backward()

    assert torch.isfinite(quantized).all()
    assert value.grad is not None and torch.equal(value.grad, torch.ones_like(value))


def test_progressive_blend_reaches_pure_binary_at_transition_end() -> None:
    schedule = ProgressiveBlend(transition_epochs=10)
    fp = torch.tensor([1.0])
    binary = torch.tensor([3.0])

    assert schedule.lambda_at(0) == 0.0
    assert schedule.lambda_at(5) == 0.5
    assert schedule.lambda_at(10) == 1.0
    torch.testing.assert_close(schedule(fp, binary, epoch=5), torch.tensor([2.0]))


def test_fake_quant_conv_bn_preserves_shape_and_gradient() -> None:
    module = FakeQuantConvBN(Conv(8, 12, 1, act=False), weight_bits=8, activation_bits=8)
    value = torch.randn(2, 8, 4, 4, requires_grad=True)

    output = module(value)
    output.mean().backward()

    assert output.shape == (2, 12, 4, 4)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert module.module.conv.weight.grad is not None


def test_integer_pv_reference_accumulates_in_s32() -> None:
    value = torch.tensor([[[[1, -2, 3], [4, 5, -6]]]], dtype=torch.int8)
    probability = torch.tensor([[[[255, 0, 0], [0, 255, 0], [0, 0, 255]]]], dtype=torch.uint8)

    accumulated = integer_pv_accumulate(probability, value)

    assert accumulated.dtype == torch.int32
    torch.testing.assert_close(accumulated, value.to(torch.int32) * 255)
