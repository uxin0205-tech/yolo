from __future__ import annotations

import pytest
import torch

from yolo_quantize import (
    LSQPlusActivationQuantizer,
    LSQPlusSpec,
    grad_scale,
    round_to_nearest_even_ste,
)


def test_round_to_nearest_even_ste_has_exact_forward_and_identity_gradient() -> None:
    values = torch.tensor(
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        dtype=torch.float64,
        requires_grad=True,
    )

    rounded = round_to_nearest_even_ste(values)

    assert torch.equal(
        rounded.detach(),
        torch.tensor([-2.0, -2.0, -0.0, 0.0, 2.0, 2.0], dtype=torch.float64),
    )
    rounded.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_grad_scale_preserves_forward_and_scales_backward() -> None:
    value = torch.tensor(3.0, requires_grad=True)

    result = grad_scale(value, 0.125)

    assert result.item() == 3.0
    result.backward()
    assert value.grad.item() == 0.125


@pytest.mark.parametrize("bits", [3, 4, 5, 6, 7, 8])
def test_lsq_plus_accepts_non_power_of_two_bit_widths(bits: int) -> None:
    spec = LSQPlusSpec(bits=bits)

    assert spec.qmin == 0
    assert spec.qmax == (1 << bits) - 1
    assert spec.rounding_mode == "nearest_even"


def test_lsq_plus_fake_quant_uses_learnable_unsigned_offset_range() -> None:
    quantizer = LSQPlusActivationQuantizer(
        LSQPlusSpec(bits=3),
        initial_scale=1.0,
        initial_offset=-1.0,
    ).double()
    values = torch.tensor([-2.0, -1.0, 0.0, 6.0, 7.0], dtype=torch.float64)

    output = quantizer(values)
    codes = quantizer.encode(values)

    assert torch.equal(codes, torch.tensor([0, 0, 1, 7, 7]))
    assert torch.allclose(
        output,
        torch.tensor([-1.0, -1.0, 0.0, 6.0, 6.0], dtype=torch.float64),
    )
    assert quantizer.scale.item() > 0.0
    assert quantizer.offset.requires_grad


def test_disabled_lsq_plus_is_an_exact_identity() -> None:
    quantizer = LSQPlusActivationQuantizer(
        LSQPlusSpec(bits=5),
        initial_scale=0.125,
        initial_offset=-0.25,
        enabled=False,
    )
    values = torch.randn(17)

    assert quantizer(values) is values


def test_lsq_plus_rejects_invalid_bit_width() -> None:
    with pytest.raises(ValueError, match="bits must be between 2 and 16"):
        LSQPlusSpec(bits=1)


def test_lsq_plus_can_initialize_unsigned_endpoints_from_observed_range() -> None:
    quantizer = LSQPlusActivationQuantizer(
        LSQPlusSpec(bits=3),
        initial_scale=1.0,
        initial_offset=0.0,
    )

    quantizer.initialize_from_range(-1.0, 3.0)

    assert torch.equal(
        quantizer.encode(torch.tensor([-1.0, 3.0])),
        torch.tensor([0, 7]),
    )
    assert quantizer.offset.item() == pytest.approx(-1.0)
    assert quantizer.scale.item() == pytest.approx(4.0 / 7.0)
