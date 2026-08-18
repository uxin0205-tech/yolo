from __future__ import annotations

import torch

from yolo_attention.binary_basis import clipped_ste_sign
from yolo_attention.config import BiasKind
from yolo_attention.normalization import BitTruePiecewiseLinearSoftmax, PiecewiseLinearSoftmax
from yolo_attention.relative_bias import RelativePositionBias


def test_pwl_fixed_hardware_contract_and_saturation() -> None:
    pwl = BitTruePiecewiseLinearSoftmax(score_floor=-10, segments=20)
    assert pwl.endpoint_table.numel() == 21
    assert pwl.endpoint_storage_bits == 336
    scores = torch.tensor([[[[-100.0, -10.0, -0.5, 0.0]]]])
    probability = pwl(scores)
    assert torch.isfinite(probability).all()
    assert torch.allclose(probability.sum(-1), torch.ones_like(probability.sum(-1)))
    assert pwl.last_centered_q.min().item() == -2560
    assert pwl.last_centered_q.max().item() == 0
    assert not probability.requires_grad


def test_float_pwl_surrogate_has_finite_nonzero_qk_and_bias_gradients() -> None:
    q = torch.randn(1, 4, 8, 4, requires_grad=True)
    k = torch.randn(1, 4, 8, 4, requires_grad=True)
    scores = clipped_ste_sign(q).transpose(-2, -1) @ clipped_ste_sign(k)
    bias = RelativePositionBias(num_heads=4, kind=BiasKind.DECOMPOSED_2D, max_size=32)
    probabilities = PiecewiseLinearSoftmax(score_floor=-10, segments=20)(bias(scores, height=2, width=2))
    loss = (probabilities * torch.randn_like(probabilities)).sum()
    loss.backward()
    for gradient in (q.grad, k.grad, bias.table_x.grad, bias.table_y.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)


def test_q8_8_rounding_is_half_up() -> None:
    pwl = BitTruePiecewiseLinearSoftmax(score_floor=-10, segments=20)
    pwl.approximate_weights(torch.tensor([-0.5 / 256, -1.5 / 256]))
    assert pwl.last_centered_q.tolist() == [0, -1]
