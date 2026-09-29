from __future__ import annotations

import pytest
import torch

from yolo_attention.config import BiasKind, RowCorrection
from yolo_attention.normalization import ExactSoftmax, IntegerLUTSoftmax
from yolo_attention.relative_bias import RelativePositionBias


@pytest.mark.parametrize("kind", [BiasKind.DENSE_2D, BiasKind.DECOMPOSED_2D])
def test_zero_initialized_bias_is_noop_and_trainable(kind: BiasKind) -> None:
    bias = RelativePositionBias(num_heads=2, kind=kind, max_size=4)
    scores = torch.randn(1, 2, 6, 6, requires_grad=True)

    output = bias(scores, height=2, width=3)

    torch.testing.assert_close(output, scores)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in bias.parameters())


def test_relative_bias_rejects_mismatched_token_geometry() -> None:
    bias = RelativePositionBias(num_heads=1, kind=BiasKind.DENSE_2D, max_size=4)
    with pytest.raises(ValueError, match="token"):
        bias(torch.zeros(1, 1, 5, 5), height=2, width=3)


def test_exact_softmax_preserves_probability_normalization() -> None:
    probabilities = ExactSoftmax()(torch.randn(2, 3, 5, 5))
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 3, 5))
    assert torch.all(probabilities >= 0)


@pytest.mark.parametrize(
    ("correction", "exact"),
    [
        (RowCorrection.NONE, False),
        (RowCorrection.MAX_ELEMENT, True),
        (RowCorrection.LARGEST_REMAINDER, True),
    ],
)
def test_integer_lut_softmax_returns_u8_probabilities(correction: RowCorrection, exact: bool) -> None:
    module = IntegerLUTSoftmax(score_step=0.125, score_min=-64, exp_bits=15, correction=correction)
    probabilities = module(torch.randn(2, 2, 7, 7))

    assert probabilities.shape == (2, 2, 7, 7)
    assert module.last_u8 is not None and module.last_u8.dtype == torch.uint8
    assert torch.all(module.last_u8 >= 0)
    row_sums = module.last_u8.to(torch.int32).sum(-1)
    if exact:
        assert torch.equal(row_sums, torch.full_like(row_sums, 255))
    else:
        assert int((row_sums - 255).abs().max()) <= 2
    torch.testing.assert_close(probabilities, module.last_u8.to(probabilities.dtype) / 255.0)
