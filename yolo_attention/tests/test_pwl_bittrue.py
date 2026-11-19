from __future__ import annotations

import math

import pytest
import torch

from yolo_attention.config import NormalizationKind, VariantConfig
from yolo_attention.normalization import BitTruePiecewiseLinearSoftmax, build_normalizer


def test_bit_true_pwl_uses_q8_8_integer_index_and_interpolation() -> None:
    module = BitTruePiecewiseLinearSoftmax(score_floor=-8.0, segments=16).eval()
    scores = torch.tensor([[[[0.0, -0.25, -0.5, -8.0, -9.0]]]])

    probabilities = module(scores)

    endpoints = module.endpoint_table.tolist()
    expected_midpoint = endpoints[15] + ((64 * (endpoints[16] - endpoints[15])) >> 7)
    assert module.last_centered_q is not None
    assert module.last_weights_int is not None
    assert module.last_centered_q.tolist() == [[[[0, -64, -128, -2048, -2048]]]]
    assert module.last_weights_int.tolist() == [
        [[[endpoints[16], expected_midpoint, endpoints[15], endpoints[0], endpoints[0]]]]
    ]
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]))


def test_bit_true_pwl_endpoint_table_is_unsigned_q1_15_and_16_bit_safe() -> None:
    module = BitTruePiecewiseLinearSoftmax(score_floor=-8.0, segments=16)

    assert module.endpoint_table.numel() == 17
    assert module.endpoint_table.dtype == torch.int64
    assert module.endpoint_table[0].item() == round(math.exp(-8.0) * 2**15)
    assert module.endpoint_table[-1].item() == 2**15
    assert int(module.endpoint_table.max()) <= 2**16 - 1
    assert module.endpoint_storage_bits == 17 * 16


def test_bit_true_pwl_factory_is_independent_from_float_pwl() -> None:
    config = VariantConfig(
        name="PWL-BITTRUE",
        normalization=NormalizationKind.BIT_TRUE_PWL,
        score_step=0.125,
        score_min=-64,
        pwl_segments=16,
    )

    module = build_normalizer(config)

    assert isinstance(module, BitTruePiecewiseLinearSoftmax)


def test_bit_true_pwl_requires_half_point_uniform_segments() -> None:
    with pytest.raises(ValueError, match="0.5"):
        BitTruePiecewiseLinearSoftmax(score_floor=-8.0, segments=15)


def test_bit_true_pwl_config_fails_before_factory_when_indexing_is_not_half_point() -> None:
    with pytest.raises(ValueError, match="0.5"):
        VariantConfig(
            name="bad-bittrue",
            normalization=NormalizationKind.BIT_TRUE_PWL,
            score_min=-64,
            score_step=0.125,
            pwl_segments=15,
        )
