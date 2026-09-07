from __future__ import annotations

import pytest
import torch

from yolo_quantize import (
    QSiLUFixedPointConfig,
    QSiLUPQ,
    QSiLUPQProfile,
    emulate_qsilu_pq_legacy,
)


def test_qsilu_is_a_parameter_free_activation_not_an_output_quantizer() -> None:
    module = QSiLUPQ()

    assert list(module.parameters()) == []
    assert module.state_dict() == {}
    assert module.output_quantizer is None


def test_qsilu_profile_preserves_the_reviewed_dyadic_contract() -> None:
    profile = QSiLUPQProfile()

    assert profile.knots == (0.0, 1.0, 2.0, 4.0, 8.0)
    assert profile.quadratic_coefficients == (
        (57.0 / 256.0, 0.0, 0.0),
        (23.0 / 256.0, 17.0 / 64.0, -17.0 / 128.0),
        (-11.0 / 512.0, 91.0 / 128.0, -37.0 / 64.0),
        (-5.0 / 1024.0, 37.0 / 64.0, -5.0 / 16.0),
    )


def test_qsilu_is_c1_at_internal_knots_and_has_exact_relu_tails() -> None:
    module = QSiLUPQ().double()
    epsilon = 1e-7

    for knot in (1.0, 2.0, 4.0, 8.0):
        probe = torch.tensor(
            [knot - epsilon, knot + epsilon],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = module(probe)
        gradient = torch.autograd.grad(output.sum(), probe)[0]
        assert float((output[1] - output[0]).abs().detach()) < 3e-7
        assert float((gradient[1] - gradient[0]).abs()) < 3e-7

    tails = torch.tensor([-12.0, -8.0, 8.0, 12.0], dtype=torch.float64)
    assert torch.equal(module(tails), torch.tensor([0.0, 0.0, 8.0, 12.0]))


def test_qsilu_legacy_bittrue_emulator_matches_upstream_golden_vector() -> None:
    input_q = torch.tensor(
        [
            -12288,
            -8192,
            -4096,
            -2048,
            -1024,
            -512,
            0,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            12288,
        ]
    )

    output_q = emulate_qsilu_pq_legacy(input_q, QSiLUFixedPointConfig())

    assert output_q.tolist() == [
        0,
        0,
        -80,
        -248,
        -284,
        -199,
        0,
        313,
        740,
        1247,
        1800,
        2930,
        4016,
        6124,
        8192,
        12288,
    ]


def test_qsilu_legacy_emulator_names_its_rounding_and_range_contract() -> None:
    config = QSiLUFixedPointConfig(total_bits=15, fraction_bits=10)

    assert config.rounding_mode == "legacy_signed_half_away_from_zero"
    assert config.qmin == -(1 << 14)
    assert config.qmax == (1 << 14) - 1

    with pytest.raises(ValueError, match="at least Q\\*\\.10"):
        QSiLUFixedPointConfig(total_bits=16, fraction_bits=9)
