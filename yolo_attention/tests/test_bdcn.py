from __future__ import annotations

import pytest
import torch

from yolo_attention.bdcn import BDCNCodebookBank, BDCNNormalizer, project_one_pot, project_two_pot
from yolo_attention.config import BDCNCodebookKind, BDCNDenominator, BDCNProjection


def make_bank(projection: BDCNProjection = BDCNProjection.FLOAT) -> BDCNCodebookBank:
    return BDCNCodebookBank(
        num_tables=2,
        levels=16,
        step=0.125,
        kind=BDCNCodebookKind.LEARNED,
        projection=projection,
    )


def test_bdcn_codebook_is_positive_monotonic_and_exact_rows_normalize() -> None:
    bank = make_bank()
    module = BDCNNormalizer(bank, torch.tensor([0, 1]), 0.125, BDCNDenominator.EXACT)
    probability = module(torch.randn(2, 2, 7, 7))
    codebook = bank.codebook()

    assert torch.all(codebook > 0)
    assert torch.all(codebook[:, :-1] >= codebook[:, 1:])
    assert torch.all(probability >= 0)
    torch.testing.assert_close(probability.sum(-1), torch.ones(2, 2, 7))


@pytest.mark.parametrize("denominator", [BDCNDenominator.RECIPROCAL_LUT, BDCNDenominator.POT_SHIFT])
def test_hardware_denominators_are_finite_and_nonnegative(denominator: BDCNDenominator) -> None:
    module = BDCNNormalizer(make_bank(), torch.tensor([0, 1]), 0.125, denominator)
    probability = module(torch.randn(1, 2, 5, 5))

    assert torch.isfinite(probability).all()
    assert (probability >= 0).all()


def test_pot_shift_row_sum_error_is_bounded_by_nearest_power_of_two() -> None:
    module = BDCNNormalizer(
        make_bank(BDCNProjection.ONE_POT),
        torch.tensor([0, 1]),
        0.125,
        BDCNDenominator.POT_SHIFT,
    ).eval()
    probability = module(torch.randn(4, 2, 31, 31))
    row_sums = probability.sum(dim=-1)

    assert row_sums.min().item() >= 2**-0.5 - 1e-6
    assert row_sums.max().item() <= 2**0.5 + 1e-6


def test_pot_shift_grouped_value_matches_materialized_probability() -> None:
    module = BDCNNormalizer(
        make_bank(BDCNProjection.ONE_POT),
        torch.tensor([0, 1]),
        0.125,
        BDCNDenominator.POT_SHIFT,
    ).eval()
    scores = torch.randn(1, 2, 13, 13)
    value = torch.randn(1, 2, 7, 13)
    expected = value @ module(scores).transpose(-2, -1)

    torch.testing.assert_close(module.aggregate(scores, value), expected, rtol=1e-5, atol=1e-6)


def test_reciprocal_lut_preserves_row_sum_closely() -> None:
    module = BDCNNormalizer(make_bank(), torch.tensor([0, 1]), 0.125, BDCNDenominator.RECIPROCAL_LUT)
    probability = module(torch.randn(2, 2, 11, 11))

    assert (probability.sum(-1) - 1.0).abs().max().item() < 0.01


def test_two_pot_projection_is_never_worse_than_one_pot() -> None:
    codebook = torch.tensor([[1.0, 0.73, 0.41, 0.19, 0.07]])
    one_error = (project_one_pot(codebook) - codebook).abs()
    two_error = (project_two_pot(codebook) - codebook).abs()
    assert torch.all(two_error <= one_error + 1e-7)


def test_exact_grouped_value_path_matches_probability_times_value() -> None:
    module = BDCNNormalizer(make_bank(), torch.tensor([0, 1]), 0.125, BDCNDenominator.EXACT)
    scores = torch.randn(1, 2, 9, 9)
    value = torch.randn(1, 2, 6, 9)
    expected = value @ module(scores).transpose(-2, -1)

    torch.testing.assert_close(module.aggregate(scores, value), expected, rtol=1e-5, atol=1e-6)
    assert module.last_row_sums is not None
    torch.testing.assert_close(module.last_row_sums, torch.ones(1, 2, 9))


def test_grouped_value_path_avoids_half_precision_bucket_overflow() -> None:
    bank = BDCNCodebookBank(
        num_tables=1,
        levels=4,
        step=0.125,
        kind=BDCNCodebookKind.LEARNED,
        projection=BDCNProjection.FLOAT,
    )
    module = BDCNNormalizer(bank, torch.tensor([0]), 0.125, BDCNDenominator.EXACT)
    scores = torch.zeros(1, 1, 66, 66, dtype=torch.float16)
    value = torch.full((1, 1, 1, 66), 1000.0, dtype=torch.float16)

    with torch.autocast("cpu", dtype=torch.float16):
        output = module.aggregate(scores, value)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), torch.full_like(output.float(), 1000.0))


def test_one_newton_step_reduces_reciprocal_lut_row_sum_error() -> None:
    bank = BDCNCodebookBank(
        num_tables=1,
        levels=16,
        step=0.125,
        kind=BDCNCodebookKind.LEARNED,
        projection=BDCNProjection.TWO_POT,
    ).eval()
    scores = torch.randn(2, 1, 9, 9)
    plain = BDCNNormalizer(
        bank,
        torch.tensor([0]),
        0.125,
        BDCNDenominator.RECIPROCAL_LUT,
        reciprocal_newton_steps=0,
    ).eval()
    refined = BDCNNormalizer(
        bank,
        torch.tensor([0]),
        0.125,
        BDCNDenominator.RECIPROCAL_LUT,
        reciprocal_newton_steps=1,
    ).eval()

    plain_error = (plain(scores).sum(dim=-1) - 1.0).abs().max()
    refined_error = (refined(scores).sum(dim=-1) - 1.0).abs().max()

    assert refined_error <= plain_error
