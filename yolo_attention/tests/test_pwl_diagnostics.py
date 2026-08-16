from __future__ import annotations

import math

import pytest
import torch

from yolo_attention.pwl_experiment import choose_pwl_range
from yolo_attention.pwl_validation import PWLValidationAccumulator


def test_pwl_accumulator_reports_score_tail_probability_and_pv_error_per_head() -> None:
    accumulator = PWLValidationAccumulator(site="model.10.m.0.attn", heads=1)
    scores = torch.tensor([[[[0.0, -1.0, -9.0], [-2.0, 0.0, -10.0]]]])
    values = torch.tensor([[[[1.0, 2.0, 4.0], [0.0, 1.0, 3.0]]]])

    accumulator.update(scores, values)
    summary = accumulator.summary()

    aggregate = summary["aggregate"]
    assert aggregate["count"] == 6
    assert aggregate["min"] == -10.0
    assert aggregate["mean"] == pytest.approx(-22.0 / 6.0)
    assert aggregate["ratio_lt_neg8"] == pytest.approx(2.0 / 6.0)
    expected_tail_mass = (math.exp(-9.0) / sum(math.exp(x) for x in (0.0, -1.0, -9.0)))
    expected_tail_mass += math.exp(-10.0) / sum(math.exp(x) for x in (-2.0, 0.0, -10.0))
    assert aggregate["exact_tail_probability_mass_mean"] == pytest.approx(expected_tail_mass / 2)
    assert aggregate["float_probability_mae"] >= 0.0
    assert aggregate["bit_true_probability_mae"] >= aggregate["float_probability_mae"]
    assert 0.0 <= aggregate["bit_true_pv_cosine_similarity"] <= 1.0
    assert summary["heads"][0]["head"] == 0


def test_pwl_accumulator_rejects_wrong_head_or_value_shape() -> None:
    accumulator = PWLValidationAccumulator(site="model.10.m.0.attn", heads=2)
    with pytest.raises(ValueError, match="heads"):
        accumulator.update(torch.zeros(1, 1, 3, 3), torch.zeros(1, 1, 2, 3))
    with pytest.raises(ValueError, match="tokens"):
        accumulator.update(torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 2, 4))


def test_pwl_pv_reference_stays_fp32_inside_autocast() -> None:
    values = torch.randn(1, 1, 4, 8)
    probability = torch.softmax(torch.randn(1, 1, 8, 8), dim=-1)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        pv = PWLValidationAccumulator._pv(values, probability)

    assert pv.dtype == torch.float32


def test_pwl_range_gate_keeps_half_point_indexing() -> None:
    assert choose_pwl_range(0.001).score_floor == -8.0
    expanded = choose_pwl_range(0.0010001)
    assert (expanded.score_floor, expanded.segments, expanded.segment_width) == (-10.0, 20, 0.5)
