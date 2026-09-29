from __future__ import annotations

import argparse

import pytest

from yolo_quantize.preparation import (
    activation_region_assignment,
    active_parent_names,
    historical_parent_names,
    weight_analysis_plan,
)


def test_preparation_separates_active_and_historical_parent_choices() -> None:
    assert active_parent_names() == ("qsilu_pq", "hardswish", "poly_shift")
    assert historical_parent_names() == ("poly_quality",)


def test_preparation_parses_explicit_activation_region_assignment() -> None:
    assert activation_region_assignment(" neck_attention = hardswish ") == (
        "neck_attention",
        "hardswish",
    )
    with pytest.raises(argparse.ArgumentTypeError, match="REGION=ACTIVATION"):
        activation_region_assignment("hardswish")


def test_structured_profile_prioritizes_sd4_and_ternary_with_one_w4_control() -> None:
    plan = weight_analysis_plan("structured")

    assert plan.uniform_bits == (4,)
    assert plan.uniform_scale_methods == ("optimal_scaled_codebook",)
    assert plan.fixed_sd4_scale_methods == ("optimal_scaled_codebook",)
    assert plan.include_fixed_sd4 is True
    assert plan.include_paper_twn is True
    assert plan.include_filterwise_twn is True
    assert plan.include_exact_scaled_ternary is True


def test_unknown_weight_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown weight analysis profile"):
        weight_analysis_plan("unknown")
