from __future__ import annotations

import math

import pytest

from yolo_attention.queue_policy import (
    SelectionInputError,
    select_architecture,
    select_bias,
    select_d1,
    select_d2,
    select_final,
    select_n0,
    select_normalization,
    select_r_denominator,
    select_recovery,
    select_scale,
)


def test_architecture_tie_uses_explicit_cost_order() -> None:
    decision = select_architecture({"i-scr": 0.4000, "h-scr": 0.4008, "t5-scr": 0.4009})

    assert decision.winners == ("i-scr",)
    assert set(decision.skipped) == {"h-scr", "t5-scr"}


def test_recovery_difference_below_point_zero_zero_one_prefers_direct() -> None:
    decision = select_recovery({"w-dir": 0.4000, "w-prog": 0.4009})

    assert decision.winners == ("w-dir",)


def test_recovery_exact_threshold_is_not_a_tie() -> None:
    decision = select_recovery({"w-dir": 0.4000, "w-prog": 0.4010})

    assert decision.winners == ("w-prog",)


def test_d1_tie_prefers_global_sharing() -> None:
    decision = select_d1(
        {"d1-shared-10": 0.3900, "d1-pattn-10": 0.3908, "d1-phead-10": 0.3909},
        five_epoch_metrics={"d1-shared": 0.3898, "d1-pattn": 0.3900, "d1-phead": 0.3901},
    )

    assert decision.winners == ("d1-shared-10",)
    assert decision.expand == ("d1-seed1",)


def test_d1_skips_second_seed_when_winner_is_converged_and_decisive() -> None:
    decision = select_d1(
        {"d1-shared-10": 0.4002, "d1-pattn-10": 0.3989, "d1-phead-10": 0.3980},
        five_epoch_metrics={"d1-shared": 0.4000, "d1-pattn": 0.3985, "d1-phead": 0.3978},
    )

    assert decision.winners == ("d1-shared-10",)
    assert decision.expand == ()


def test_bias_near_tie_prefers_decomposed_over_dense() -> None:
    decision = select_bias({"v1-b0": 0.399, "v1-bd": 0.4010, "v1-br": 0.4005})

    assert decision.winners == ("v1-br",)


def test_scale_near_tie_prefers_power_of_two_then_fixed_head() -> None:
    decision = select_scale({"v1-dyn": 0.4010, "v1-shead": 0.4008, "v1-p2": 0.4004})

    assert decision.winners == ("v1-p2",)


def test_normalization_uses_profile_cost_inside_map_tie_band() -> None:
    decision = select_normalization(
        {"n0-exact": 0.4000, "n1-lut": 0.4005},
        {
            "n0-exact": {"estimated_memory_traffic": 100, "arithmetic_cost_proxy": 80},
            "n1-lut": {"estimated_memory_traffic": 70, "arithmetic_cost_proxy": 60},
        },
    )

    assert decision.winners == ("n1-lut",)


def test_d2_rejects_candidates_outside_a0_gate_then_prefers_one_pot() -> None:
    decision = select_d2(
        {"d2-fp": 0.391, "d2-1p": 0.3905, "d2-2p": 0.3899},
        a0_map=0.400,
    )

    assert decision.winners == ("d2-1p",)
    assert "d2-2p" in decision.skipped


def test_n0_retains_at_most_two_candidates_inside_loss_gate() -> None:
    decision = select_n0(
        {
            "n0-lut": 0.395,
            "n0-pwl": 0.391,
            "n0-relu": 0.389,
            "n0-shift": 0.3889,
        },
        a0_map=0.400,
        cost_order=("n0-relu", "n0-shift", "n0-pwl", "n0-lut"),
    )

    assert decision.winners == ("n0-lut", "n0-pwl")
    assert set(decision.skipped) == {"n0-relu", "n0-shift"}


def test_r1_expands_newton_only_when_accuracy_or_row_sum_gate_is_missed() -> None:
    accuracy_failure = select_r_denominator(
        r0_map=0.4000,
        r1_map=0.3979,
        r1_row_sum_max_error=0.005,
    )
    row_failure = select_r_denominator(
        r0_map=0.4000,
        r1_map=0.3990,
        r1_row_sum_max_error=0.0101,
    )
    passing = select_r_denominator(
        r0_map=0.4000,
        r1_map=0.3980,
        r1_row_sum_max_error=0.01,
    )

    assert accuracy_failure.expand == ("r1-newton",)
    assert row_failure.expand == ("r1-newton",)
    assert passing.winners == ("r1-rlut",)
    assert passing.expand == ()


def test_final_selection_uses_map_band_then_memory_then_arithmetic() -> None:
    decision = select_final(
        {"a0": 0.4000, "n1": 0.4006, "bdcn": 0.4005},
        {
            "a0": {"estimated_memory_traffic": 100.0, "arithmetic_cost_proxy": 50.0},
            "n1": {"estimated_memory_traffic": 80.0, "arithmetic_cost_proxy": 60.0},
            "bdcn": {"estimated_memory_traffic": 80.0, "arithmetic_cost_proxy": 40.0},
        },
    )

    assert decision.winners == ("bdcn",)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "0.4", None])
def test_selection_rejects_missing_nonfinite_or_non_numeric_metrics(value: object) -> None:
    with pytest.raises(SelectionInputError):
        select_architecture({"i-scr": value, "h-scr": 0.4, "t5-scr": 0.4})


def test_final_selection_fails_closed_without_required_profile_fields() -> None:
    with pytest.raises(SelectionInputError, match="estimated_memory_traffic"):
        select_final(
            {"a0": 0.4, "n1": 0.4005},
            {
                "a0": {"estimated_memory_traffic": 100.0, "arithmetic_cost_proxy": 50.0},
                "n1": {"arithmetic_cost_proxy": 40.0},
            },
        )
