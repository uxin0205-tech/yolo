from __future__ import annotations

import pytest

from yolo_quantize import BalanceCandidate, QuantizationBalanceSelector


def _candidate(
    candidate_id: str,
    *,
    worst_total_delta: float,
    packed_weight_bytes: int,
    passed: bool = True,
) -> BalanceCandidate:
    return BalanceCandidate(
        candidate_id=candidate_id,
        policy_id="qsilu-a8",
        format_id=candidate_id,
        worst_total_delta=worst_total_delta,
        packed_weight_bytes=packed_weight_bytes,
        reference_fp32_bytes=400,
        hard_gate_passed=passed,
    )


def test_selector_excludes_any_candidate_that_misses_the_hard_gate() -> None:
    result = QuantizationBalanceSelector().select(
        (
            _candidate("w8", worst_total_delta=-0.004, packed_weight_bytes=100),
            _candidate(
                "w4-failed",
                worst_total_delta=-0.021,
                packed_weight_bytes=50,
                passed=False,
            ),
        )
    )

    assert result.eligible_ids == ("w8",)
    assert result.rejected_ids == ("w4-failed",)
    assert result.pareto_ids == ("w8",)


def test_selector_keeps_the_accuracy_storage_pareto_frontier() -> None:
    result = QuantizationBalanceSelector().select(
        (
            _candidate("w8", worst_total_delta=-0.002, packed_weight_bytes=100),
            _candidate(
                "w7-dominated", worst_total_delta=-0.003, packed_weight_bytes=110
            ),
            _candidate("w6", worst_total_delta=-0.006, packed_weight_bytes=75),
            _candidate("w4", worst_total_delta=-0.014, packed_weight_bytes=50),
        )
    )

    assert result.pareto_ids == ("w8", "w6", "w4")
    assert result.accuracy_id == "w8"
    assert result.hardware_id == "w4"


def test_balanced_role_compresses_within_two_tenths_of_a_point() -> None:
    result = QuantizationBalanceSelector(accuracy_tolerance=0.002).select(
        (
            _candidate("w8", worst_total_delta=-0.002, packed_weight_bytes=100),
            _candidate("w6", worst_total_delta=-0.0035, packed_weight_bytes=75),
            _candidate("w5", worst_total_delta=-0.009, packed_weight_bytes=62),
        )
    )

    assert result.balanced_id == "w6"


def test_selector_enforces_the_map50_one_and_a_half_point_budget_itself() -> None:
    result = QuantizationBalanceSelector().select(
        (
            _candidate("edge", worst_total_delta=-0.015, packed_weight_bytes=70),
            _candidate(
                "old-gate-only",
                worst_total_delta=-0.0151,
                packed_weight_bytes=50,
            ),
        )
    )

    assert result.eligible_ids == ("edge",)
    assert result.rejected_ids == ("old-gate-only",)


def test_selector_fails_closed_on_duplicate_ids_or_invalid_costs() -> None:
    selector = QuantizationBalanceSelector()
    duplicate = _candidate("same", worst_total_delta=-0.01, packed_weight_bytes=50)

    with pytest.raises(ValueError, match="unique"):
        selector.select((duplicate, duplicate))
    with pytest.raises(ValueError, match="packed_weight_bytes"):
        BalanceCandidate(
            candidate_id="bad",
            policy_id="qsilu-a8",
            format_id="w4",
            worst_total_delta=-0.01,
            packed_weight_bytes=0,
            reference_fp32_bytes=400,
            hard_gate_passed=True,
        )
