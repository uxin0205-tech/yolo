from __future__ import annotations

import pytest

from achitechure_2.decisions import (
    CandidateMetrics,
    Decision,
    choose_c_best,
    classify_candidate,
    should_extend,
    trigger_c3_p5_fallback,
    trigger_r1,
    validate_conditional_candidates,
)


def metric(name: str, score: float, latency: float = 10.0, gflops: float = 70.0, params: int = 20_000_000):
    return CandidateMetrics(name, score, latency, gflops, params)


def test_extension_gate_exact_rules() -> None:
    flat = [0.4] * 100
    assert should_extend(flat, best_epoch=90, early_stopped=False).extend
    improving = [0.4] * 80 + [0.401] * 20
    assert should_extend(improving, best_epoch=80, early_stopped=False).extend
    assert not should_extend(flat, best_epoch=80, early_stopped=False).extend
    assert not should_extend(flat, best_epoch=90, early_stopped=True).extend


def test_pass_conditional_reject_boundaries_and_triggers() -> None:
    c0 = metric("C0", 0.5, 10.0, 100.0)
    passed = classify_candidate(metric("C1", 0.495, 9.5, 95.0), c0)
    conditional = classify_candidate(metric("C2", 0.492, 9.3, 91.9), c0)
    rejected = classify_candidate(metric("C3", 0.4919, 8.0, 80.0), c0)
    assert passed.decision is Decision.PASS and passed.eligible
    assert conditional.decision is Decision.CONDITIONAL and conditional.eligible
    assert trigger_r1(conditional)
    assert rejected.decision is Decision.REJECT and trigger_c3_p5_fallback(rejected)


def test_conditional_requires_cost_and_pareto_sort_is_deterministic() -> None:
    c0 = metric("C0", 0.5, 10.0, 100.0)
    no_cost = classify_candidate(metric("C2", 0.493, 9.5, 95.0), c0)
    candidate_a = classify_candidate(metric("C1", 0.496, 9.0, 90.0, 19_000_000), c0)
    candidate_b = classify_candidate(metric("C3", 0.496, 8.8, 91.0, 18_000_000), c0)
    assert not no_cost.eligible
    assert choose_c_best((no_cost, candidate_a, candidate_b)) == candidate_b
    assert choose_c_best((no_cost,)) is None


def test_conditional_candidates_require_parent_trigger_and_r1_fusion() -> None:
    c0 = metric("C0", 0.5, 10.0, 100.0)
    c2 = classify_candidate(metric("C2", 0.493, 9.0, 90.0), c0)
    r1 = classify_candidate(metric("R1", 0.496, 9.1, 91.0), c0)
    c3 = classify_candidate(metric("C3", 0.49, 9.0, 90.0), c0)
    p5 = classify_candidate(metric("C3-P5", 0.496, 9.5, 95.0), c0)
    with pytest.raises(ValueError, match="fusion"):
        validate_conditional_candidates((c2, r1))
    validate_conditional_candidates((c2, r1), r1_fusion_passed=True)
    validate_conditional_candidates((c3, p5))
    with pytest.raises(ValueError, match="C3-P5 requires"):
        validate_conditional_candidates((p5,))
