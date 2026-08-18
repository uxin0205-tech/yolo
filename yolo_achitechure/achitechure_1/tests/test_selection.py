from __future__ import annotations

from achitechure_1.selection import Candidate, choose_architecture, phase_gate


def test_phase_gate_rolls_back_only_when_child_loses_more_than_point001() -> None:
    assert phase_gate("parent", 0.5000, "child", 0.4990) == "child"
    assert phase_gate("parent", 0.5000, "child", 0.4989) == "parent"


def test_architecture_tie_uses_the_contractual_cost_order() -> None:
    full = Candidate("a1", 0.5005, 2.0, 60.0, 40_000_000, 20.0)
    partial = Candidate("a2", 0.5000, 1.5, 61.0, 41_000_000, 21.0)

    assert choose_architecture((full, partial)).name == "a2"


def test_map_wins_when_difference_exceeds_point001() -> None:
    accurate = Candidate("a1", 0.5011, 2.0, 60.0, 40_000_000, 20.0)
    fast = Candidate("a2", 0.5000, 1.0, 30.0, 20_000_000, 10.0)

    assert choose_architecture((accurate, fast)).name == "a1"
