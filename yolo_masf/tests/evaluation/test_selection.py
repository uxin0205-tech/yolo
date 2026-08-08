from __future__ import annotations

import json
from pathlib import Path

import pytest

from masf_yolo.evaluation.selection import (
    CandidateMetrics,
    freeze_selection,
    require_selection_before_test,
    select_best_partial,
)


def _candidate(variant: str, **changes: float) -> CandidateMetrics:
    values = {
        "map50_95": 0.5,
        "ap_s": 0.4,
        "ball_recall": 0.7,
        "ball_ap_s": 0.35,
        "tiny_recall": 0.6,
        "blur_recall": 0.55,
        "gflops": 100.0,
        "params": 20_000_000.0,
        "peak_activation": 10_000_000.0,
        "traffic": 100_000_000.0,
    }
    values.update(changes)
    return CandidateMetrics(variant_id=variant, **values)


def test_m3_wins_efficiency_equivalent_boundary() -> None:
    m2 = _candidate("M2", map50_95=0.502, ball_recall=0.708)
    m3 = _candidate("M3", map50_95=0.500, ball_recall=0.700, gflops=80)

    result = select_best_partial(m2, m3)

    assert result.selected == "M3"
    assert result.reason == "efficiency_equivalent"


def test_recall_difference_equal_point01_uses_quality_ranking() -> None:
    m2 = _candidate("M2", map50_95=0.502, ball_recall=0.71)
    m3 = _candidate("M3", map50_95=0.500, ball_recall=0.70, gflops=80)

    assert select_best_partial(m2, m3).selected == "M2"


def test_ranking_uses_quality_then_hardware_order() -> None:
    base = _candidate("M2")
    better_blur = _candidate("M3", blur_recall=0.56, gflops=1000)
    assert select_best_partial(base, better_blur).selected == "M3"

    same_quality_cheaper = _candidate("M3", gflops=99)
    assert select_best_partial(base, same_quality_cheaper).selected == "M3"


def test_selection_is_immutable_and_required_before_test(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    result = select_best_partial(_candidate("M2", map50_95=0.6), _candidate("M3"))

    freeze_selection(path, result, val_hashes={"M2": "a", "M3": "b"})
    require_selection_before_test(path)
    first = json.loads(path.read_text())
    freeze_selection(path, result, val_hashes={"M2": "a", "M3": "b"})
    assert json.loads(path.read_text()) == first

    with pytest.raises(RuntimeError, match="immutable"):
        freeze_selection(path, result, val_hashes={"M2": "changed", "M3": "b"})
    with pytest.raises(RuntimeError, match="selection"):
        require_selection_before_test(tmp_path / "missing.json")
