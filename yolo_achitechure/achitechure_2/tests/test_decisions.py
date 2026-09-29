from __future__ import annotations

import pytest

from achitechure_2.decisions import (
    CandidateMetrics,
    ClassMetrics,
    evaluate_float_results,
)


def _metric(
    candidate_id: str,
    *,
    accuracy: float = 0.50,
    params: int = 20_000_000,
    gflops: float = 70.0,
    latency: float | None = 10.0,
    threshold: float = 0.35,
) -> CandidateMetrics:
    ball = ClassMetrics(0.70, 0.48, 0.66, 0.45, 0.80, 0.75, 0.774)
    bat = ClassMetrics(0.68, 0.46, 0.64, 0.43, 0.70, 0.65, 0.674)
    return CandidateMetrics(
        candidate_id=candidate_id,
        coco_box_map50=0.60 + accuracy / 10,
        coco_box_map50_95=accuracy,
        coco_person_ap50=0.70,
        coco_person_ap50_95=accuracy + 0.05,
        bbat5_pose_box_map50=0.72,
        bbat5_pose_box_map50_95=accuracy,
        bbat5_keypoint_map50=0.65,
        bbat5_keypoint_map50_95=accuracy,
        pose_official_combined_fitness=accuracy * 2,
        classes={"ball": ball, "bat": bat},
        macro_f1=(ball.f1 + bat.f1) / 2,
        micro_f1=0.73,
        f1_confidence_threshold=threshold,
        params=params,
        gflops=gflops,
        latency_ms=latency,
        peak_vram_mb=None,
    )


def test_report_keeps_ap50_ap50_95_map_and_f1_separate() -> None:
    report = evaluate_float_results((_metric("C0"), _metric("C1", accuracy=0.496)))
    payload = report.to_dict()
    c1 = payload["candidates"][1]["metrics"]

    assert c1["coco_box_map50"] != c1["coco_box_map50_95"]
    assert c1["classes"]["ball"]["ap50"] != c1["classes"]["ball"]["ap50_95"]
    assert c1["macro_f1"] != c1["micro_f1"]
    assert c1["f1_confidence_threshold"] == 0.35
    assert c1["pose_official_combined_fitness"] == pytest.approx(0.992)


def test_first_round_is_measurement_only_and_never_selects_c_best() -> None:
    report = evaluate_float_results(
        (
            _metric("C0"),
            _metric("C1", accuracy=0.497, params=18_000_000, gflops=62.0, latency=9.0),
            _metric("C2", accuracy=0.492, params=17_000_000, gflops=60.0, latency=8.8),
            _metric("C3", accuracy=0.490, params=16_000_000, gflops=58.0, latency=8.5),
        )
    )

    assert report.selection_status == "pending_user_decision"
    assert report.c_best is None
    assert report.quantization_eligibility == {
        "C0": True,
        "C1": "pending_user_decision",
        "C2": "pending_user_decision",
        "C3": "pending_user_decision",
    }
    assert all(item.decision is None for item in report.candidates)
    assert report.candidates[1].descriptive_bands["coco_box_map50_95"] == "drop_le_0.005"
    assert report.candidates[2].descriptive_bands["coco_box_map50_95"] == "drop_0.005_to_0.008"
    assert report.candidates[3].descriptive_bands["coco_box_map50_95"] == "drop_gt_0.008"


def test_pareto_is_descriptive_and_missing_gpu_cost_stays_pending() -> None:
    report = evaluate_float_results(
        (
            _metric("C0"),
            _metric("C1", accuracy=0.50, params=18_000_000, gflops=60.0, latency=9.0),
            _metric("C2", accuracy=0.49, params=19_000_000, gflops=65.0, latency=9.5),
            _metric("C3", accuracy=0.51, params=17_000_000, gflops=58.0, latency=None),
        )
    )

    assert "C1" in report.pareto_front
    assert "C2" not in report.pareto_front
    assert report.pareto_pending == ("C3",)


def test_all_candidates_reuse_c0_search_val_f1_threshold() -> None:
    with pytest.raises(ValueError, match="F1 threshold"):
        evaluate_float_results((_metric("C0"), _metric("C1", threshold=0.4)))


def test_macro_f1_must_equal_unweighted_ball_bat_mean() -> None:
    value = _metric("C0")
    with pytest.raises(ValueError, match="Macro F1"):
        CandidateMetrics(**{**value.__dict__, "macro_f1": 0.99})
