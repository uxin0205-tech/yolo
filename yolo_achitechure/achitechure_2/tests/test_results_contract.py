from __future__ import annotations

import csv
from pathlib import Path


def test_results_template_is_rectangular_pending_and_metric_complete() -> None:
    path = Path("results/results-template.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["candidate_id"] for row in rows] == [
        "C0-Handoff",
        "C0",
        "C1",
        "C2",
        "C3",
    ]
    assert all(None not in row for row in rows)
    assert all(row["run_status"] == "not_run" for row in rows)
    assert all(row["formal_comparison_ready"] == "false" for row in rows)
    assert all(row["selection_status"] == "pending_user_decision" for row in rows)
    assert all(row["c_best"] == "" for row in rows)
    assert all(row["spec_version"] == "2.0.3" for row in rows)
    required = {
        "coco_box_map50",
        "coco_box_map50_95",
        "coco_person_ap50",
        "coco_person_ap50_95",
        "bbat5_pose_box_map50",
        "bbat5_pose_box_map50_95",
        "bbat5_keypoint_map50",
        "bbat5_keypoint_map50_95",
        "pose_official_combined_fitness",
        "ball_f1",
        "bat_f1",
        "macro_f1",
        "micro_f1",
        "f1_confidence_threshold",
    }
    assert required.issubset(rows[0])


def test_report_template_keeps_pose_execution_as_explicit_choice() -> None:
    report = Path("results/report-template.md").read_text(encoding="utf-8")

    assert "Pose 是否經使用者 opt-in 並完成" in report
    assert "C_best：null（等待使用者決定）" in report
    assert "Official combined fitness" in report
