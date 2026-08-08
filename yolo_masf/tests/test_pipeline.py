from __future__ import annotations

import json
from pathlib import Path

from masf_yolo.evaluation.profiling import HardwareProfile
from masf_yolo.pipeline import candidate_from_artifacts, normalize_profile, run_final_audit


def _complete_class_metrics() -> dict[str, object]:
    return {
        "per_class": {"ball": {"ap": 0.4}, "bat": {"ap": 0.5}},
        "class_diagnostics": {
            "ball": {"gt_count": 2, "prediction_count": 2},
            "bat": {"gt_count": 1, "prediction_count": 1},
        },
    }


def test_candidate_uses_exact_quality_and_hardware_fields() -> None:
    metrics = {
        "map50_95": 0.51,
        "ap_s": 0.41,
        "ball_recall": 0.71,
        "ball_ap_s": 0.37,
        "ball_subsets": {
            "tiny": {"recall": 0.61},
            "blur_proxy": {"recall": 0.56},
        },
    }
    profile = HardwareProfile(20, 30, 0.00000006, 40, 50, 2, 3, 60)

    candidate = candidate_from_artifacts("M2", metrics, profile)

    assert candidate.variant_id == "M2"
    assert candidate.map50_95 == 0.51
    assert candidate.tiny_recall == 0.61
    assert candidate.gflops == profile.gflops
    assert candidate.peak_activation == 50


def test_b0_profile_does_not_mislabel_p3_as_p2_activation() -> None:
    profile = HardwareProfile(20, 30, 0.00000006, 40, 50, 2, 3, 60)

    assert normalize_profile("B0", profile).p2_activation_bytes is None
    assert normalize_profile("M7", profile) == profile


def test_final_audit_fails_closed_when_any_required_artifact_is_missing(tmp_path: Path) -> None:
    result = run_final_audit(tmp_path)

    assert result["ok"] is False
    assert result["errors"]
    assert (tmp_path / "final_audit.json").is_file()


def test_final_audit_rejects_missing_or_selection_eligible_b0_reference(tmp_path: Path) -> None:
    result = run_final_audit(tmp_path)
    assert "missing B0 reference manifest" in result["errors"]

    reference = tmp_path / "references"
    reference.mkdir()
    (reference / "b0.json").write_text(
        json.dumps(
            {
                "reference_id": "B0",
                "checkpoint_hash": "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d",
                "data_exposed": True,
                "selection_eligible": True,
            }
        )
    )

    result = run_final_audit(tmp_path)
    assert "B0 reference must be data-exposed and selection-ineligible" in result["errors"]


def test_final_audit_rejects_metrics_without_ball_and_bat_sections(tmp_path: Path) -> None:
    metrics = tmp_path / "evaluation" / "val" / "b0" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}")

    result = run_final_audit(tmp_path)

    assert "missing val per_class class ball: B0" in result["errors"]
    assert "missing val per_class class bat: B0" in result["errors"]
    assert "missing val class_diagnostics class ball: B0" in result["errors"]
    assert "missing val class_diagnostics class bat: B0" in result["errors"]


def test_final_audit_accepts_one_complete_phase1_matrix(tmp_path: Path) -> None:
    for stage in ("b1_a", "b1_b", "formal_m7", "formal_m0", "formal_m1", "formal_m2", "formal_m3"):
        path = tmp_path / "training" / stage
        path.mkdir(parents=True)
        checkpoint = path / "canonical.pt"
        checkpoint.write_bytes(stage.encode())
        (path / "run.json").write_text(json.dumps({"canonical": str(checkpoint), "strict_reload": True}))
    (tmp_path / "selection.json").write_text(json.dumps({"selected": "M3", "val_hashes": {"M2": "a", "M3": "b"}}))
    for split in ("val", "test"):
        for variant in ("B0", "B1", "M7", "M0", "M1", "M2", "M3"):
            path = tmp_path / "evaluation" / split / variant.lower()
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(json.dumps(_complete_class_metrics()))
    for variant in ("B0", "B1", "M7", "M0", "M1", "M2", "M3"):
        path = tmp_path / "profiles" / variant.lower()
        path.mkdir(parents=True)
        (path / "profile.json").write_text("{}")
    reference = tmp_path / "references"
    reference.mkdir()
    (reference / "b0.json").write_text(
        json.dumps(
            {
                "reference_id": "B0",
                "checkpoint_hash": "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d",
                "data_exposed": True,
                "selection_eligible": False,
            }
        )
    )

    result = run_final_audit(tmp_path)

    assert result == {"ok": True, "errors": [], "best_partial": "M3"}
