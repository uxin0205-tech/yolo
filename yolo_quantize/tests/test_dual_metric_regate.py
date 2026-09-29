from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yolo_quantize.dual_metric_regate import regate_candidate_report
from yolo_quantize.metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw(root: Path, name: str, map50: float, map50_95: float) -> dict[str, str]:
    path = root / name / "metrics.json"
    path.parent.mkdir(parents=True)
    metrics = {
        **{key: map50 for key in FULL35_MAP50_KEYS},
        **{key: map50_95 for key in FULL35_MAP50_95_KEYS},
    }
    path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    return {"path": str(path), "sha256": _sha256(path)}


def test_dual_regate_catches_map50_95_loss_hidden_by_green_map50(
    tmp_path: Path,
) -> None:
    accepted = _raw(tmp_path, "accepted", 0.80, 0.70)
    matched = _raw(tmp_path, "matched", 0.79, 0.68)
    candidate = _raw(tmp_path, "candidate", 0.79, 0.659)
    report = tmp_path / "race.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "reference": {
                    "roles": {
                        "accepted": {
                            "source_metrics_report": accepted["path"],
                            "source_metrics_report_sha256": accepted["sha256"],
                        },
                        "matched": {
                            "source_metrics_report": matched["path"],
                            "source_metrics_report_sha256": matched["sha256"],
                        },
                    }
                },
                "results": {
                    "candidate-a": {
                        "status": "completed",
                        "metrics_report": candidate["path"],
                        "metrics_report_sha256": candidate["sha256"],
                        "gate": {"passed": True, "decision": "green"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = regate_candidate_report(
        source_report=report,
        expected_source_sha256=_sha256(report),
        metric_contract_id="dual-search-v1",
    )

    result = payload["candidates"]["candidate-a"]
    assert result["source_map50_gate_passed"] is True
    assert result["dual_gate_passed"] is False
    assert result["decision"] == "recover"
    assert result["worst_map50_delta"] == pytest.approx(-0.01)
    assert result["worst_map50_95_delta"] == pytest.approx(-0.041)
    assert result["worst_incremental_map50_95_delta"] == pytest.approx(-0.021)
    assert result["metric_summaries"]["candidate"] == {
        "mean_required_map50": pytest.approx(0.79),
        "mean_required_map50_95": pytest.approx(0.659),
        "joint_priority_map50": pytest.approx(0.79),
        "joint_priority_map50_95": pytest.approx(0.659),
    }
    assert result["metric_summaries"]["semantics"] == {
        "mean_required": "diagnostic_mean_not_official_map",
        "joint_priority": "fixed_0.2_0.2_0.2_0.4_selection_score_not_official_map",
    }


def test_dual_regate_fails_when_raw_metric_digest_drifts(tmp_path: Path) -> None:
    accepted = _raw(tmp_path, "accepted", 0.80, 0.70)
    matched = _raw(tmp_path, "matched", 0.79, 0.68)
    candidate = _raw(tmp_path, "candidate", 0.79, 0.67)
    report = tmp_path / "race.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "reference": {
                    "roles": {
                        "accepted": {
                            "source_metrics_report": accepted["path"],
                            "source_metrics_report_sha256": accepted["sha256"],
                        },
                        "matched": {
                            "source_metrics_report": matched["path"],
                            "source_metrics_report_sha256": matched["sha256"],
                        },
                    }
                },
                "results": {
                    "candidate-a": {
                        "status": "completed",
                        "metrics_report": candidate["path"],
                        "metrics_report_sha256": "0" * 64,
                        "gate": {"passed": True, "decision": "green"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="SHA-256"):
        regate_candidate_report(
            source_report=report,
            expected_source_sha256=_sha256(report),
            metric_contract_id="dual-search-v1",
        )


def test_dual_regate_resolves_hash_pinned_activation_reference_report(
    tmp_path: Path,
) -> None:
    accepted = _raw(tmp_path, "accepted", 0.80, 0.70)
    matched = _raw(tmp_path, "matched", 0.79, 0.69)
    candidate = _raw(tmp_path, "candidate", 0.79, 0.68)
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "status": "completed",
                "reference": {
                    "roles": {
                        "accepted": {
                            "source_metrics_report": accepted["path"],
                            "source_metrics_report_sha256": accepted["sha256"],
                        },
                        "matched": {
                            "source_metrics_report": matched["path"],
                            "source_metrics_report_sha256": matched["sha256"],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "activation.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "contract": {
                    "reference_reuse": {
                        "report": str(reference),
                        "sha256": _sha256(reference),
                    }
                },
                "results": {
                    "regional-hardswish": {
                        "status": "completed",
                        "metrics_report": candidate["path"],
                        "metrics_report_sha256": candidate["sha256"],
                        "gate": {"passed": True, "decision": "green"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = regate_candidate_report(
        source_report=report,
        expected_source_sha256=_sha256(report),
        metric_contract_id="dual-activation-v1",
    )

    result = payload["candidates"]["regional-hardswish"]
    assert result["decision"] == "green"
    assert result["worst_map50_delta"] == pytest.approx(-0.01)
    assert result["worst_map50_95_delta"] == pytest.approx(-0.02)
    assert result["worst_incremental_map50_95_delta"] == pytest.approx(-0.01)
