from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yolo_quantize import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS
from yolo_quantize.metric_regate import regate_search_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_role(root: Path, role: str, value: float) -> dict[str, object]:
    report = root / role / "metrics.json"
    report.parent.mkdir(parents=True)
    metrics = {
        **{key: value - 0.1 for key in FULL35_MAP50_95_KEYS},
        **{key: value for key in FULL35_MAP50_KEYS},
    }
    report.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    record: dict[str, object] = {
        "status": "completed",
        "policy_id": "accepted-silu" if role == "accepted" else "qsilu-a8",
        "metrics_report": str(report),
        "metrics_report_sha256": _sha256(report),
    }
    if role == "candidate":
        record["weight_quantization"] = {"format_id": "w8"}
    return record


def _source_report(tmp_path: Path) -> Path:
    roles = {
        "accepted": _write_role(tmp_path, "accepted", 0.80),
        "matched": _write_role(tmp_path, "matched", 0.79),
        "candidate": _write_role(tmp_path, "candidate", 0.788),
    }
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "status": "completed",
                "contract": {"plan_id": "legacy-v4"},
                "roles": roles,
            }
        ),
        encoding="utf-8",
    )
    return source


def test_regate_uses_pinned_raw_map50_without_rerunning_validation(
    tmp_path: Path,
) -> None:
    source = _source_report(tmp_path)

    payload = regate_search_report(
        source_report=source,
        expected_source_sha256=_sha256(source),
        metric_contract_id="full35-search-map50-v1",
    )

    assert payload["gpu_used"] is False
    assert payload["validation_rerun"] is False
    assert payload["gate"]["thresholds"]["metric_family"] == "map50"
    assert payload["gate"]["thresholds"]["total_drop"] == 0.015
    assert payload["gate"]["worst_total_delta"] == pytest.approx(-0.012)
    assert payload["gate"]["decision"] == "green"
    assert tuple(payload["roles"]["candidate"]["metrics"]) == FULL35_MAP50_KEYS


def test_regate_fails_closed_if_source_or_raw_metrics_digest_drift(
    tmp_path: Path,
) -> None:
    source = _source_report(tmp_path)

    with pytest.raises(RuntimeError, match="source report SHA-256"):
        regate_search_report(
            source_report=source,
            expected_source_sha256="0" * 64,
            metric_contract_id="full35-search-map50-v1",
        )

    payload = json.loads(source.read_text(encoding="utf-8"))
    metrics_path = Path(payload["roles"]["candidate"]["metrics_report"])
    metrics_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metrics SHA-256"):
        regate_search_report(
            source_report=source,
            expected_source_sha256=_sha256(source),
            metric_contract_id="full35-search-map50-v1",
        )
