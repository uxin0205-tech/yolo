import hashlib
import json

import pytest

from yolo_quantize.metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS
from yolo_quantize.search_evidence import read_dual_search_metrics


def test_dual_reader_uses_raw_report_not_legacy_eight_metric_summary(tmp_path):
    metrics = {k: 0.5 for k in (*FULL35_MAP50_KEYS, *FULL35_MAP50_95_KEYS)}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"metrics": metrics}))
    record = {
        "metrics": {k: 0.5 for k in FULL35_MAP50_KEYS},
        "metrics_report": str(path),
        "metrics_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert len(record["metrics"]) == 8
    assert read_dual_search_metrics(record) == metrics
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        read_dual_search_metrics(record)


def test_dual_reader_rejects_missing_map95(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"metrics": {k: 0.5 for k in FULL35_MAP50_KEYS}}))
    record = {
        "metrics_report": str(path),
        "metrics_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="missing required dual"):
        read_dual_search_metrics(record)
