from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/full35_profile_analysis.py"
SPEC = importlib.util.spec_from_file_location("full35_profile_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def test_weighted_errors_cover_preregistered_candidates() -> None:
    reports = ANALYSIS._weighted_errors(
        torch.ones(64, dtype=torch.int64),
        minimum=-8.0,
        maximum=8.0,
    )

    assert set(reports) == {
        "hardswish",
        "relu",
        "qsilu_pq",
        "poly_shift",
        "poly_quality",
    }
    assert all(
        math.isfinite(metric)
        for report in reports.values()
        for metric in report.values()
    )


def test_sensitivity_marks_training_only_regions_outside_deployment() -> None:
    rows = ANALYSIS._sensitivity_rows(
        {
            "rows": [
                {
                    "region": "pose_flow",
                    "deltas": {
                        "coco/box/map50_95": 0.0,
                        "bbat/box/map50_95": 0.0,
                        "bbat/pose/map50_95": 0.0,
                    },
                }
            ]
        }
    )

    assert rows[0]["deployment_graph"] is False
    assert rows[0]["risk_tier"] == "low"
