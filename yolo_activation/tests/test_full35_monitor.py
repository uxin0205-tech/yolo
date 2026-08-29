from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/full35_monitor.py"
SPEC = importlib.util.spec_from_file_location("full35_monitor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MONITOR
SPEC.loader.exec_module(MONITOR)


def test_compact_status_reads_only_latest_event_and_metrics(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events = run / "logs/events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        '{"kind":"macro","step":1}\n'
        '{"kind":"macro","step":2,"context":{"epoch":0,"macro_in_epoch":2},'
        '"values":{"amp/overflow_retries":0,"loss/joint_mean":1.5}}\n',
        encoding="utf-8",
    )
    metrics = run / "validation/epoch-0000/bittrue/metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "epoch": 0,
                "metrics": {
                    "coco/box/map50_95": 0.49,
                    "bbat/box/map50_95": 0.62,
                    "bbat/pose/map50_95": 0.90,
                },
            }
        ),
        encoding="utf-8",
    )

    status = MONITOR.compact_status(run)

    assert status["train"] == {
        "kind": "macro",
        "step": 2,
        "epoch": 0,
        "macro": 2,
        "overflow": 0,
        "joint_loss": 1.5,
    }
    assert status["bittrue"]["bbat_pose"] == 0.90
