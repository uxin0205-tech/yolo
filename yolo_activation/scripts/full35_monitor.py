#!/usr/bin/env python3
"""Print one compact JSON line for a running Full35 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts/runs/full35"


def _last_json_line(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0 and buffer.count(b"\n") < 2:
            size = min(4096, position)
            position -= size
            handle.seek(position)
            buffer = handle.read(size) + buffer
    for line in reversed(buffer.splitlines()):
        if line.strip():
            return json.loads(line)
    return None


def compact_status(run: Path) -> dict[str, Any]:
    event = _last_json_line(run / "logs/events.jsonl")
    train = None
    if event is not None:
        context = event.get("context", {})
        report = context.get("report", {})
        values = event.get("values", {})
        train = {
            "kind": event.get("kind"),
            "step": event.get("step"),
            "epoch": context.get("epoch", report.get("epoch")),
            "macro": context.get("macro_in_epoch"),
            "overflow": values.get("amp/overflow_retries"),
            "joint_loss": values.get("loss/joint_mean"),
        }

    metrics_paths = sorted(run.glob("validation/epoch-*/bittrue/metrics.json"))
    bittrue = None
    if metrics_paths:
        payload = json.loads(metrics_paths[-1].read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        bittrue = {
            "epoch": payload["epoch"],
            "coco": metrics["coco/box/map50_95"],
            "bbat_box": metrics["bbat/box/map50_95"],
            "bbat_pose": metrics["bbat/pose/map50_95"],
        }
    return {"run": run.name, "train": train, "bittrue": bittrue}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full35 單行低 token 監測")
    parser.add_argument("run", help="run 名稱或絕對路徑")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = Path(args.run).expanduser()
    run = candidate if candidate.is_absolute() else DEFAULT_RUN_ROOT / candidate
    print(json.dumps(compact_status(run.resolve()), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
