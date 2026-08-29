#!/usr/bin/env python3
"""Run deterministic Phase 0 activation checks and emit machine-readable JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab import (
    ValidationConfig,
    available_activations,
    default_validation_plan,
    validate_activation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="驗證 Phase 0 activation 的曲線、固定點與 ONNX graph"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        choices=available_activations(),
        default=list(available_activations()),
    )
    parser.add_argument("--samples", type=int, default=24_001)
    parser.add_argument(
        "--onnx-dir", type=Path, help="若指定，匯出並稽核各 activation 的 ONNX graph"
    )
    parser.add_argument(
        "--output", type=Path, help="另存 JSON 報告；未指定時只輸出 stdout"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ValidationConfig(samples=args.samples)
    reports = []
    for name in args.names:
        onnx_path = args.onnx_dir / f"{name}.onnx" if args.onnx_dir else None
        reports.append(validate_activation(name, config, onnx_path=onnx_path).to_dict())
    payload = {
        "schema_version": 1,
        "scope": "pure-function proxy; not detector accuracy or hardware latency",
        "plan": asdict(default_validation_plan()),
        "reports": reports,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
