"""Rebuild the human-readable report from immutable pipeline artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def rebuild_report(config_path: Path) -> str:
    from masf_yolo.cli import load_config
    from masf_yolo.variants import EVALUATED_MODELS

    config_path = config_path.resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    artifacts = root / config.values["artifacts_root"]
    dataset = _load(artifacts / "dataset" / "manifest.json") or {}
    environment = _load(artifacts / "environment.json") or {}
    selection = _load(artifacts / "selection.json") or {}
    final_audit = _load(artifacts / "final_audit.json") or {}
    state = _load(artifacts / "state.json") or {}
    b0_reference = _load(artifacts / "references" / "b0.json") or {}
    lines = [
        "# MASF-YOLO Phase 1 Report",
        "",
        f"Pipeline state: {state.get('status', 'not started')} / {state.get('stage', 'none')}",
        f"Dataset hash: {dataset.get('dataset_hash', 'missing')}",
        f"Environment: Ultralytics {environment.get('ultralytics', 'missing')}, "
        f"PyTorch {environment.get('torch', 'missing')}, device {environment.get('device_name', 'missing')}",
        f"BEST_PARTIAL: {selection.get('selected', 'not selected')}",
        f"Selection reason: {selection.get('reason', 'not available')}",
        f"Final audit: {'PASS' if final_audit.get('ok') is True else 'INCOMPLETE'}",
        "",
        "## B0 reference warning",
        "",
        "B0 is pose-derived and data-exposed; its metrics are operational reference values, "
        "not a leak-free comparison.",
        f"Checkpoint hash: {b0_reference.get('checkpoint_hash', 'missing')}",
        f"Provenance: {b0_reference.get('provenance', 'missing')}",
        "",
        "## Training artifacts",
        "",
    ]
    for stage in (
        "b1_a",
        "b1_b",
        "formal_m7",
        "formal_m0",
        "formal_m1",
        "formal_m2",
        "formal_m3",
    ):
        record = _load(artifacts / "training" / stage / "run.json")
        if record:
            lines.append(
                f"- {stage}: canonical `{record.get('canonical')}`, strict reload "
                f"{'passed' if record.get('strict_reload') else 'failed'}"
            )
        else:
            lines.append(f"- {stage}: pending")
    lines.extend(["", "## Evaluation and profiling", ""])
    for variant in EVALUATED_MODELS:
        val = _load(artifacts / "evaluation" / "val" / variant.lower() / "metrics.json")
        test = _load(artifacts / "evaluation" / "test" / variant.lower() / "metrics.json")
        profile = _load(artifacts / "profiles" / variant.lower() / "profile.json")
        lines.append(
            f"- {variant}: val mAP50-95={val.get('map50_95') if val else 'pending'}, "
            f"test mAP50-95={test.get('map50_95') if test else 'pending'}, "
            f"GFLOPs={profile.get('gflops') if profile else 'pending'}"
        )
    errors = final_audit.get("errors", [])
    lines.extend(["", "## Audit errors", ""])
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    report_path = artifacts / "report.md"
    _atomic_text(report_path, "\n".join(lines) + "\n")
    return str(report_path.resolve())
