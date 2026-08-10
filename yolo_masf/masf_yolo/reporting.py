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


def _display(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _class_lines(metrics: dict[str, Any], class_name: str, label: str) -> list[str]:
    per_class = metrics.get("per_class", {}).get(class_name, {})
    diagnostics = metrics.get("class_diagnostics", {}).get(class_name, {})
    lines = [
        f"  - {label}: AP50={_display(per_class.get('ap50'))}, "
        f"AP50-95={_display(per_class.get('ap'))}, "
        f"precision={_display(diagnostics.get('precision'))}, "
        f"recall={_display(diagnostics.get('recall'))}, "
        f"GT={_display(diagnostics.get('gt_count'))}, "
        f"predictions={_display(diagnostics.get('prediction_count'))}, "
        f"missed={_display(diagnostics.get('missed_count'))}, "
        f"false positives={_display(diagnostics.get('false_positive_count'))}"
    ]
    subsets = diagnostics.get("subsets", {})
    lines.append(
        f"    - size/blur recall: "
        f"tiny={_display(subsets.get('tiny', {}).get('recall'))} "
        f"(n={_display(subsets.get('tiny', {}).get('gt_count'))}), "
        f"small={_display(subsets.get('small', {}).get('recall'))} "
        f"(n={_display(subsets.get('small', {}).get('gt_count'))}), "
        f"large={_display(subsets.get('large', {}).get('recall'))} "
        f"(n={_display(subsets.get('large', {}).get('gt_count'))}), "
        f"blur={_display(subsets.get('blur_proxy', {}).get('recall'))} "
        f"(n={_display(subsets.get('blur_proxy', {}).get('gt_count'))})"
    )
    return lines


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
        "## Data exposure warning",
        "",
        "B0, B1, and every MFAM variant use a pose-derived initializer that has already "
        "seen BBT5. All metrics are data-exposed operational ablations, not leak-free "
        "generalization estimates.",
        f"Checkpoint hash: {b0_reference.get('checkpoint_hash', 'missing')}",
        f"Provenance: {b0_reference.get('provenance', 'missing')}",
        "",
        "## Training budget warning",
        "",
        "SP2P 使用序列式訓練預算：先繼承已完成 10+90 epochs 的 SP2-B 與 "
        "BEST_PARTIAL，再執行自己的 10+90 epochs。其結果不可直接視為與僅從 "
        "B1-B 訓練 100 epochs 的單段架構公平消融。",
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
        "formal_p3m",
        "sp2_a",
        "sp2_b",
        "sp2p_a",
        "sp2p_b",
    ):
        record = _load(artifacts / "training" / stage / "run.json")
        if record:
            lines.append(
                f"- {stage}: canonical `{record.get('canonical')}`, strict reload "
                f"{'passed' if record.get('strict_reload') else 'failed'}"
            )
            if stage.startswith("sp2p"):
                lines.append(
                    f"  - architecture={record.get('architecture_variant', 'missing')}, "
                    f"BEST_PARTIAL={record.get('selected_partial', 'missing')}, "
                    f"parents={record.get('parent_hashes', {})}"
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
        lines.extend(["", f"### {variant}", ""])
        for split_label, metrics in (("Validation", val), ("Test", test)):
            if metrics is None:
                lines.append(f"- {split_label}: pending")
                continue
            lines.append(
                f"- {split_label} overall: mAP50-95={_display(metrics.get('map50_95'))}"
            )
            lines.extend(_class_lines(metrics, "ball", "Ball"))
            lines.extend(_class_lines(metrics, "bat", "Bat"))
    errors = final_audit.get("errors", [])
    lines.extend(["", "## Audit errors", ""])
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    report_path = artifacts / "report.md"
    _atomic_text(report_path, "\n".join(lines) + "\n")
    return str(report_path.resolve())
