"""Artifact-backed run reports, summaries and small reproducible figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "metrics/mAP75(B)": "mAP75",
    "metrics/mAP50-95": "mAP50_95",
    "metrics/mAP50": "mAP50",
    "metrics/mAP75": "mAP75",
    "metrics/mAPs(B)": "mAPs",
    "metrics/mAPm(B)": "mAPm",
    "metrics/mAPl(B)": "mAPl",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
}

PLAN_NAMES = {
    "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan",
    # E-series validation was completed before the plan was revised and is
    # still valid evidence for the unchanged source/architecture checks.
    "YOLO11 BinaryAttention simplified full-COCO plan",
}
E_REPORT_VARIANTS = {"E0", "E1-S", "E1", "E2-DUAL"}
FULL_REPORT_VARIANTS = {
    "T0", "T1", "T2",
    "T3", "T4", "T5",
    "T6-O", "T6-F", "T6-A", "T6-O/F", "T6-O/A", "T6-F/A", "T6",
    "T7-D", "T7-R", "T7-P", "T7-V", "T7-PV",
    "N4-FP", "N4-I8", "N4-I4", "N4-PV",
}


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _read_results(run: Path) -> tuple[dict, list[dict]]:
    candidates = [run / "training_curves.csv", run / "ultralytics" / "train" / "results.csv"]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            row = rows[-1]
            metrics = {"epochs_recorded": len(rows)}
            for source, target in METRIC_ALIASES.items():
                if source in row and row[source] not in (None, ""):
                    try:
                        metrics[target] = float(row[source])
                    except ValueError:
                        metrics[target] = row[source]
            return metrics, rows
    return {}, []


def _run_record(run: Path, root: Path) -> dict:
    resolved = _read_json(run / "resolved_config.json", {}) or {}
    status = _read_json(run / "status.json", {}) or {}
    validation = _read_json(run / "validation_metrics.json", {}) or {}
    diagnostics = _read_json(run / "attention_diagnostics.json", {}) or {}
    deltas = _read_json(run / "parameter_delta_diagnostics.json", {}) or {}
    results, _ = _read_results(run)
    record = {
        "variant": resolved.get("id", run.parent.parent.parent.name),
        "variant_label": resolved.get("id", run.parent.parent.parent.name),
        "stage": status.get("stage", run.parent.parent.name),
        "run": str(run.relative_to(root)),
        "run_id": run.name,
        "config_hash": resolved.get("config_hash"),
        "qk_scale_contract": (resolved.get("quantization_contract") or {}).get("qk_scale"),
        "p8_scale_contract": (resolved.get("quantization_contract") or {}).get("p8_scale"),
        "v8_scale_contract": (resolved.get("quantization_contract") or {}).get("v8_scale"),
        "bias_parameterization_contract": (resolved.get("quantization_contract") or {}).get("bias_parameterization"),
        "bias_initialization_contract": (resolved.get("quantization_contract") or {}).get("bias_initialization"),
        "attention_type": resolved.get("attention_type"),
        "qk_mode": resolved.get("qk_mode"),
        "p_bits": resolved.get("p_bits"),
        "v_bits": resolved.get("v_bits"),
        "magnitude_bits": resolved.get("magnitude_bits"),
        "kd_target_family": resolved.get("kd_target_family"),
        "num_binary_qk": resolved.get("num_binary_qk"),
        "num_softmax": resolved.get("num_softmax"),
        "num_pv": resolved.get("num_pv"),
        "completed": status.get("completed", False),
        "valid_for_research": status.get("valid_for_research", False),
        "engineering_gates": status.get("engineering_gates", "not executed by plan"),
        "checkpoint_weight_source": status.get("checkpoint_weight_source"),
        "metrics_weight_source": status.get("metrics_weight_source"),
        "trainable_scope": status.get("trainable_scope"),
        "trainable_parameter_count": status.get("trainable_parameter_count"),
        "frozen_parameter_count": status.get("frozen_parameter_count"),
    }
    record.update(results)
    record.update({key: value for key, value in validation.items() if key not in {"status"}})
    record.update({f"diagnostic_{key}": value for key, value in diagnostics.items() if isinstance(value, (str, int, float, bool))})
    record.update({f"delta_{key}": value for key, value in deltas.items() if isinstance(value, (str, int, float, bool))})
    return record


def _formal_runs(root: Path) -> list[Path]:
    canonical: dict[tuple[str, str], tuple[int, Path]] = {}
    for status in root.glob("artifacts/runs/*/*/*/status.json"):
        run = status.parent
        resolved = _read_json(run / "resolved_config.json", {}) or {}
        state = _read_json(status, {}) or {}
        stage = state.get("stage", run.parent.name)
        variant = str(resolved.get("id", ""))
        expected_stage = "validation" if variant in E_REPORT_VARIANTS else "full" if variant in FULL_REPORT_VARIANTS else None
        accepted_plan = (
            resolved.get("plan_name") == "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan"
            or (variant in E_REPORT_VARIANTS and resolved.get("plan_name") == "YOLO11 BinaryAttention simplified full-COCO plan")
        )
        if (
            accepted_plan
            and stage == expected_stage
            and state.get("completed") is True
            and state.get("valid_for_research") is True
        ):
            key = (variant, stage)
            candidate = (run.stat().st_mtime_ns, run)
            if key not in canonical or candidate[0] > canonical[key][0]:
                canonical[key] = candidate
    return [candidate[1] for _key, candidate in sorted(canonical.items())]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["variant"])
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, title: str, rows: list[dict], kind: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if kind == "loss":
        for run in _formal_runs(path.parents[3]):
            _, curves = _read_results(run)
            if not curves:
                continue
            x = list(range(1, len(curves) + 1))
            for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
                values = [float(item[key]) for item in curves if item.get(key, "") not in (None, "")]
                if len(values) == len(x):
                    ax.plot(x, values, label=f"{run.parent.parent.parent.name} {key.split('/')[-1]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    else:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row.get("variant", "unknown"), []).append(row)
        labels, values = [], []
        for label, items in groups.items():
            candidates = [item.get("mAP50_95") for item in items if isinstance(item.get("mAP50_95"), (int, float))]
            if candidates:
                labels.append(label)
                values.append(max(candidates))
        if values:
            ax.bar(labels, values)
            ax.tick_params(axis="x", rotation=60)
            ax.set_ylabel("AP50–95")
    if not ax.has_data():
        ax.text(0.5, 0.5, "No completed formal runs", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _subset_plot(path: Path, rows: list[dict], title: str, prefixes: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if any(str(row.get("variant", "")).startswith(prefix) for prefix in prefixes)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, values = [], []
    for row in selected:
        value = row.get("mAP50_95")
        if isinstance(value, (float, int)):
            labels.append(row.get("variant", ""))
            values.append(value)
    if values:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("AP50–95")
    else:
        ax.text(0.5, 0.5, "No completed formal runs", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_experiment_report(run: Path, variant: dict, status: dict | None = None) -> Path:
    """Write one self-contained report from the run's JSON/CSV artifacts."""

    status = status or (_read_json(run / "status.json", {}) or {})
    metrics = _read_json(run / "validation_metrics.json", {}) or {}
    diagnostics = _read_json(run / "attention_diagnostics.json", {}) or {}
    deltas = _read_json(run / "parameter_delta_diagnostics.json", {}) or {}
    report = [
        f"# {variant.get('id', 'unknown')} BinaryAttention run",
        "",
        f"- 目的：{variant.get('purpose', 'YOLO11m BinaryAttention accuracy/fidelity experiment')}。",
        f"- 相對前一 variant：`{variant.get('base_variant', 'E0')}`；本 run 改動為 `{variant.get('attention_type')}` / `{variant.get('qk_mode')}`。",
        f"- dataset：{status.get('data_manifest', 'COCO2017 full train2017/val2017')}；epochs：{status.get('epochs', 10)}；batch：{status.get('batch', 128)}；seed：{status.get('seed', 0)}。",
        f"- initialization：`{status.get('initialization', 'full-precision source checkpoint fine-tuning')}`；paper profile：`{status.get('paper_profile', 'BinaryAttention QAT fine-tuning')}`。",
        f"- checkpoint/metrics weights：`{status.get('checkpoint_weight_source', 'source checkpoint')}` / `{status.get('metrics_weight_source', 'source checkpoint')}`。",
        f"- config hash：`{variant.get('config_hash')}`。",
        "",
        "## Attention / loss",
        "",
        f"- QK：`{variant.get('qk_mode')}`；QAT：`{variant.get('use_qat')}`；bias：`{variant.get('bias_type')}`。",
        f"- quantization contract：`{json.dumps(variant.get('quantization_contract', {}), ensure_ascii=False, sort_keys=True)}`。",
        f"- KD：`{variant.get('distillation_type')}`，components=`{variant.get('kd_components', [])}`。",
        f"- T6 KD target family：`{variant.get('kd_target_family')}`；N4 non-KD parent：`{variant.get('base_variant') if str(variant.get('id', '')).startswith('N4') else 'not applicable'}`。",
        f"- 理論成本：binary QK={variant.get('num_binary_qk')}、softmax={variant.get('num_softmax')}、PV={variant.get('num_pv')}。",
        "- T3/T4/T5 對應 parallel/full-basis/matched-basis dual attention；N4 的 magnitude rank-1 term 真正加入 score。",
        "",
        "## Metrics",
        "",
    ]
    metric_keys = ("mAP50_95", "mAP50", "mAP75", "mAPs", "mAPm", "mAPl")
    for key in metric_keys:
        report.append(f"- {key}: {metrics.get(key, 'not available')}")
    report += [
        "",
        "## Diagnostics",
        "",
        f"- {json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)}",
        f"- Attention-only delta proof：{json.dumps(deltas, ensure_ascii=False, sort_keys=True)}",
        f"- G0–G5：未執行（依本計畫明確排除）。",
        "- 所有 T/N runs 從 full-precision checkpoint 直接做 10-epoch attention-only fine-tuning；非-attention parameters 凍結。",
        "- 限制：P8/V8/INT4 是 PyTorch fake quantization；不得解讀為真實 1-bit 硬體速度提升。",
        "",
        f"結論：`{status.get('conclusion', '尚無 COCO 結果；此 artifact 可供正式訓練/驗證重建。')}`",
        "",
    ]
    path = run / "experiment_report.md"
    path.write_text("\n".join(report))
    return path


def build_summary(root: Path) -> Path:
    rows = [_run_record(run, root) for run in _formal_runs(root)]
    output = root / "artifacts" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    selection = _read_json(output / "paper_qat_selection.json", {}) or {}
    summary_json = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    (output / "binary_attention_summary.json").write_text(summary_json)
    _write_csv(output / "binary_attention_summary.csv", rows)
    lines = [
        "# BinaryAttention run reports index",
        "",
        "Generated from completed 10-epoch attention-only T/N runs and E-series zero-training validation artifacts.",
        "",
        "| Variant | AP50–95 | AP50 | QK | softmax | PV | Valid | Run |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant', '')} | {row.get('mAP50_95', '')} | {row.get('mAP50', '')} | "
            f"{row.get('num_binary_qk', '')} | {row.get('num_softmax', '')} | {row.get('num_pv', '')} | "
            f"{row.get('valid_for_research', False)} | `{row.get('run', '')}` |"
        )
    index_text = "\n".join(lines) + "\n"
    (output / "run_reports_index.md").write_text(index_text)
    # Overwrite names from the previous draft so stale metrics cannot be
    # mistaken for results by downstream tooling.
    (output / "summary.json").write_text(summary_json)
    _write_csv(output / "summary.csv", rows)
    (output / "run_index.md").write_text(index_text)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot(figures / "training_loss_map.png", "Training loss / validation mAP", rows, "loss")
    _plot(figures / "variant_accuracy.png", "Formal variant AP50–95", rows, "accuracy")
    _subset_plot(figures / "t3_kd_comparison.png", rows, "T3/T4/T5 dual attention", ("T3", "T4", "T5"))
    _subset_plot(figures / "t4_comparison.png", rows, "T6 KD comparison", ("T6",))
    _subset_plot(figures / "t5_comparison.png", rows, "T7 bias/P/V comparison", ("T7",))
    _subset_plot(figures / "n_series_comparison.png", rows, "N-series accuracy and theoretical cost", ("N",))

    final = [
        "# YOLO11 BinaryAttention final report",
        "",
        "本報告由 `binary_attention_summary.json/csv` 與每個 formal run artifact 自動產生。",
        "",
        "## Scope and paper relationship",
        "",
        "本研究是 YOLO11m 的 10-epoch attention-only adaptation，不冒充論文的 300-epoch full-model ImageNet recipe。",
        "所有 T/N variants 使用完整 COCO2017、640px、effective batch 128、AdamW、lr0=5e-5、min-lr=5e-6、weight decay 0.02。",
        "Binary variants 使用 clipped-STE QAT，並從同一 FP checkpoint 獨立初始化；非-attention parameters 與其 BatchNorm 統計量凍結。",
        "論文：https://arxiv.org/abs/2603.09582；官方程式：https://github.com/EdwardChasel/BinaryAttention。",
        "",
        "## Experiment coverage",
        "",
        f"已記錄 completed research runs（validation/full）：{len(rows)}。",
        "E0/E1-S/E1/E2-DUAL 為 zero-training validation；E2-DUAL 僅使用指定 matched residual dual basis。",
        "所有 T/N variants 均只更新 attention parameters，使用 10 epochs；其餘 parameters 凍結。",
        "每個 T/N strict checkpoint 與其 COCO metrics 均使用同一個 epoch-last EMA 權重。",
        "T3/T4/T5 分別執行 parallel/full-basis/matched-basis dual attention；T6 KD 只套用在 T1–T5 中 mAP 最佳的分支。",
        "T7 執行 dense/decomposed relative-position bias 與 P/V fake quantization；N4 從 T7+T1–T5 選 parent，但 N4 本身不使用 KD。",
        "Scaled Q/K 契約為 mean_abs_channel_token_per_sample_head；P8 使用 static_unsigned_1_over_255；V8 為逐 channel 跨 token scale。",
        "T7-PV 使用選定 T7 bias 與 P8/V8 fake quantization；N4-I8/I4 分別測試 magnitude 8/4-bit。",
        "",
        "## Selection and limitations",
        "",
        f"10-epoch final-family best variant：{selection.get('selected_variant', 'pending')}；AP50–95：{selection.get('selected_mAP50_95', 'pending')}。",
        "所有 P8/V8/INT8/INT4 結果均為 fake quantization；本報告不宣稱真實 1-bit 硬體速度提升。",
        "",
        "詳見 `binary_attention_summary.csv`、`run_reports_index.md` 與 `figures/`。",
        "",
    ]
    (output / "binary_attention_final_report.md").write_text("\n".join(final))
    return output
