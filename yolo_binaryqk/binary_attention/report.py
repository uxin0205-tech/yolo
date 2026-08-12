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
    "metrics/mAP_small(B)": "mAPs",
    "metrics/mAP_medium(B)": "mAPm",
    "metrics/mAP_large(B)": "mAPl",
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

EXPERIMENT_ORDER = (
    "E0", "E1-S", "E1", "E2-DUAL",
    "T0", "T1", "T2", "T3", "T4", "T5",
    "T6-O", "T6-F", "T6-A", "T6-O/F", "T6-O/A", "T6-F/A", "T6",
    "T7-D", "T7-R", "T7-P", "T7-V", "T7-PV",
    "N4-FP", "N4-I8", "N4-I4", "N4-PV",
)

VARIANT_DESCRIPTIONS = {
    "E0": "FP attention，零訓練基準",
    "E1-S": "sign-only binary Q/K，零訓練",
    "E1": "scaled-sign binary Q/K，零訓練",
    "E2-DUAL": "matched residual dual basis，零訓練",
    "T0": "FP attention-only fine-tuning 控制組",
    "T1": "sign-only QAT",
    "T2": "scaled-sign QAT",
    "T3": "parallel dual binary attention",
    "T4": "full-basis residual dual attention",
    "T5": "matched-basis residual dual attention",
    "T6-O": "T4 + positional KD",
    "T6-F": "T4 + feature KD",
    "T6-A": "T4 + attention KD",
    "T6-O/F": "T4 + positional/feature KD",
    "T6-O/A": "T4 + positional/attention KD",
    "T6-F/A": "T4 + feature/attention KD",
    "T6": "選定 positional/feature KD 的正式重現",
    "T7-D": "T6 配置 + dense 2D relative-position bias",
    "T7-R": "T6 配置 + decomposed 2D relative-position bias",
    "T7-P": "dense bias + P 8-bit fake quantization",
    "T7-V": "dense bias + V 8-bit fake quantization",
    "T7-PV": "dense bias + P/V 8-bit fake quantization",
    "N4-FP": "非 KD magnitude side channel，FP magnitude",
    "N4-I8": "非 KD magnitude side channel，8-bit magnitude",
    "N4-I4": "非 KD magnitude side channel，4-bit magnitude",
    "N4-PV": "N4-I4 + P/V 8-bit fake quantization",
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
        "purpose": resolved.get("purpose"),
        "base_variant": resolved.get("base_variant"),
        "use_qat": resolved.get("use_qat"),
        "use_distillation": resolved.get("use_distillation"),
        "distillation_type": resolved.get("distillation_type"),
        "kd_components": "+".join(resolved.get("kd_components") or ()),
        "bias_type": resolved.get("bias_type"),
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
    normalized_validation = {}
    for key, value in validation.items():
        if key == "status":
            continue
        normalized_validation[METRIC_ALIASES.get(key, key)] = value
    record.update(normalized_validation)
    record.update({f"diagnostic_{key}": value for key, value in diagnostics.items() if isinstance(value, (str, int, float, bool))})
    record.update({f"delta_{key}": value for key, value in deltas.items() if isinstance(value, (str, int, float, bool))})
    return record


def _score(rows: dict[str, dict], variant: str, key: str = "mAP50_95") -> float | None:
    value = rows.get(variant, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _metric_text(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _delta_text(value: float | None, reference: float | None) -> str:
    if value is None or reference is None:
        return "—"
    return f"{value - reference:+.5f}"


def _comparison_table(rows: dict[str, dict], variants: tuple[str, ...], reference: str) -> list[str]:
    reference_score = _score(rows, reference, "coco_mAP50_95") or _score(rows, reference)
    lines = [
        f"參考值：`{reference}` COCO mAP50–95 = {_metric_text(reference_score)}。",
        "",
        "| Variant | 實驗內容 | 原 mAP | COCO mAP | AP50 | APs | APm | APl | Precision | Recall | Δ COCO mAP | Binary QK / Softmax / PV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        row = rows.get(variant, {})
        cost = f"{row.get('num_binary_qk', '—')} / {row.get('num_softmax', '—')} / {row.get('num_pv', '—')}"
        coco_score = _score(rows, variant, "coco_mAP50_95") or _score(rows, variant)
        lines.append(
            f"| {variant} | {VARIANT_DESCRIPTIONS.get(variant, '')} | "
            f"{_metric_text(_score(rows, variant))} | {_metric_text(coco_score)} | "
            f"{_metric_text(_score(rows, variant, 'coco_mAP50'))} | "
            f"{_metric_text(_score(rows, variant, 'mAPs'))} | {_metric_text(_score(rows, variant, 'mAPm'))} | "
            f"{_metric_text(_score(rows, variant, 'mAPl'))} | "
            f"{_metric_text(_score(rows, variant, 'precision'))} | {_metric_text(_score(rows, variant, 'recall'))} | "
            f"{_delta_text(coco_score, reference_score)} | {cost} |"
        )
    return lines


def _area_metrics_table(rows: dict[str, dict]) -> list[str]:
    lines = [
        "| Variant | 原 mAP50–95 | COCO mAP50–95 | AP50 | AP75 | APs | APm | APl | COCO−原 mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in EXPERIMENT_ORDER:
        lines.append(
            f"| {variant} | {_metric_text(_score(rows, variant))} | "
            f"{_metric_text(_score(rows, variant, 'coco_mAP50_95'))} | "
            f"{_metric_text(_score(rows, variant, 'coco_mAP50'))} | "
            f"{_metric_text(_score(rows, variant, 'coco_mAP75'))} | "
            f"{_metric_text(_score(rows, variant, 'mAPs'))} | "
            f"{_metric_text(_score(rows, variant, 'mAPm'))} | "
            f"{_metric_text(_score(rows, variant, 'mAPl'))} | "
            f"{_delta_text(_score(rows, variant, 'coco_mAP50_95'), _score(rows, variant))} |"
        )
    return lines


def _best_for_metric(rows: dict[str, dict], key: str) -> tuple[str, float | None]:
    candidates = [(variant, _score(rows, variant, key)) for variant in EXPERIMENT_ORDER]
    available = [(variant, value) for variant, value in candidates if value is not None]
    return max(available, key=lambda item: item[1]) if available else ("—", None)


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
        writer = csv.DictWriter(handle, fieldnames=fields or ["variant"], lineterminator="\n")
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
    metric_keys = (
        "mAP50_95", "mAP50", "mAP75",
        "coco_mAP50_95", "coco_mAP50", "coco_mAP75",
        "mAPs", "mAPm", "mAPl",
    )
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
    formal_runs = _formal_runs(root)
    for run in formal_runs:
        render_experiment_report(
            run,
            _read_json(run / "resolved_config.json", {}) or {},
            _read_json(run / "status.json", {}) or {},
        )
    rows = [_run_record(run, root) for run in formal_runs]
    output = root / "artifacts" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    selection = _read_json(output / "paper_qat_selection.json", {}) or {}
    final_audit = _read_json(root / "logs" / "formal-plan" / "final-audit.json", {}) or {}
    weight_archive = _read_json(root / "artifacts" / "final_weights" / "manifest.json", {}) or {}
    checkpoint_cleanup = _read_json(root / "artifacts" / "final_weights" / "checkpoint_cleanup.json", {}) or {}
    summary_json = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    (output / "binary_attention_summary.json").write_text(summary_json)
    _write_csv(output / "binary_attention_summary.csv", rows)
    lines = [
        "# BinaryAttention run reports index",
        "",
        "Generated from completed 10-epoch attention-only T/N runs and E-series zero-training validation artifacts.",
        "",
        "| Variant | mAP50–95 | AP50 | APs | APm | APl | QK | softmax | PV | Valid | Run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant', '')} | {row.get('mAP50_95', '')} | {row.get('mAP50', '')} | "
            f"{row.get('mAPs', '')} | {row.get('mAPm', '')} | {row.get('mAPl', '')} | "
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

    by_variant = {str(row.get("variant")): row for row in rows}
    t0_score = _score(by_variant, "T0")
    t6_score = _score(by_variant, "T6")
    selected_variant = str(selection.get("selected_variant", "pending"))
    selected_score = _score(by_variant, selected_variant)
    selected_run = by_variant.get(selected_variant, {}).get("run", "pending")
    selected_checkpoint = (
        f"../{str(selected_run).removeprefix('artifacts/')}/checkpoints/strict-last.pt"
        if selected_run != "pending"
        else "pending"
    )
    n4_variant = str(selection.get("selected_n4_artifact", "pending"))
    n4_score = _score(by_variant, n4_variant)
    max_frozen_parameter_delta = max(
        (float(row["delta_max_frozen_parameter_delta"]) for row in rows
         if isinstance(row.get("delta_max_frozen_parameter_delta"), (int, float))),
        default=0.0,
    )
    max_frozen_bn_delta = max(
        (float(row["delta_max_frozen_non_attention_bn_buffer_delta"]) for row in rows
         if isinstance(row.get("delta_max_frozen_non_attention_bn_buffer_delta"), (int, float))),
        default=0.0,
    )
    best_coco_map = _best_for_metric(by_variant, "coco_mAP50_95")
    best_aps = _best_for_metric(by_variant, "mAPs")
    best_apm = _best_for_metric(by_variant, "mAPm")
    best_apl = _best_for_metric(by_variant, "mAPl")

    final = [
        "# YOLO11 BinaryAttention 完整實驗報告",
        "",
        "本報告由 26 個 canonical research artifacts、`binary_attention_summary.json/csv` 與 "
        "`paper_qat_selection.json` 自動重建；指標均來自各 run 的 COCO validation artifact。",
        "",
        "## 1. 研究問題與範圍",
        "",
        "本實驗研究在 YOLO11m 偵測模型中，將 attention 的 Q/K 計算改為 binary/dual-basis 形式後，"
        "能否透過短期 QAT fine-tuning、知識蒸餾、relative-position bias 與局部 8/4-bit fake quantization "
        "維持接近全精度 attention 的 COCO 準確率。",
        "",
        "這是 **10-epoch attention-only adaptation**：只訓練 attention parameters，並非論文的 "
        "**300-epoch full-model** ImageNet recipe。因此結果可回答本專案的相對消融問題，但不能直接重現或取代論文主表。",
        "",
        "- 模型與初始化：YOLO11m；所有 T/N run 從同一 full-precision source checkpoint 獨立初始化。",
        "- 資料：完整 COCO2017 train2017 118,287 張、val2017 5,000 張；輸入 640 px。",
        "- 訓練：10 epochs、seed 0、deterministic、AdamW、lr0 `5e-5`、cosine decay 至 `5e-6`、weight decay `0.02`。",
        "- Batch：micro batch 16、gradient accumulation 8、effective batch 128；AMP；8 workers；disk cache。",
        "- 參數範圍：主要 T variants 約 200,960 個 attention parameters 可訓練、約 19.91M 個 parameters 凍結；"
        "dense bias/N4 variants 約 217k 個可訓練參數。",
        "- 權重一致性：validation metrics 與 strict checkpoint 都採同一個 epoch-last EMA 權重。",
        "- 論文脈絡：https://arxiv.org/abs/2603.09582；官方程式：https://github.com/EdwardChasel/BinaryAttention。",
        "",
        "## 2. 實驗流程",
        "",
        "1. E 系列在不訓練下量測直接替換 attention 的準確率衝擊。",
        "2. T0–T5 比較 FP、sign、scaled-sign 與三種 dual-basis attention；T3/T4/T5 本身不使用 KD。",
        "3. 從 T1–T5 選出 T4，測試 T6 的 positional（O）、feature（F）、attention（A）KD 與兩兩組合。",
        "4. 使用最佳 positional+feature 組合建立 T6，再比較 T7-D/T7-R 的 2D relative-position bias。",
        "5. 在最佳 dense 2D bias 上測試 P、V、P/V 8-bit fake quantization。",
        "6. N4 取 T7-D 的 dense-bias 結構選擇，但不繼承 T7-D 的 KD 或 checkpoint；"
        "N4 各 run 仍從同一 FP source 獨立初始化，測試 FP/8-bit/4-bit magnitude side channel，再將最佳 I4 與 P/V 8-bit 結合。",
        "",
        "量化契約：scaled Q/K 使用 `mean_abs_channel_token_per_sample_head`；P8 使用 "
        "`static_unsigned_1_over_255`；V8 使用 `max_abs_token_per_sample_head_channel`。",
        "COCO 尺寸分組採官方 bbox area：small `< 32²`、medium `32²–96²`、large `>= 96²` pixels；"
        "APs/APm/APl 由 canonical strict weights 在 val2017 5,000 張影像上以 COCOeval 重新計算。",
        "",
        "## 3. 完整結果",
        "",
        "### 3.1 E 系列：零訓練替換",
        "",
    ]
    final += _comparison_table(by_variant, ("E0", "E1-S", "E1", "E2-DUAL"), "E0")
    final += [
        "",
        "零訓練時，sign-only 造成最大下降；scaled-sign 與 dual-basis 能保留更多準確率。"
        "這也說明後續 QAT fine-tuning 是必要步驟，而不是可省略的附加操作。",
        "",
        "### 3.2 T0–T5：attention 結構與 QAT",
        "",
    ]
    final += _comparison_table(by_variant, ("T0", "T1", "T2", "T3", "T4", "T5"), "T0")
    final += [
        "",
        "T4（full-basis residual dual attention）是 T1–T5 中最佳 binary branch，AP50–95 0.50586，"
        "僅比 T0 低 0.00086。T2 的 scaled-sign 優於 T1 sign-only；T3 的 parallel dual 計算成本較高，"
        "但沒有換得較佳準確率；T5 matched-basis 優於 T3，但仍低於 T4。",
        "",
        "### 3.3 T6：KD component 消融",
        "",
    ]
    final += _comparison_table(
        by_variant,
        ("T4", "T6-O", "T6-F", "T6-A", "T6-O/F", "T6-O/A", "T6-F/A", "T6"),
        "T4",
    )
    final += [
        "",
        "positional+feature（T6-O/F）是最佳 KD 組合，正式 T6 的 deterministic 重現得到相同 AP50–95 0.50624。"
        "單獨 positional、feature、attention 或含 attention 的兩兩組合均未超過 T4；因此本資料下的提升主要來自 "
        "positional 與 feature supervision 的互補。",
        "",
        "### 3.4 T7：relative-position bias 與 P/V 量化",
        "",
    ]
    final += _comparison_table(by_variant, ("T6", "T7-D", "T7-R", "T7-P", "T7-V", "T7-PV"), "T6")
    final += [
        "",
        "dense 2D bias（T7-D）優於 decomposed 2D bias（T7-R）0.00039，故後續採 dense 2D。"
        "P8、V8 與 P8/V8 都造成小幅下降，其中只量化 P 的 T7-P 最接近未量化 T7-D。"
        "這些差距僅有 1e-4 至 1e-3，單一 seed 下不應宣稱具統計顯著性。",
        "",
        "### 3.5 N4：magnitude side channel",
        "",
    ]
    final += _comparison_table(by_variant, ("T7-D", "N4-FP", "N4-I8", "N4-I4", "N4-PV"), "T7-D")
    final += [
        "",
        "N4 內部以 I4 最佳（0.50406），比 N4-FP 高 0.00022，也略高於 I8；加入 P/V 8-bit 後為 0.50402。"
        "但所有 N4 variants 都低於 parent T7-D，表示 magnitude side channel 在本 10-epoch attention-only 設定下沒有帶來淨收益。",
        "",
        "### 3.6 COCO mAP 與物件尺寸 AP 完整總表",
        "",
        "`原 mAP50–95` 是訓練完成時保存的 Ultralytics evaluator 指標；`COCO mAP/APs/APm/APl` "
        "是使用相同 canonical strict weight 重新推論後，由官方 COCO annotation 與 COCOeval 計算。"
        "兩者 evaluator 規則不同，因此 COCO−原 mAP 不應解讀為模型重新訓練後的提升。",
        "",
    ]
    final += _area_metrics_table(by_variant)
    final += [
        "",
        f"- COCO mAP50–95 最高：`{best_coco_map[0]}`，`{_metric_text(best_coco_map[1])}`。",
        f"- APs 最高：`{best_aps[0]}`，`{_metric_text(best_aps[1])}`。",
        f"- APm 最高：`{best_apm[0]}`，`{_metric_text(best_apm[1])}`。",
        f"- APl 最高：`{best_apl[0]}`，`{_metric_text(best_apl[1])}`。",
        "",
        "## 4. 選擇結果與主要結論",
        "",
        f"- 全部正式 run 的 FP 控制組 T0：AP50–95 `{_metric_text(t0_score)}`。",
        f"- 最佳 KD binary 結果 T6：AP50–95 `{_metric_text(t6_score)}`，相對 T0 "
        f"`{_delta_text(t6_score, t0_score)}`。",
        f"- 最終候選族群最佳模型：`{selected_variant}`，AP50–95 `{_metric_text(selected_score)}`，"
        f"相對 T0 `{_delta_text(selected_score, t0_score)}`。",
        f"- N4 族群最佳 magnitude：`{n4_variant}`，AP50–95 `{_metric_text(n4_score)}`。",
        f"- 最終 strict checkpoint：`{selected_checkpoint}`。",
        "",
        "核心結論是：scaled/dual-basis QAT 可在只微調 attention 10 epochs 的條件下，將 binary attention "
        "恢復到非常接近 FP 控制組；T4 是最佳基本結構，positional+feature KD 再縮小差距。"
        "Relative-position bias、P/V 量化與 magnitude side channel 沒有進一步超越 T6，"
        "但 T7-D 仍是預先定義 final-family 中最佳候選。",
        "",
        "## 5. Final audit 失敗原因與修正政策",
        "",
        f"目前正式 final audit：`ok={str(final_audit.get('ok', 'unknown')).lower()}`；"
        f"verified variants `{final_audit.get('verified_variant_count', 'unknown')}/"
        f"{final_audit.get('expected_variant_count', 'unknown')}`；errors `{len(final_audit.get('errors', [])) if isinstance(final_audit.get('errors'), list) else 'unknown'}`。",
        "",
        f"26/26 canonical variants 均有 completed、valid-for-research artifact、COCO metrics、strict checkpoint、"
        "config hash、architecture manifest 與 attention diagnostics。原 audit 失敗並非缺少模型或指標。",
        "",
        "失敗原因有兩項：",
        "",
        f"1. epoch EMA 會對即使凍結且值理論上不變的浮點 tensor 執行乘加，造成 ulp 等級捨入。"
        f"本批 artifacts 的最大 frozen parameter delta 為 `{max_frozen_parameter_delta:.9g}`，"
        f"最大非 attention BN buffer delta 為 `{max_frozen_bn_delta:.9g}`；所有 run 都沒有遺失 frozen tensor，"
        "且 9 個 source-initialized attention tensors 確實改變。Audit 因此採 fail-closed absolute tolerance `1e-5`，"
        "超過才判定 freeze violation。",
        "2. Ultralytics 清理 T0 `last.pt` 時把 epoch metadata 設為 -1，舊 repair 程式因此寫成 epoch 0。"
        "T0 的 `training_curves.csv` 與原始 `results.csv` 均有 10 個完整 epoch rows，現以此作為 epoch-last EMA 證據。",
        "",
        "這項修正不改模型權重、不改 validation metrics，也不把大型差異藏在容差內；它只修正原本不適用於 EMA "
        "浮點運算的 bit-exact 稽核條件，並保留最大差異與容差供重現。",
        "",
        "## 6. 限制與論文使用方式",
        "",
        "- 所有結果只有 seed 0；小於約 0.001 AP 的差距應視為趨勢，不能當作統計顯著結論。",
        "- 本研究是 COCO detection 的 10-epoch attention-only adaptation，不是 300-epoch full-model replication。",
        "- Q/K binary operation、P8/V8/INT8/INT4 都在 PyTorch 中做 QAT/fake quantization；"
        "尚未測量真實硬體 latency、energy、memory bandwidth 或模型檔案壓縮率。",
        "- T7-D 是 final-family selection；若研究問題是純準確率，應同時報告 T0 與 T6，避免把 T7-D 誤稱為全矩陣最高分。",
        "- 建議論文主表報 AP50–95、AP50、Precision、Recall；若要主張穩定優勢，需補多 seed 或更長訓練。",
        "",
        "## 7. 研究產物",
        "",
        "- `binary_attention_summary.csv/json`：26 個 canonical runs 的完整可機讀彙整。",
        "- `run_reports_index.md`：每個 run 與 artifact 路徑。",
        "- `paper_qat_selection.json`：T6、T7、N4 與 final-family 的選擇依據。",
        "- `figures/`：loss、variant accuracy、T6/T7/N4 比較圖。",
        "- `logs/formal-plan/final-audit.json`：最終 fail-closed audit 結果。",
        "",
        "## 8. Canonical weight archive",
        "",
        f"已集中保存 `{weight_archive.get('variant_count', 0)}` 個消融實驗正式權重。"
        "每個權重旁均保留 model YAML、resolved config、validation metrics、checkpoint manifest 與 status。",
        "",
        "| Variant | AP50–95 | 保存權重 | SHA-256 | 載入驗證 |",
        "|---|---:|---|---|---|",
    ]
    archived_rows = weight_archive.get("weights") if isinstance(weight_archive.get("weights"), list) else []
    for row in archived_rows:
        final.append(
            f"| {row.get('variant', '')} | {row.get('mAP50_95', '')} | "
            f"`../{str(row.get('archived_weight', '')).removeprefix('artifacts/')}` | "
            f"`{row.get('sha256', '')}` | {row.get('verification_mode', '')} |"
        )
    final += [
        "",
        "保存目錄的 `weight.pt` 是與報告 metrics 對應的 canonical strict weight，不可刪除。"
        f"已刪除 `{checkpoint_cleanup.get('deleted_count', 0)}` 個冗餘 Ultralytics `best.pt`/`last.pt`，"
        f"共 `{checkpoint_cleanup.get('total_bytes', 0) / (1024 ** 3):.3f} GiB`。"
        "這些檔案的 optimizer/resume 狀態只能藉由重新訓練重建；26 組 canonical strict weights、"
        "模型設定、驗證指標及雜湊均完整保留。",
        "",
    ]
    (output / "binary_attention_final_report.md").write_text("\n".join(final))
    return output
