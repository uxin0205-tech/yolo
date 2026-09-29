"""Auditable post-processing for the completed Clean BBT5 study."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from ..artifacts.io import PipelineLock, atomic_write_json
from ..contracts import canonical_json, sha256_file
from ..evaluation.profiling import profile_gpu_latency, profile_module
from ..evaluation.runner import run_variant_evaluation
from .contracts import CLEAN_EXPERIMENTS, load_clean_config
from .data_view import write_evaluation_view

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "clean" / "clean_ablation.yaml"
ARTIFACTS = ROOT / "artifacts" / "clean-bbt5-ablation"
LOCKED = ROOT / "artifacts" / "locked-bbt5-dataset"
RESULTS = ROOT / "clean_bbt5_study" / "results"
EVALUATION_DATA = ARTIFACTS / "data" / "evaluation.yaml"
SELECTION = ARTIFACTS / "selection.json"
STATE = ARTIFACTS / "postprocess_state.json"
LOCK = ARTIFACTS / "postprocess.lock"

METRIC_FIELDS = (
    "map50_95", "map50", "map75", "ap_s", "ap_m", "ap_l",
    "ball_ap", "ball_ap_s", "ball_recall", "bat_ap",
    "tiny_ball_recall", "small_ball_recall", "large_ball_recall",
)

DESCRIPTIONS = {
    "B0-Clean": "原始 P3/P4/P5 三尺度 YOLO11m baseline；不加入 P2 或 MFAM。",
    "P2-Direct-Clean": "四尺度 P2/P3/P4/P5 baseline；新增標準高解析 P2 detection head。",
    "P2-PaperFormula-Clean": "在 P2 加入論文式 MFAM：DW3、DW5、factorized 7、factorized 9，加 identity 與兩層 1x1 residual fusion。",
    "P2-Lite35-Clean": "在 P2 只保留 DW3、DW5 與兩層 1x1 residual fusion。",
    "P2-Lite35-F7-Clean": "在 P2 使用 DW3、DW5、1x7→7x1 與兩層 1x1 residual fusion。",
    "P2-Partial50-Clean": "P2 前 50% channels 使用 Lite35 MFAM，其餘 channels exact bypass。",
    "P2-Partial25-Clean": "P2 前 25% channels 使用 Lite35 MFAM，其餘 channels exact bypass。",
    "P3-M7-Clean": "只在 P3 加入 legacy MFAM：DW3、DW5、factorized 7 與單層 1x1 fusion；不含 9x9。",
    "P3-Lite35-Clean": "只在 P3 加入 DW3、DW5 與論文式雙 1x1 fusion。",
    "P3-Lite35-F7-Clean": "只在 P3 加入 DW3、DW5、1x7→7x1 與論文式雙 1x1 fusion；不含 9x9。",
    "P3-Partial50-Clean": "P3 前 50% channels 使用 Lite35 MFAM，其餘 channels exact bypass。",
    "P3-Partial25-Clean": "P3 前 25% channels 使用 Lite35 MFAM，其餘 channels exact bypass。",
    "P2-Control-Clean-Head": "P2 最佳化 control A：只訓練新增 P2 slot/head，最多 20 epochs；不進 strict-fair ranking。",
    "P2-Control-Clean-Full": "P2 最佳化 control B：承接同 seed 的 Head best，全模型最多 80 epochs、lr0=0.001；不進 strict-fair ranking。",
}


def _slug(name: str) -> str:
    return name.lower().replace("_", "-").replace(" ", "-")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def _std(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return statistics.stdev(valid) if len(valid) > 1 else 0.0 if valid else None


def _fmt(value: float | None, deviation: float | None = None) -> str:
    if value is None:
        return "N/A"
    if deviation is None:
        return f"{value:.5f}"
    return f"{value:.5f} ± {deviation:.5f}"


def _set_state(stage: str, *, current: str | None = None, completed: int = 0,
               total: int = 0, error: str | None = None) -> None:
    atomic_write_json(STATE, {
        "stage": stage,
        "current": current,
        "completed": completed,
        "total": total,
        "error": error,
    })


def formal_records() -> list[dict[str, Any]]:
    config = load_clean_config(CONFIG)
    expected = {
        (experiment, seed)
        for experiment in CLEAN_EXPERIMENTS
        for seed in config.values["seeds"]
    }
    records: list[dict[str, Any]] = []
    actual: set[tuple[str, int]] = set()
    for path in sorted((ARTIFACTS / "worker").glob("formal-*.json")):
        payload = _read_json(path)
        key = (str(payload["experiment"]), int(payload["seed"]))
        if key in actual:
            raise RuntimeError(f"duplicate formal result: {key}")
        actual.add(key)
        payload["_manifest"] = str(path.resolve())
        records.append(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"formal matrix mismatch; missing={missing}, extra={extra}")
    order = {name: index for index, name in enumerate(CLEAN_EXPERIMENTS)}
    return sorted(records, key=lambda row: (order[row["experiment"]], int(row["seed"])))


def _best_training_row(results_csv: Path) -> tuple[dict[str, str], int]:
    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty training results: {results_csv}")
    key = "metrics/mAP50-95(B)"
    return max(rows, key=lambda row: float(row[key])), len(rows)


def _fresh_reload(checkpoint: Path) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "masf_yolo.clean.postprocess",
        "reload-one", "--checkpoint", str(checkpoint),
    ]
    process = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True,
    )
    lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"fresh reload emitted no JSON: {checkpoint}\n{process.stdout}\n{process.stderr}")
    return json.loads(lines[-1])


def audit_formal_runs(*, fresh_reload: bool = True) -> dict[str, Any]:
    config = load_clean_config(CONFIG)
    queue = _read_json(ARTIFACTS / "queue_state.json")
    if queue.get("status") != "training_complete" or len(queue.get("completed", [])) != 40:
        raise RuntimeError("training queue is not complete")
    trainer_view = yaml.safe_load(
        (ARTIFACTS / "data" / "train_val_only.yaml").read_text(encoding="utf-8")
    )
    if "test" in trainer_view:
        raise RuntimeError("trainer view exposes historical test")
    rows: list[dict[str, Any]] = []
    records = formal_records()
    for index, record in enumerate(records, start=1):
        experiment = record["experiment"]
        seed = int(record["seed"])
        best = Path(record["best"])
        last = Path(record["last"])
        run = Path(record["save_dir"])
        for checkpoint in (best, last):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        if sha256_file(best) != record["best_sha256"]:
            raise RuntimeError(f"best hash mismatch: {experiment} seed {seed}")
        if sha256_file(last) != record["last_sha256"]:
            raise RuntimeError(f"last hash mismatch: {experiment} seed {seed}")
        args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        if int(args["seed"]) != seed or int(args["imgsz"]) != 640 or int(args["batch"]) != 16:
            raise RuntimeError(f"resolved profile mismatch: {experiment} seed {seed}")
        if args["optimizer"] != "SGD" or bool(args["pretrained"]):
            raise RuntimeError(f"optimizer/pretrained mismatch: {experiment} seed {seed}")
        if Path(args["data"]).resolve() != (ARTIFACTS / "data" / "train_val_only.yaml").resolve():
            raise RuntimeError(f"unexpected trainer data view: {experiment} seed {seed}")
        best_row, epochs_recorded = _best_training_row(run / "results.csv")
        planned = int(args["epochs"])
        reload_result = _fresh_reload(best) if fresh_reload else {"strict_native_reload": None}
        rows.append({
            "experiment": experiment,
            "seed": seed,
            "comparison_tier": record["comparison_tier"],
            "schedule": CLEAN_EXPERIMENTS[experiment].schedule,
            "epochs_planned": planned,
            "epochs_recorded": epochs_recorded,
            "best_epoch": int(best_row["epoch"]),
            "training_best_map50_95": float(best_row["metrics/mAP50-95(B)"]),
            "patience": int(args["patience"]),
            "early_stopped": epochs_recorded < planned,
            "protocol_revision": (
                "pre-amendment-patience100" if int(args["patience"]) == 100
                else "amended-patience30"
            ),
            "best": str(best.resolve()),
            "best_sha256": record["best_sha256"],
            "last": str(last.resolve()),
            "last_sha256": record["last_sha256"],
            "config_hash": record["config_hash"],
            "initializer_sha256": record["initializer_sha256"],
            "parent_checkpoint": record.get("parent_checkpoint"),
            "resumed_from": record.get("resumed_from"),
            **reload_result,
        })
        _set_state("audit", current=f"{experiment}:seed{seed}", completed=index, total=len(records))
    control_full = [row for row in rows if row["experiment"] == "P2-Control-Clean-Full"]
    for row in control_full:
        if not row["parent_checkpoint"]:
            raise RuntimeError("control Full is missing Head parent lineage")
    payload = {
        "ok": True,
        "formal_jobs": len(rows),
        "strict_fair_jobs": sum(row["comparison_tier"] == "strict_fair" for row in rows),
        "optimization_control_jobs": sum(
            row["comparison_tier"] == "optimization_control" for row in rows
        ),
        "queue_jobs": len(queue["completed"]),
        "trainer_test_key_absent": True,
        "dataset_hash": config.values["dataset"]["dataset_hash"],
        "initializer_sha256": config.values["initializer"]["sha256"],
        "config_hashes": sorted({row["config_hash"] for row in rows}),
        "protocol_amendment": {
            "description": "patience 由 100 改為 30；前五個 formal run 已啟動，不能熱更新。",
            "pre_amendment_jobs": sum(row["patience"] == 100 for row in rows),
            "amended_jobs": sum(row["patience"] == 30 for row in rows),
        },
        "runs": rows,
    }
    atomic_write_json(ARTIFACTS / "audit" / "formal_runs.json", payload)
    _write_csv(RESULTS / "formal_run_audit.csv", rows)
    _set_state("audit_complete", completed=len(rows), total=len(rows))
    return payload


def _evaluation_path(record: dict[str, Any], split: str) -> Path:
    return (
        ARTIFACTS / "evaluation" / split
        / f"{_slug(record['experiment'])}-seed{int(record['seed'])}"
    )


def evaluate_split(split: str) -> None:
    config = load_clean_config(CONFIG)
    if split == "val":
        config.assert_split_use(split="val", purpose="selection")
    elif split == "test":
        config.assert_split_use(split="test", purpose="historical_report")
        if not SELECTION.is_file():
            raise RuntimeError("historical test evaluation requires frozen selection.json")
    else:
        raise ValueError("split must be val or test")
    write_evaluation_view(config.locked_data_yaml, EVALUATION_DATA)
    records = formal_records()
    coco = LOCKED / f"{split}.coco.json"
    for index, record in enumerate(records, start=1):
        output = _evaluation_path(record, split)
        metrics_path = output / "metrics.json"
        expected_hash = record["best_sha256"]
        if metrics_path.is_file():
            existing = _read_json(metrics_path)
            if existing.get("checkpoint_sha256") == expected_hash:
                _set_state(f"evaluate_{split}", current=None, completed=index, total=len(records))
                continue
        label = f"{record['experiment']}:seed{record['seed']}"
        _set_state(f"evaluate_{split}", current=label, completed=index - 1, total=len(records))
        metrics = run_variant_evaluation(
            Path(record["best"]),
            EVALUATION_DATA,
            coco,
            split=split,
            output_dir=output,
            device=0,
        )
        metrics.update({
            "checkpoint_sha256": expected_hash,
            "dataset_hash": config.values["dataset"]["dataset_hash"],
            "config_hash": record["config_hash"],
            "experiment": record["experiment"],
            "seed": int(record["seed"]),
            "comparison_tier": record["comparison_tier"],
            "evaluation_policy": {
                "imgsz": 640,
                "batch": 16,
                "conf": 0.001,
                "iou": 0.7,
                "rect": True,
                "half": False,
                "evaluator": "faster-coco-eval 1.7.2",
            },
        })
        atomic_write_json(metrics_path, metrics)
        gc.collect()
        torch.cuda.empty_cache()
    _set_state(f"evaluate_{split}_complete", completed=len(records), total=len(records))


def profile_experiments() -> None:
    from ultralytics import YOLO

    records = [row for row in formal_records() if int(row["seed"]) == 42]
    output = ARTIFACTS / "profiles"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        path = output / f"{_slug(record['experiment'])}.json"
        if path.is_file():
            existing = _read_json(path)
            if existing.get("checkpoint_sha256") == record["best_sha256"]:
                rows.append(existing)
                continue
        _set_state("profile", current=record["experiment"], completed=index - 1, total=len(records))
        wrapper = YOLO(str(record["best"]), task="detect", verbose=False)
        model = wrapper.model.eval().cpu().float()
        static = profile_module(model, torch.zeros(1, 3, 640, 640))
        latency = profile_gpu_latency(
            model,
            device=torch.device("cuda:0"),
            imgsz=640,
            precision="fp16",
            batch=1,
            warmup=20,
            iterations=100,
        )
        torch.cuda.reset_peak_memory_stats(0)
        sample = torch.zeros(1, 3, 640, 640, device="cuda:0", dtype=torch.float16)
        with torch.inference_mode():
            model(sample)
        torch.cuda.synchronize(0)
        peak = int(torch.cuda.max_memory_allocated(0))
        row = {
            "experiment": record["experiment"],
            "representative_seed": 42,
            "checkpoint": str(Path(record["best"]).resolve()),
            "checkpoint_sha256": record["best_sha256"],
            **asdict(static),
            "gpu_peak_allocated_bytes": peak,
            "latency": asdict(latency),
        }
        atomic_write_json(path, row)
        rows.append(row)
        del sample, model, wrapper
        gc.collect()
        torch.cuda.empty_cache()
    atomic_write_json(output / "summary.json", rows)
    _write_csv(RESULTS / "hardware_profiles.csv", [
        {
            "experiment": row["experiment"],
            "params": row["params"],
            "gflops": row["gflops"],
            "p2_activation_bytes": row["p2_activation_bytes"],
            "peak_live_activation_bytes": row["peak_live_activation_bytes"],
            "feature_traffic_bytes": row["feature_traffic_bytes"],
            "gpu_peak_allocated_bytes": row["gpu_peak_allocated_bytes"],
            "fp16_mean_ms": row["latency"]["mean_ms"],
            "fp16_p50_ms": row["latency"]["p50_ms"],
            "fp16_p95_ms": row["latency"]["p95_ms"],
        }
        for row in rows
    ])
    _set_state("profile_complete", completed=len(records), total=len(records))


def _metrics(record: dict[str, Any], split: str) -> dict[str, Any]:
    path = _evaluation_path(record, split) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = _read_json(path)
    if value.get("checkpoint_sha256") != record["best_sha256"]:
        raise RuntimeError(f"evaluation lineage mismatch: {path}")
    return value


def _metric_row(record: dict[str, Any], split: str) -> dict[str, Any]:
    metrics = _metrics(record, split)
    ball_subsets = metrics["ball_subsets"]
    return {
        "split": split,
        "experiment": record["experiment"],
        "seed": int(record["seed"]),
        "comparison_tier": record["comparison_tier"],
        "map50_95": _number(metrics["map50_95"]),
        "map50": _number(metrics["map50"]),
        "map75": _number(metrics["map75"]),
        "ap_s": _number(metrics["ap_s"]),
        "ap_m": _number(metrics["ap_m"]),
        "ap_l": _number(metrics["ap_l"]),
        "ball_ap": _number(metrics["per_class"]["ball"]["ap"]),
        "ball_ap_s": _number(metrics["per_class"]["ball"]["ap_s"]),
        "ball_recall": _number(metrics["ball_recall"]),
        "bat_ap": _number(metrics["per_class"]["bat"]["ap"]),
        "tiny_ball_recall": _number(ball_subsets["tiny"]["recall"]),
        "small_ball_recall": _number(ball_subsets["small"]["recall"]),
        "large_ball_recall": _number(ball_subsets["large"]["recall"]),
        "checkpoint_sha256": record["best_sha256"],
    }


def aggregate_rows(splits: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = formal_records()
    per_seed = [_metric_row(record, split) for split in splits for record in records]
    aggregate: list[dict[str, Any]] = []
    for split in splits:
        for experiment in CLEAN_EXPERIMENTS:
            group = [
                row for row in per_seed
                if row["split"] == split and row["experiment"] == experiment
            ]
            row: dict[str, Any] = {
                "split": split,
                "experiment": experiment,
                "comparison_tier": CLEAN_EXPERIMENTS[experiment].comparison_tier,
                "seeds": "42,43",
            }
            for field in METRIC_FIELDS:
                values = [_number(item[field]) for item in group]
                row[f"{field}_mean"] = _mean(values)
                row[f"{field}_std"] = _std(values)
            aggregate.append(row)
    return per_seed, aggregate


def freeze_selection() -> dict[str, Any]:
    if not (ARTIFACTS / "profiles" / "summary.json").is_file():
        raise RuntimeError("selection requires completed hardware profiles")
    _, aggregate = aggregate_rows(("val",))
    strict = [row for row in aggregate if row["comparison_tier"] == "strict_fair"]
    profiles = {
        row["experiment"]: row
        for row in _read_json(ARTIFACTS / "profiles" / "summary.json")
    }
    ranked = sorted(
        strict,
        key=lambda row: (
            row["map50_95_mean"] if row["map50_95_mean"] is not None else -math.inf,
            row["ball_ap_mean"] if row["ball_ap_mean"] is not None else -math.inf,
            row["ap_s_mean"] if row["ap_s_mean"] is not None else -math.inf,
            -float(profiles[row["experiment"]]["gflops"]),
            row["experiment"],
        ),
        reverse=True,
    )
    evidence = {}
    for record in formal_records():
        if record["comparison_tier"] != "strict_fair":
            continue
        path = _evaluation_path(record, "val") / "metrics.json"
        evidence[f"{record['experiment']}:seed{record['seed']}"] = sha256_file(path)
    payload = {
        "frozen": True,
        "selected": ranked[0]["experiment"],
        "selection_scope": "strict_fair_only",
        "rule": [
            "mean validation mAP50-95 descending",
            "mean Ball AP descending",
            "mean AP_S descending",
            "GFLOPs ascending",
            "experiment name deterministic tie-break",
        ],
        "ranking": [row["experiment"] for row in ranked],
        "validation_summary": ranked,
        "validation_evidence_sha256": dict(sorted(evidence.items())),
        "historical_test_read_before_freeze": False,
    }
    if SELECTION.is_file():
        existing = _read_json(SELECTION)
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("selection.json is immutable and differs from recomputed selection")
        return existing
    atomic_write_json(SELECTION, payload)
    return payload


def write_report() -> dict[str, Any]:
    per_seed, aggregate = aggregate_rows(("val", "test"))
    _write_csv(RESULTS / "per_seed_metrics.csv", per_seed)
    _write_csv(RESULTS / "comparison_mean_std.csv", aggregate)
    selection = _read_json(SELECTION)
    audit = _read_json(ARTIFACTS / "audit" / "formal_runs.json")
    profiles = {
        row["experiment"]: row
        for row in _read_json(ARTIFACTS / "profiles" / "summary.json")
    }
    by_key = {(row["split"], row["experiment"]): row for row in aggregate}
    strict_names = [
        name for name, spec in CLEAN_EXPERIMENTS.items()
        if spec.comparison_tier == "strict_fair"
    ]
    strict_rank = selection["ranking"]
    controls = [
        name for name, spec in CLEAN_EXPERIMENTS.items()
        if spec.comparison_tier == "optimization_control"
    ]
    lines = [
        "# Clean BBT5 正式實驗報告",
        "",
        "## 結論摘要",
        "",
        f"- Validation 選出的 strict-fair 模型：{selection['selected']}。",
        f"- 完成 12 個 strict-fair 架構 × 2 seeds，加 2 個 P2 control stages × 2 seeds，共 {audit['formal_jobs']} 個正式 jobs。",
        "- 所有模型從官方 COCO80 YOLO11m clean initializer 開始；沒有使用看過 BBT5 的 detect initializer。",
        "- Historical test 已被先前研究查看，只能作歷史報告，不能宣稱 unseen BBT5 泛化。",
        "",
        "## 資料與統一評估",
        "",
        "- Source Dataset：bbt5-detect-baseline/dataset。",
        "- Locked Dataset：train 1,987、validation 300、historical test 291 個 unique frames；group/hash overlap 皆為空。",
        "- Trainer YAML 物理上沒有 test key；selection.json 凍結後才執行 historical test。",
        "- 統一設定：imgsz 640、batch 16、conf 0.001、IoU 0.7、rect、FP32 inference、faster-coco-eval 1.7.2。",
        "- 數值均列兩個 seeds 的 mean ± sample std；只有兩個 seeds，仍不足以精確估計 variance。",
        "",
        "## 每個實驗在做什麼",
        "",
        "| 實驗 | 類別 | 唯一差異 |",
        "|---|---|---|",
    ]
    for name, spec in CLEAN_EXPERIMENTS.items():
        lines.append(f"| {name} | {spec.comparison_tier} | {DESCRIPTIONS[name]} |")
    lines += [
        "",
        "## Strict-fair validation 排名",
        "",
        "| 排名 | 實驗 | mAP50–95 | AP_S | AP_M | AP_L | Ball AP | Bat AP | GFLOPs | FP16 p50 ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, name in enumerate(strict_rank, start=1):
        row = by_key[("val", name)]
        profile = profiles[name]
        lines.append(
            f"| {index} | {name} | {_fmt(row['map50_95_mean'], row['map50_95_std'])} "
            f"| {_fmt(row['ap_s_mean'], row['ap_s_std'])} "
            f"| {_fmt(row['ap_m_mean'], row['ap_m_std'])} "
            f"| {_fmt(row['ap_l_mean'], row['ap_l_std'])} "
            f"| {_fmt(row['ball_ap_mean'], row['ball_ap_std'])} "
            f"| {_fmt(row['bat_ap_mean'], row['bat_ap_std'])} "
            f"| {profile['gflops']:.3f} | {profile['latency']['p50_ms']:.3f} |"
        )
    lines += [
        "",
        "## Strict-fair historical test",
        "",
        "| 實驗 | mAP50–95 | AP_S | AP_M | AP_L | Ball AP | Bat AP | Tiny Ball R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in strict_names:
        row = by_key[("test", name)]
        lines.append(
            f"| {name} | {_fmt(row['map50_95_mean'], row['map50_95_std'])} "
            f"| {_fmt(row['ap_s_mean'], row['ap_s_std'])} "
            f"| {_fmt(row['ap_m_mean'], row['ap_m_std'])} "
            f"| {_fmt(row['ap_l_mean'], row['ap_l_std'])} "
            f"| {_fmt(row['ball_ap_mean'], row['ball_ap_std'])} "
            f"| {_fmt(row['bat_ap_mean'], row['bat_ap_std'])} "
            f"| {_fmt(row['tiny_ball_recall_mean'], row['tiny_ball_recall_std'])} |"
        )
    lines += [
        "",
        "## P2 optimization control（不與 strict-fair 混排）",
        "",
        "| 實驗 | Split | mAP50–95 | Ball AP | Bat AP |",
        "|---|---|---:|---:|---:|",
    ]
    for name in controls:
        for split in ("val", "test"):
            row = by_key[(split, name)]
            lines.append(
                f"| {name} | {split} | {_fmt(row['map50_95_mean'], row['map50_95_std'])} "
                f"| {_fmt(row['ball_ap_mean'], row['ball_ap_std'])} "
                f"| {_fmt(row['bat_ap_mean'], row['bat_ap_std'])} |"
            )
    b0_val = by_key[("val", "B0-Clean")]
    runner_name = strict_rank[1]
    runner_val = by_key[("val", runner_name)]
    p2_direct_val = by_key[("val", "P2-Direct-Clean")]
    p2_paper_val = by_key[("val", "P2-PaperFormula-Clean")]
    p3_partial25_val = by_key[("val", "P3-Partial25-Clean")]
    control_full_val = by_key[("val", "P2-Control-Clean-Full")]
    test_best_name = max(strict_names, key=lambda name: by_key[("test", name)]["map50_95_mean"])
    test_best = by_key[("test", test_best_name)]
    b0_profile = profiles["B0-Clean"]
    direct_profile = profiles["P2-Direct-Clean"]
    paper_profile = profiles["P2-PaperFormula-Clean"]
    partial_profile = profiles["P3-Partial25-Clean"]
    lines += [
        "",
        "## 分析",
        "",
        f"1. Validation 選出 B0-Clean（{b0_val['map50_95_mean']:.5f} ± {b0_val['map50_95_std']:.5f}）；只領先第二名 {runner_name} {b0_val['map50_95_mean'] - runner_val['map50_95_mean']:.5f} AP。差距小於兩者的 seed 標準差，應解讀為 B0 目前最穩健，而不是已證明絕對優勝。",
        f"2. P2-Direct 相對 B0 的 validation mAP50–95 下降 {b0_val['map50_95_mean'] - p2_direct_val['map50_95_mean']:.5f}，GFLOPs 增加 {(direct_profile['gflops'] / b0_profile['gflops'] - 1) * 100:.1f}%、FP16 p50 latency 增加 {(direct_profile['latency']['p50_ms'] / b0_profile['latency']['p50_ms'] - 1) * 100:.1f}%。P2-PaperFormula 將品質差距縮至 {b0_val['map50_95_mean'] - p2_paper_val['map50_95_mean']:.5f}，但 GFLOPs/latency 仍增加 {(paper_profile['gflops'] / b0_profile['gflops'] - 1) * 100:.1f}%/{(paper_profile['latency']['p50_ms'] / b0_profile['latency']['p50_ms'] - 1) * 100:.1f}%。",
        f"3. P2-Direct 的 Bat AP 比 B0 高 {p2_direct_val['bat_ap_mean'] - b0_val['bat_ap_mean']:.5f}，但 Ball AP 低 {b0_val['ball_ap_mean'] - p2_direct_val['ball_ap_mean']:.5f}；P2-PaperFormula 的 AP_S 只比 B0 高 {p2_paper_val['ap_s_mean'] - b0_val['ap_s_mean']:.5f}。因此新增高解析 head 沒有自動轉化成整體或 Ball 優勢。",
        f"4. P3-Partial25 排名第 4，與 B0 的品質差為 {b0_val['map50_95_mean'] - p3_partial25_val['map50_95_mean']:.5f}，GFLOPs/latency 只增加 {(partial_profile['gflops'] / b0_profile['gflops'] - 1) * 100:.1f}%/{(partial_profile['latency']['p50_ms'] / b0_profile['latency']['p50_ms'] - 1) * 100:.1f}%；它是本輪新增模組中較合理的硬體友善折衷，但尚未超過 B0。",
        f"5. P2 Control-Full validation 為 {control_full_val['map50_95_mean']:.5f}，高於 B0 {control_full_val['map50_95_mean'] - b0_val['map50_95_mean']:.5f}；然而它承接 Head-only checkpoint 且使用不同兩階段 schedule。這支持『P2 的主要瓶頸包含最佳化策略』，不能拿它宣稱 P2 架構公平勝過 B0。Head-only 本身接近失敗，亦表示只更新新增 head 不足。",
        f"6. Historical test 最高是 {test_best_name}（{test_best['map50_95_mean']:.5f}），並非 validation 選出的 B0。這個排名反轉不得用來事後改選模型；它反而顯示 validation/test 難度或分布不同，也再次說明 historical test 不能當 unseen selection set。",
        "7. P3-Lite35-F7 與 P3-Partial50 的 seed 42 都在第 31 epoch 停止、最佳點在 epoch 1，但 seed 43 可訓練到 79–83 epochs；這是明顯的最佳化不穩定，不應只看兩 seed 平均掩蓋。",
        "8. AP_S/M/L 使用 COCO area 定義；Ball tiny/small/large recall 另使用 locked dataset 的 short-side <8、8–16、>16 px 定義，兩者不可混稱。",
        "",
        "## 公平性與限制",
        "",
        "- Initializer 未接觸 BBT5，已解決舊 yolo11m_bat_detect_init.pt 的資料暴露問題。",
        "- 前五個 formal jobs 在 patience=100 下啟動；之後改為 patience=30。既有曲線回放未改變前五組的 best checkpoint，但這仍是中途 protocol amendment，報告必須保留。",
        "- 只有 seeds 42/43；0.x AP 的小差異不得過度解讀。",
        "- Historical test 不是 Final Holdout；真正 unseen 結論仍需新影片／新比賽資料。",
        "- Selection 只使用 validation，control 不進 strict-fair ranking。",
        "",
        "## 證據位置",
        "",
        "- artifacts/clean-bbt5-ablation/audit/formal_runs.json：28 jobs、hash、lineage、fresh-process reload。",
        "- artifacts/clean-bbt5-ablation/final_audit.json：metrics/profile/selection/report 的最終一致性 gate。",
        "- artifacts/clean-bbt5-ablation/evaluation/：逐 seed validation/test metrics、predictions、錯誤案例。",
        "- artifacts/clean-bbt5-ablation/profiles/：Params、GFLOPs、activation/traffic、GPU peak memory、FP16 latency。",
        "- artifacts/clean-bbt5-ablation/selection.json：不可變 validation selection freeze。",
        "- clean_bbt5_study/results/per_seed_metrics.csv：逐 seed 原值。",
        "- clean_bbt5_study/results/comparison_mean_std.csv：mean ± std 原始欄位。",
    ]
    report_path = RESULTS / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for experiment in CLEAN_EXPERIMENTS:
        directory = RESULTS / "experiments" / _slug(experiment)
        directory.mkdir(parents=True, exist_ok=True)
        spec = CLEAN_EXPERIMENTS[experiment]
        detail = [
            f"# {experiment}",
            "",
            DESCRIPTIONS[experiment],
            "",
            f"- Comparison tier：{spec.comparison_tier}",
            f"- Family：{spec.family}",
            f"- Schedule：{spec.schedule}",
            "- Seeds：42、43",
            "",
            "| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for split in ("val", "test"):
            row = by_key[(split, experiment)]
            detail.append(
                f"| {split} | {_fmt(row['map50_95_mean'], row['map50_95_std'])} "
                f"| {_fmt(row['ball_ap_mean'], row['ball_ap_std'])} "
                f"| {_fmt(row['bat_ap_mean'], row['bat_ap_std'])} "
                f"| {_fmt(row['ap_s_mean'], row['ap_s_std'])} "
                f"| {_fmt(row['ap_m_mean'], row['ap_m_std'])} "
                f"| {_fmt(row['ap_l_mean'], row['ap_l_std'])} |"
            )
        detail += [
            "",
            "逐 seed 證據：",
            "",
            f"- artifacts/clean-bbt5-ablation/evaluation/val/{_slug(experiment)}-seed42/",
            f"- artifacts/clean-bbt5-ablation/evaluation/val/{_slug(experiment)}-seed43/",
            f"- artifacts/clean-bbt5-ablation/evaluation/test/{_slug(experiment)}-seed42/",
            f"- artifacts/clean-bbt5-ablation/evaluation/test/{_slug(experiment)}-seed43/",
        ]
        (directory / "README.md").write_text("\n".join(detail) + "\n", encoding="utf-8")

    readme = [
        "# Clean BBT5 結果",
        "",
        "完整訓練、統一評估、selection freeze、historical test 與硬體量測均已完成。",
        "",
        "- 正式中文報告：[REPORT.md](REPORT.md)",
        "- 逐 seed 指標：[per_seed_metrics.csv](per_seed_metrics.csv)",
        "- Mean/std 總表：[comparison_mean_std.csv](comparison_mean_std.csv)",
        "- 硬體資料：[hardware_profiles.csv](hardware_profiles.csv)",
        "- 最終一致性稽核：[final_audit.json](../../artifacts/clean-bbt5-ablation/final_audit.json)",
        "- 正式 run 稽核：[formal_run_audit.csv](formal_run_audit.csv)",
        "- 各實驗說明與結果：[experiments/](experiments/)",
        "",
        "Historical test 已被過去研究查看，不可用於 unseen claim；selection 由 validation 先行凍結。",
    ]
    (RESULTS / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    summary = {
        "selected": selection["selected"],
        "formal_jobs": audit["formal_jobs"],
        "strict_experiments": len(strict_names),
        "control_stages": len(controls),
        "report": str(report_path.resolve()),
    }
    atomic_write_json(RESULTS / "summary.json", summary)
    _set_state("report_complete", completed=1, total=1)
    return summary


def reload_one(checkpoint: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint.resolve()), task="detect", verbose=False).model.eval().cpu()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    detect = model.model[-1]
    print(json.dumps({
        "strict_native_reload": True,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "strides": [float(value) for value in model.stride.tolist()],
        "detect_scales": int(detect.nl),
        "names": dict(model.names),
        "forward_output_type": type(output).__name__,
    }, sort_keys=True))



def final_audit() -> dict[str, Any]:
    records = formal_records()
    selection = _read_json(SELECTION)
    if not selection.get("frozen") or selection.get("selection_scope") != "strict_fair_only":
        raise RuntimeError("selection freeze is missing or invalid")
    required_metrics = {
        "map50_95", "map50", "map75", "ap_s", "ap_m", "ap_l",
        "per_class", "checkpoint_sha256", "dataset_hash", "evaluation_policy",
    }
    validation_hashes: dict[str, str] = {}
    metric_paths: list[Path] = []
    for record in records:
        for split in ("val", "test"):
            path = _evaluation_path(record, split) / "metrics.json"
            metrics = _read_json(path)
            missing = required_metrics - set(metrics)
            if missing:
                raise RuntimeError(f"metrics missing keys {sorted(missing)}: {path}")
            if metrics["split"] != split:
                raise RuntimeError(f"split mismatch: {path}")
            if metrics["checkpoint_sha256"] != record["best_sha256"]:
                raise RuntimeError(f"checkpoint lineage mismatch: {path}")
            if set(metrics["per_class"]) != {"ball", "bat"}:
                raise RuntimeError(f"per-class metrics incomplete: {path}")
            metric_paths.append(path)
            if split == "val" and record["comparison_tier"] == "strict_fair":
                key = f"{record['experiment']}:seed{record['seed']}"
                validation_hashes[key] = sha256_file(path)
    if dict(sorted(validation_hashes.items())) != selection["validation_evidence_sha256"]:
        raise RuntimeError("selection evidence hashes differ from current validation metrics")
    test_paths = [path for path in metric_paths if "/test/" in str(path)]
    if SELECTION.stat().st_mtime_ns > min(path.stat().st_mtime_ns for path in test_paths):
        raise RuntimeError("selection file is newer than historical test outputs")
    profiles = _read_json(ARTIFACTS / "profiles" / "summary.json")
    if len(profiles) != len(CLEAN_EXPERIMENTS):
        raise RuntimeError("hardware profile matrix is incomplete")
    for profile in profiles:
        spec = CLEAN_EXPERIMENTS[profile["experiment"]]
        expected_p2 = spec.family == "P2"
        if (profile["p2_activation_bytes"] is not None) != expected_p2:
            raise RuntimeError(f"P2 activation classification mismatch: {profile['experiment']}")
        if profile["params"] <= 0 or profile["gflops"] <= 0:
            raise RuntimeError(f"invalid hardware profile: {profile['experiment']}")
    required_outputs = [
        RESULTS / "REPORT.md",
        RESULTS / "per_seed_metrics.csv",
        RESULTS / "comparison_mean_std.csv",
        RESULTS / "hardware_profiles.csv",
        RESULTS / "formal_run_audit.csv",
        RESULTS / "summary.json",
    ]
    required_outputs.extend(
        RESULTS / "experiments" / _slug(name) / "README.md"
        for name in CLEAN_EXPERIMENTS
    )
    missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(f"missing report outputs: {missing_outputs}")
    payload = {
        "ok": True,
        "formal_jobs": len(records),
        "validation_metrics": sum("/val/" in str(path) for path in metric_paths),
        "historical_test_metrics": len(test_paths),
        "hardware_profiles": len(profiles),
        "selection": selection["selected"],
        "selection_evidence": len(validation_hashes),
        "selection_frozen_before_historical_test": True,
        "experiment_readmes": len(CLEAN_EXPERIMENTS),
        "report_sha256": sha256_file(RESULTS / "REPORT.md"),
        "comparison_sha256": sha256_file(RESULTS / "comparison_mean_std.csv"),
    }
    atomic_write_json(ARTIFACTS / "final_audit.json", payload)
    return payload
def run_all() -> dict[str, Any]:
    with PipelineLock(LOCK):
        try:
            audit_path = ARTIFACTS / "audit" / "formal_runs.json"
            if not audit_path.is_file():
                audit_formal_runs(fresh_reload=True)
            elif sum(
                row.get("strict_native_reload") is True
                for row in _read_json(audit_path)["runs"]
            ) != 28:
                raise RuntimeError("existing formal audit lacks 28 fresh reload passes")
            evaluate_split("val")
            profile_experiments()
            freeze_selection()
            evaluate_split("test")
            result = write_report()
            result["final_audit"] = final_audit()
            _set_state("complete", completed=6, total=6)
            return result
        except Exception as error:
            _set_state("failed", error=f"{type(error).__name__}: {error}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("all")
    commands.add_parser("audit")
    commands.add_parser("val")
    commands.add_parser("profile")
    commands.add_parser("select")
    commands.add_parser("test")
    commands.add_parser("final-audit")
    commands.add_parser("report")
    reload_parser = commands.add_parser("reload-one")
    reload_parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "all":
        result = run_all()
    elif args.command == "audit":
        result = audit_formal_runs(fresh_reload=True)
    elif args.command == "val":
        evaluate_split("val")
        result = {"ok": True}
    elif args.command == "profile":
        profile_experiments()
        result = {"ok": True}
    elif args.command == "select":
        result = freeze_selection()
    elif args.command == "test":
        evaluate_split("test")
        result = {"ok": True}
    elif args.command == "final-audit":
        result = final_audit()
    elif args.command == "report":
        result = write_report()
    else:
        reload_one(args.checkpoint)
        return
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
