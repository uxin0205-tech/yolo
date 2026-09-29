"""完整 C2/C3 與量化結果的最終自動整理及可追溯選型。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import file_sha256
from .full_training import FullRunConfig, _verify_export_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 根節點必須是 mapping")
    return payload


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _metric_vector(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "coco_box_map50": float(payload["detect"]["box"]["ap"]["map50"]),
        "coco_box_map50_95": float(
            payload["detect"]["box"]["ap"]["map50_95"]
        ),
        "bbat5_pose_box_map50": float(payload["pose"]["box"]["ap"]["map50"]),
        "bbat5_pose_box_map50_95": float(
            payload["pose"]["box"]["ap"]["map50_95"]
        ),
        "bbat5_keypoint_map50": float(
            payload["pose"]["keypoints"]["ap"]["map50"]
        ),
        "bbat5_keypoint_map50_95": float(
            payload["pose"]["keypoints"]["ap"]["map50_95"]
        ),
        "macro_f1": float(payload["pose"]["keypoints"]["f1"]["macro_f1"]),
        "micro_f1": float(payload["pose"]["keypoints"]["f1"]["micro_f1"]),
        "pose_official_combined_fitness": float(
            payload["pose"]["official_combined_fitness"]
        ),
    }


def _joint(metrics: dict[str, float]) -> float:
    return (
        metrics["coco_box_map50_95"]
        + 0.25 * metrics["bbat5_keypoint_map50_95"]
    ) / 1.25


def _cost_index(float20: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        item["metrics"]["candidate_id"]: {
            "params": float(item["metrics"]["params"]),
            "gflops": float(item["metrics"]["gflops"]),
            "latency_ms": float(item["metrics"]["latency_ms"]),
            "peak_vram_mb": float(item["metrics"]["peak_vram_mb"]),
        }
        for item in float20["candidates"]
    }


def _candidate_record(
    config: FullRunConfig,
    candidate: str,
    *,
    costs: dict[str, dict[str, float]],
) -> dict[str, Any]:
    run_dir = config.run_root / f"{candidate.lower()}-full-seed{config.seed}"
    complete_path = run_dir / "complete.json"
    complete = _read(complete_path)
    if complete.get("status") != "completed_formal_training":
        raise ValueError(f"{candidate} full training 未完成")
    epoch = int(complete["best_state"]["joint_formal"]["epoch"])
    metrics_path = run_dir / f"validation/epoch-{epoch:04d}/float/metrics.json"
    validation = _read(metrics_path)
    if (
        validation.get("formal_split_used") is not True
        or validation.get("backend") != "float"
    ):
        raise ValueError(f"{candidate} formal metrics 契約漂移")
    metrics = _metric_vector(validation)
    quant_root = (
        Path(str(config.payload["quantization"]["result_root"])).expanduser()
    )
    if not quant_root.is_absolute():
        quant_root = PROJECT_ROOT / quant_root
    quant_path = quant_root / candidate.lower() / "complete.json"
    quant = _read(quant_path)
    if quant.get("status") != "completed_q0_q1_q2l":
        raise ValueError(f"{candidate} quant matrix 未完成")
    q_metrics = quant["metrics"]
    quant_stage_scores = {
        stage: _joint({name: float(value) for name, value in values.items()})
        for stage, values in q_metrics.items()
    }
    compatible_stages: list[str] = []
    if quant.get("ptq_compatible") is True:
        compatible_stages.append("q1")
    if quant.get("qat_lite_compatible") is True:
        compatible_stages.append("q2l")
    recommended_stage = (
        max(compatible_stages, key=quant_stage_scores.__getitem__)
        if compatible_stages
        else "float_only"
    )
    return {
        "candidate": candidate,
        "formal_metrics": metrics,
        "formal_joint_score": _joint(metrics),
        "formal_best_epoch": epoch,
        "formal_metrics_path": str(metrics_path),
        "formal_metrics_sha256": file_sha256(metrics_path),
        "full_checkpoint": complete["best_inference_checkpoint"],
        "full_checkpoint_sha256": file_sha256(
            Path(complete["best_inference_checkpoint"])
        ),
        "cost": costs[candidate],
        "quantization": {
            "simulation_only": True,
            "ptq_compatible": bool(quant["ptq_compatible"]),
            "qat_lite_compatible": bool(quant["qat_lite_compatible"]),
            "recommended_stage": recommended_stage,
            "stage_joint_scores": quant_stage_scores,
            "result_path": str(quant_path),
            "result_sha256": file_sha256(quant_path),
            "deployment_int8_validated": False,
        },
    }


def _select(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "selection_status": "no_eligible_candidate",
            "c_best": None,
            "near_tie_candidates": [],
            "rule_zh": "Float20沒有候選通過0.008與成本門檻，因此停止。",
        }
    highest = max(float(item["formal_joint_score"]) for item in records)
    near = [
        item
        for item in records
        if highest - float(item["formal_joint_score"]) <= 0.008 + 1e-12
    ]
    winner = min(
        near,
        key=lambda item: (
            float(item["cost"]["latency_ms"]),
            float(item["cost"]["gflops"]),
            -float(item["formal_metrics"]["macro_f1"]),
            str(item["candidate"]),
        ),
    )
    return {
        "selection_status": "selected_after_formal_and_quant_validation",
        "c_best": winner["candidate"],
        "near_tie_candidates": [item["candidate"] for item in near],
        "highest_formal_joint_score": highest,
        "selected_formal_joint_score": winner["formal_joint_score"],
        "selected_quant_stage": winner["quantization"]["recommended_stage"],
        "rule_zh": (
            "先取正式joint score最高者；差距不超過0.008視為近似，"
            "近似候選依latency、GFLOPs、Macro F1與candidate ID依序決勝。"
        ),
    }


def _markdown(
    records: list[dict[str, Any]],
    selection: dict[str, Any],
) -> str:
    lines = [
        "# C2/C3完整訓練與量化接續結果",
        "",
        f"- 狀態：{selection['selection_status']}",
        f"- C_best：{selection['c_best']}",
        f"- 近似候選：{', '.join(selection['near_tie_candidates']) or '無'}",
        f"- 選型規則：{selection['rule_zh']}",
        "",
        "## 候選摘要",
        "",
        "| 候選 | 正式joint | COCO mAP50-95 | Pose Keypoint mAP50-95 | Macro F1 | latency ms | PTQ | QAT-lite | 建議量化階段 |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in records:
        metrics = item["formal_metrics"]
        quant = item["quantization"]
        lines.append(
            "| {candidate} | {joint:.6f} | {detect:.6f} | {pose:.6f} | "
            "{f1:.6f} | {latency:.4f} | {ptq} | {qat} | {stage} |".format(
                candidate=item["candidate"],
                joint=item["formal_joint_score"],
                detect=metrics["coco_box_map50_95"],
                pose=metrics["bbat5_keypoint_map50_95"],
                f1=metrics["macro_f1"],
                latency=item["cost"]["latency_ms"],
                ptq="相容" if quant["ptq_compatible"] else "不相容",
                qat="相容" if quant["qat_lite_compatible"] else "不相容",
                stage=quant["recommended_stage"],
            )
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- PTQ與QAT-lite均為W8A8 fake-quant simulation，不是部署INT8或Bit-True證據。",
            "- C_best只在本次C2/C3與既定資料、seed、預算及門檻範圍內成立。",
            "",
        ]
    )
    return "\n".join(lines)


def export_downstream_results(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    config = FullRunConfig.load(config_path)
    float20 = _verify_export_manifest(config.float_results)
    matrix = _read(config.run_root / "matrix-complete.json")
    status = matrix.get("status")
    candidates = [str(value) for value in matrix.get("eligible_candidates", [])]
    if status not in {
        "completed_no_eligible_candidates",
        "completed_formal_training_matrix",
    }:
        raise ValueError(f"full matrix status 漂移：{status}")
    costs = _cost_index(float20)
    records = [
        _candidate_record(config, candidate, costs=costs)
        for candidate in candidates
    ]
    selection = _select(records)
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else PROJECT_ROOT / "results/full35-j3-c2-c3-final-seed0"
    )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        destination / "results.json",
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "records": records,
                "selection": selection,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_text(destination / "REPORT.md", _markdown(records, selection))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "selection_status": selection["selection_status"],
        "c_best": selection["c_best"],
        "files": {
            path.name: file_sha256(path)
            for path in sorted(destination.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    _atomic_text(
        destination / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest
