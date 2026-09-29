"""將完整 Float20 matrix 與 cost profiles 匯出為正式 JSON、CSV、報告與圖表。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import CANDIDATES, SPEC_PATH, SPEC_VERSION, file_sha256
from .decisions import CandidateMetrics, ClassMetrics, evaluate_float_results
from .screen_training import ScreenRunConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MICRO_F1_METHOD = "estimated_from_precision_recall_curves_and_supports"


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必須是 mapping")
    return dict(value)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        status = run("status", "--short")
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "revision": None,
            "working_tree_dirty": None,
            "inspection_error": f"{type(error).__name__}: {error}",
        }
    return {
        "revision": revision,
        "working_tree_dirty": bool(status),
        "inspection_error": None,
    }


def _best_validation(
    config: ScreenRunConfig,
    candidate: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_dir = config.run_root / f"{candidate.lower()}-control-seed{config.seed}"
    complete_path = run_dir / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "completed_screening":
        raise ValueError(f"{candidate} complete status 漂移")
    expected_steps = int(config.payload["training"]["expected_optimizer_steps_per_candidate"])
    if (
        complete.get("epochs_completed") != config.epochs
        or complete.get("global_macro_steps") != expected_steps
    ):
        raise ValueError(f"{candidate} epoch／optimizer steps 不完整")
    validation_paths = sorted(run_dir.glob("validation/epoch-*/float/metrics.json"))
    expected_validation_paths = [
        run_dir / f"validation/epoch-{epoch:04d}/float/metrics.json" for epoch in range(config.epochs)
    ]
    if validation_paths != expected_validation_paths:
        raise ValueError(f"{candidate} validation events 不完整／不連續")
    for label in (
        "best-detect",
        "best-pose-research",
        "best-pose-official",
        "best-joint-screening",
    ):
        for kind in ("checkpoints", "inference"):
            if not (run_dir / kind / f"{label}.pt").is_file():
                raise FileNotFoundError(f"{candidate} 缺少 {kind}/{label}.pt")
    best = _mapping(complete.get("best_state"), f"{candidate}.best_state")
    joint = _mapping(best.get("joint_screening"), f"{candidate}.best_joint")
    epoch = int(joint["epoch"])
    if epoch < 0 or epoch >= config.epochs:
        raise ValueError(f"{candidate} best joint epoch 無效：{epoch}")
    metrics_path = run_dir / f"validation/epoch-{epoch:04d}/float/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("epoch") != epoch
        or metrics.get("backend") != "float"
        or metrics.get("formal_split_used") is not False
    ):
        raise ValueError(f"{candidate} best validation 契約漂移")
    return complete, metrics, metrics_path


def _class_metrics(
    pose: dict[str, Any],
    *,
    class_id: str,
    expected_name: str,
) -> ClassMetrics:
    box_ap = _mapping(
        _mapping(_mapping(pose["box"], "pose.box")["ap"], "pose.box.ap")["per_class"][class_id],
        f"pose.box.class.{class_id}",
    )
    keypoint = _mapping(pose["keypoints"], "pose.keypoints")
    keypoint_ap = _mapping(
        _mapping(keypoint["ap"], "pose.keypoints.ap")["per_class"][class_id],
        f"pose.keypoints.class.{class_id}",
    )
    f1 = _mapping(
        _mapping(keypoint["f1"], "pose.keypoints.f1")["per_class"][class_id],
        f"pose.keypoints.f1.class.{class_id}",
    )
    if {
        str(box_ap.get("name")),
        str(keypoint_ap.get("name")),
        str(f1.get("name")),
    } != {expected_name}:
        raise ValueError(f"Pose class {class_id} 名稱漂移")
    return ClassMetrics(
        ap50=float(box_ap["ap50"]),
        ap50_95=float(box_ap["ap50_95"]),
        keypoint_ap50=float(keypoint_ap["ap50"]),
        keypoint_ap50_95=float(keypoint_ap["ap50_95"]),
        precision=float(f1["precision"]),
        recall=float(f1["recall"]),
        f1=float(f1["f1"]),
    )


def _candidate_metrics(
    candidate: str,
    validation: dict[str, Any],
    profile: dict[str, Any],
) -> CandidateMetrics:
    detect = _mapping(validation["detect"], "detect")
    detect_ap = _mapping(_mapping(detect["box"], "detect.box")["ap"], "detect.box.ap")
    person = _mapping(detect_ap["per_class"]["0"], "detect.person")
    if person.get("name") != "person":
        raise ValueError("COCO class 0 必須是 person")
    pose = _mapping(validation["pose"], "pose")
    if pose.get("status") != "measured":
        raise ValueError(f"{candidate} Pose 未完成")
    pose_box = _mapping(_mapping(pose["box"], "pose.box")["ap"], "pose.box.ap")
    pose_keypoint_node = _mapping(pose["keypoints"], "pose.keypoints")
    pose_keypoint = _mapping(pose_keypoint_node["ap"], "pose.keypoints.ap")
    keypoint_f1 = _mapping(pose_keypoint_node["f1"], "pose.keypoints.f1")
    if keypoint_f1.get("micro_f1_method") != MICRO_F1_METHOD:
        raise ValueError(f"{candidate} Micro F1 method 漂移")
    classes = {
        "ball": _class_metrics(pose, class_id="0", expected_name="ball"),
        "bat": _class_metrics(pose, class_id="1", expected_name="bat"),
    }
    both = _mapping(
        _mapping(profile["tasks"], f"{candidate}.profile.tasks")["both"],
        f"{candidate}.profile.both",
    )
    return CandidateMetrics(
        candidate_id=candidate,
        coco_box_map50=float(detect_ap["map50"]),
        coco_box_map50_95=float(detect_ap["map50_95"]),
        coco_person_ap50=float(person["ap50"]),
        coco_person_ap50_95=float(person["ap50_95"]),
        bbat5_pose_box_map50=float(pose_box["map50"]),
        bbat5_pose_box_map50_95=float(pose_box["map50_95"]),
        bbat5_keypoint_map50=float(pose_keypoint["map50"]),
        bbat5_keypoint_map50_95=float(pose_keypoint["map50_95"]),
        pose_official_combined_fitness=float(pose["official_combined_fitness"]),
        classes=classes,
        macro_f1=float(keypoint_f1["macro_f1"]),
        micro_f1=float(keypoint_f1["micro_f1"]),
        f1_confidence_threshold=float(keypoint_f1["confidence_threshold"]),
        params=int(profile["params"]),
        gflops=float(both["gflops"]),
        latency_ms=float(both["latency_median_ms"]),
        peak_vram_mb=float(both["peak_allocated_mib"]),
    )


def _candidate_build_report(
    config: ScreenRunConfig,
    candidate: str,
    complete: dict[str, Any],
) -> dict[str, Any]:
    run_dir = config.run_root / f"{candidate.lower()}-control-seed{config.seed}"
    path = run_dir / "run-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    build = _mapping(payload.get("candidate_build"), f"{candidate}.candidate_build")
    if (
        build.get("resolved_id") != candidate
        or build.get("candidate_id") != candidate
        or build.get("model_contract_unchanged") is not True
        or build.get("parent_unchanged") is not True
    ):
        raise ValueError(f"{candidate} candidate build 契約漂移")
    expected_fields = list(CANDIDATES[candidate].changed_fields)
    if build.get("changed_fields") != expected_fields:
        raise ValueError(f"{candidate} changed_fields 漂移")
    transfer = _mapping(build.get("transfer"), f"{candidate}.transfer")
    for key in ("loaded", "matched", "missing", "unexpected", "shape_mismatch"):
        if not isinstance(transfer.get(key), list):
            raise TypeError(f"{candidate}.transfer.{key} 必須是 list")
    if (
        transfer["loaded"] != transfer["matched"]
        or transfer.get("loaded_count") != len(transfer["loaded"])
        or transfer.get("matched_count") != len(transfer["matched"])
    ):
        raise ValueError(f"{candidate} transfer counts／matched 清單漂移")
    if payload.get("lineage") != complete.get("lineage"):
        raise ValueError(f"{candidate} run-manifest／complete lineage 漂移")
    return {
        "run_manifest": str(path),
        "run_manifest_sha256": file_sha256(path),
        "candidate_build": build,
        "summary": {
            "matched": len(transfer["matched"]),
            "missing": len(transfer["missing"]),
            "unexpected": len(transfer["unexpected"]),
            "shape_mismatch": len(transfer["shape_mismatch"]),
        },
    }


def _profile_index(
    path: Path,
    config: ScreenRunConfig,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "completed"
        or payload.get("spec_version") != SPEC_VERSION
        or payload.get("spec_sha256") != file_sha256(SPEC_PATH)
        or payload.get("training_yaml_sha256") != file_sha256(config.path)
    ):
        raise ValueError("cost profile metadata 漂移")
    entries = payload.get("profiles")
    if not isinstance(entries, list):
        raise TypeError("cost profiles 必須是 list")
    indexed = {
        str(entry["candidate"]): _mapping(entry, "profile") for entry in entries if isinstance(entry, dict)
    }
    if tuple(indexed) != config.candidates:
        raise ValueError("cost profile candidate 順序／矩陣漂移")
    return payload, indexed


def _csv_rows(
    config: ScreenRunConfig,
    handoff: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    profile_report: dict[str, Any],
    git: dict[str, Any],
) -> list[dict[str, Any]]:
    template = PROJECT_ROOT / "results/results-template.csv"
    with template.open(encoding="utf-8", newline="") as handle:
        fields = next(csv.reader(handle))
    rows: list[dict[str, Any]] = []
    handoff_row = {field: "" for field in fields}
    handoff_row.update(
        {
            "run_status": "handoff_reference",
            "formal_comparison_ready": "false",
            "handoff_revision": handoff["revision_id"],
            "winner_id": handoff["winner_id"],
            "fusion_kind": handoff["fusion_kind"],
            "candidate_id": "C0-Handoff",
            "comparison_role": "handoff_reference",
            "pose_executed": "false",
            "spec_version": SPEC_VERSION,
            "spec_sha256": file_sha256(SPEC_PATH),
            "parent_checkpoint_sha256": handoff["checkpoint"]["sha256"],
            "git_revision": git["revision"] or "",
            "seed": config.seed,
            "selection_status": "pending_user_decision",
            "quantization_eligibility": "eligible_q0_q1",
            "notes": "不訓練；只作 J3 winner 精確重建 reference",
        }
    )
    rows.append(handoff_row)
    for record in records:
        metric = record["metrics"]
        profile = record["profile"]
        classes = metric["classes"]
        lineage = record["lineage"]
        transfer = record["build_report"]["summary"]
        row = {field: "" for field in fields}
        row.update(
            {
                "run_status": "completed_screening",
                "formal_comparison_ready": "true",
                "handoff_revision": handoff["revision_id"],
                "winner_id": handoff["winner_id"],
                "fusion_kind": handoff["fusion_kind"],
                "candidate_id": metric["candidate_id"],
                "comparison_role": ("control" if metric["candidate_id"] == "C0" else "candidate"),
                "resolved_region": "shared-c3k2",
                "pose_executed": "true",
                "spec_version": SPEC_VERSION,
                "spec_sha256": file_sha256(SPEC_PATH),
                "architecture_yaml_sha256": lineage["architecture_yaml_sha256"],
                "training_yaml_sha256": lineage["training_yaml_sha256"],
                "detect_dataset_yaml_sha256": lineage["detect_dataset_yaml_sha256"],
                "pose_dataset_yaml_sha256": lineage["pose_dataset_yaml_sha256"],
                "parent_checkpoint_sha256": lineage["parent_checkpoint_sha256"],
                "checkpoint_sha256": profile["checkpoint_sha256"],
                "git_revision": git["revision"] or "",
                "seed": config.seed,
                "detect_logical_batch": config.detect_logical_batch,
                "fraction": config.fraction,
                "augmentation_scale": config.scale,
                "cache": str(config.cache).lower(),
                "imgsz": config.imgsz,
                "optimizer_steps": record["complete"]["global_macro_steps"],
                "validation_events": config.epochs,
                "coco_box_map50": metric["coco_box_map50"],
                "coco_box_map50_95": metric["coco_box_map50_95"],
                "coco_person_ap50": metric["coco_person_ap50"],
                "coco_person_ap50_95": metric["coco_person_ap50_95"],
                "bbat5_pose_box_map50": metric["bbat5_pose_box_map50"],
                "bbat5_pose_box_map50_95": metric["bbat5_pose_box_map50_95"],
                "bbat5_keypoint_map50": metric["bbat5_keypoint_map50"],
                "bbat5_keypoint_map50_95": metric["bbat5_keypoint_map50_95"],
                "pose_official_combined_fitness": metric["pose_official_combined_fitness"],
                "pose_checkpoint_selection": "best-joint-screening EMA",
                "ball_box_ap50": classes["ball"]["ap50"],
                "ball_box_ap50_95": classes["ball"]["ap50_95"],
                "ball_keypoint_ap50": classes["ball"]["keypoint_ap50"],
                "ball_keypoint_ap50_95": classes["ball"]["keypoint_ap50_95"],
                "ball_precision": classes["ball"]["precision"],
                "ball_recall": classes["ball"]["recall"],
                "ball_f1": classes["ball"]["f1"],
                "bat_box_ap50": classes["bat"]["ap50"],
                "bat_box_ap50_95": classes["bat"]["ap50_95"],
                "bat_keypoint_ap50": classes["bat"]["keypoint_ap50"],
                "bat_keypoint_ap50_95": classes["bat"]["keypoint_ap50_95"],
                "bat_precision": classes["bat"]["precision"],
                "bat_recall": classes["bat"]["recall"],
                "bat_f1": classes["bat"]["f1"],
                "macro_f1": metric["macro_f1"],
                "micro_f1": metric["micro_f1"],
                "f1_confidence_threshold": metric["f1_confidence_threshold"],
                "params": metric["params"],
                "gflops": metric["gflops"],
                "latency_ms": metric["latency_ms"],
                "peak_vram_mb": metric["peak_vram_mb"],
                "measurement_device": profile_report["device"]["name"],
                "selection_status": "pending_user_decision",
                "c_best": "",
                "quantization_eligibility": (
                    "eligible_q0_q1" if metric["candidate_id"] == "C0" else "pending_user_decision"
                ),
                "detect_physical_microbatch": record["microbatch"],
                "pose_train_batch": config.pose_batch,
                "detect_validation_batch": config.detect_val_batch,
                "pose_validation_batch": config.pose_val_batch,
                "micro_f1_method": MICRO_F1_METHOD,
                "notes": (
                    "固定20% train-only screening；formal split 未使用；"
                    f"transfer matched={transfer['matched']} "
                    f"missing={transfer['missing']} "
                    f"unexpected={transfer['unexpected']} "
                    f"shape_mismatch={transfer['shape_mismatch']}"
                ),
            }
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _markdown_report(
    config: ScreenRunConfig,
    handoff: dict[str, Any],
    records: list[dict[str, Any]],
    assessment: dict[str, Any],
    profile_report: dict[str, Any],
    git: dict[str, Any],
) -> str:
    lines = [
        "# architecture_2 Full35 J3 C0～C3 Float20 報告",
        "",
        "狀態：completed_screening；Pose=true；只代表固定20% train-only screen。",
        "",
        "## 結論摘要",
        "",
        f"- Handoff：{handoff['revision_id']}／{handoff['winner_id']}。",
        f"- Parent checkpoint SHA256：{handoff['checkpoint']['sha256']}。",
        f"- 候選：{', '.join(item['metrics']['candidate_id'] for item in records)}。",
        "- Detect 與 Pose 都完成20 epochs；formal validation split 未使用。",
        f"- Pareto front：{', '.join(assessment['pareto_front']) or '無'}。",
        "- c_best=null、selection_status=pending_user_decision；不自動量化。",
        "",
        "## 精度與 F1（各候選 best-joint-screening EMA）",
        "",
        "| 候選 | best epoch | COCO mAP50 | COCO mAP50-95 | person AP50-95 | Pose box mAP50-95 | Keypoint mAP50-95 | Macro F1 | Micro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        value = record["metrics"]
        lines.append(
            "| {candidate} | {epoch} | {coco50} | {coco} | {person} | {pose_box} | "
            "{pose} | {macro} | {micro} |".format(
                candidate=value["candidate_id"],
                epoch=record["best_joint_epoch"],
                coco50=_fmt(value["coco_box_map50"]),
                coco=_fmt(value["coco_box_map50_95"]),
                person=_fmt(value["coco_person_ap50_95"]),
                pose_box=_fmt(value["bbat5_pose_box_map50_95"]),
                pose=_fmt(value["bbat5_keypoint_map50_95"]),
                macro=_fmt(value["macro_f1"]),
                micro=_fmt(value["micro_f1"]),
            )
        )
    lines.extend(
        [
            "",
            (
                "F1 採 Pose keypoint route；Macro 是 ball／bat 未加權平均，Micro 由固定 threshold "
                "下的 curves／supports 估算。C1～C3 使用 C0 best-joint 的同一 threshold。"
            ),
            "",
            "## Ball／Bat 分類別",
            "",
            "| 候選 | 類別 | Box AP50 | Box AP50-95 | Kpt AP50 | Kpt AP50-95 | P | R | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        value = record["metrics"]
        for class_name in ("ball", "bat"):
            item = value["classes"][class_name]
            lines.append(
                "| {candidate} | {name} | {ap50} | {ap} | {kap50} | {kap} | {p} | {r} | {f1} |".format(
                    candidate=value["candidate_id"],
                    name=class_name,
                    ap50=_fmt(item["ap50"]),
                    ap=_fmt(item["ap50_95"]),
                    kap50=_fmt(item["keypoint_ap50"]),
                    kap=_fmt(item["keypoint_ap50_95"]),
                    p=_fmt(item["precision"]),
                    r=_fmt(item["recall"]),
                    f1=_fmt(item["f1"]),
                )
            )
    lines.extend(
        [
            "",
            "## 成本（batch=1、640、PyTorch AMP）",
            "",
            "| 候選 | Params | both GFLOPs | latency median ms | latency p95 ms | peak allocated MiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        value = record["metrics"]
        both = record["profile"]["tasks"]["both"]
        lines.append(
            f"| {value['candidate_id']} | {value['params']:,} | "
            f"{value['gflops']:.3f} | {value['latency_ms']:.3f} | "
            f"{both['latency_p95_ms']:.3f} | {value['peak_vram_mb']:.1f} |"
        )
    lines.extend(
        [
            "",
            (
                f"量測裝置：{profile_report['device']['name']}；warmup="
                f"{profile_report['device']['warmup']}、iterations="
                f"{profile_report['device']['iterations']}。這是 PyTorch Float/AMP profile，"
                "不是 INT8 deployment latency。"
            ),
            "",
            "## C0 差值、描述性 band 與 Pareto",
            "",
            "| 候選 | COCO Δ | Pose box Δ | Keypoint Δ | Params reduction | GFLOPs reduction | Latency reduction | 描述性 bands |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in assessment["candidates"]:
        metrics = item["metrics"]
        delta = item["metric_delta_vs_c0"]
        cost = item["cost_reduction_vs_c0"]
        lines.append(
            f"| {metrics['candidate_id']} | {delta['coco_box_map50_95']:+.6f} | "
            f"{delta['bbat5_pose_box_map50_95']:+.6f} | "
            f"{delta['bbat5_keypoint_map50_95']:+.6f} | "
            f"{cost['params']:.2%} | {cost['gflops']:.2%} | "
            f"{cost['latency_ms']:.2%} | "
            f"{json.dumps(item['descriptive_bands'], ensure_ascii=False)} |"
        )
    lines.extend(
        [
            "",
            "0.005／0.008 只作描述性敏感度，不是 PASS／REJECT gate。Pareto 也不替使用者選 C_best。",
            "",
            "## Architecture Delta 與權重 transfer",
            "",
            "| 候選 | 唯一變因 | 修改路徑 | matched | missing | unexpected | shape mismatch |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        build = record["build_report"]["candidate_build"]
        summary = record["build_report"]["summary"]
        lines.append(
            f"| {record['metrics']['candidate_id']} | "
            f"{', '.join(build['changed_fields']) or '無'} | "
            f"{', '.join(build['changed_module_paths']) or '無'} | "
            f"{summary['matched']} | {summary['missing']} | "
            f"{summary['unexpected']} | {summary['shape_mismatch']} |"
        )
    lines.extend(
        [
            "",
            (
                "完整 matched／missing／unexpected／shape-mismatch tensor 清單與 "
                "run-manifest hash 見 transfer-reports.json。"
            ),
            "",
            "## Lineage 與限制",
            "",
            f"- spec：{SPEC_VERSION}／{file_sha256(SPEC_PATH)}。",
            f"- training YAML SHA256：{file_sha256(config.path)}。",
            f"- git revision：{git['revision']}；working tree dirty={git['working_tree_dirty']}。",
            "- COCO 23,657 train／5,000 train-only search-val；BBAT5 1,073 train／600 search-val。",
            "- 本輪不使用 COCO val2017 或 BBAT5 formal val，不足以直接定稿正式部署 winner。",
            "- uniform OKS sigma 未針對棒球兩點任務校準。",
            "- 下一步由使用者選擇 C_best、量化候選、額外 seed 或完整資料驗證。",
            "",
        ]
    )
    return "\n".join(lines)


def _figures(
    output: Path,
    records: list[dict[str, Any]],
    assessment: dict[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    ids = [item["metrics"]["candidate_id"] for item in records]
    x = list(range(len(ids)))

    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.25
    axis.bar(
        [value - width for value in x],
        [item["metrics"]["coco_box_map50_95"] for item in records],
        width,
        label="COCO box mAP50-95",
    )
    axis.bar(
        x,
        [item["metrics"]["bbat5_pose_box_map50_95"] for item in records],
        width,
        label="BBAT5 Pose box mAP50-95",
    )
    axis.bar(
        [value + width for value in x],
        [item["metrics"]["bbat5_keypoint_map50_95"] for item in records],
        width,
        label="BBAT5 keypoint mAP50-95",
    )
    axis.set_xticks(x, ids)
    axis.set_ylim(0, 1)
    axis.set_ylabel("mAP")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    accuracy_path = figure_root / "01-accuracy-comparison.png"
    figure.savefig(accuracy_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(ids, [item["metrics"]["params"] / 1e6 for item in records])
    axes[0].set_ylabel("Parameters (M)")
    axes[1].bar(ids, [item["metrics"]["latency_ms"] for item in records])
    axes[1].set_ylabel("Both-route median latency (ms)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    cost_path = figure_root / "02-cost-comparison.png"
    figure.savefig(cost_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    pareto = set(assessment["pareto_front"])
    for record in records:
        value = record["metrics"]
        score = (value["coco_box_map50_95"] + 0.25 * value["bbat5_keypoint_map50_95"]) / 1.25
        axis.scatter(
            value["gflops"],
            score,
            s=90,
            marker="o" if value["candidate_id"] in pareto else "x",
        )
        axis.annotate(value["candidate_id"], (value["gflops"], score))
    axis.set_xlabel("Both-route GFLOPs")
    axis.set_ylabel("Joint screening score")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    pareto_path = figure_root / "03-pareto.png"
    figure.savefig(pareto_path, dpi=160)
    plt.close(figure)
    return [str(accuracy_path), str(cost_path), str(pareto_path)]


def export_float20_results(
    config_path: str | Path,
    *,
    profiles_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """完整 matrix/profile 完成後，一次產生所有正式 screening 交付。"""

    config = ScreenRunConfig.load(config_path)
    matrix_path = config.run_root / "matrix-complete.json"
    if not matrix_path.is_file():
        raise FileNotFoundError("matrix-complete.json 尚未產生")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if (
        matrix.get("status") != "completed_screening_matrix"
        or matrix.get("selection_status") != "pending_user_decision"
        or matrix.get("c_best") is not None
    ):
        raise ValueError("matrix completion／selection contract 漂移")
    profile_path = (
        Path(profiles_path).expanduser().resolve()
        if profiles_path is not None
        else config.run_root / "profiles/cost-profiles.json"
    )
    profile_report, profile_by_candidate = _profile_index(profile_path, config)
    handoff = json.loads(config.handoff_manifest.read_text(encoding="utf-8"))
    threshold_path = config.run_root / "shared-controls/c0-f1-thresholds.json"
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    fixed_threshold = float(threshold["thresholds"]["pose_keypoints"])
    batch_plan = json.loads((config.run_root / "shared-controls/batch-plan.json").read_text(encoding="utf-8"))
    microbatch = int(batch_plan["selected_detect_microbatch"])

    records: list[dict[str, Any]] = []
    metric_objects: list[CandidateMetrics] = []
    for candidate in config.candidates:
        complete, validation, metrics_path = _best_validation(config, candidate)
        build_report = _candidate_build_report(config, candidate, complete)
        profile = profile_by_candidate[candidate]
        metrics = _candidate_metrics(candidate, validation, profile)
        if abs(metrics.f1_confidence_threshold - fixed_threshold) > 1e-12:
            raise ValueError(f"{candidate} 沒有使用 C0 best-joint fixed threshold")
        metric_objects.append(metrics)
        records.append(
            {
                "metrics": asdict(metrics),
                "complete": complete,
                "best_joint_epoch": int(complete["best_state"]["joint_screening"]["epoch"]),
                "best_state": complete["best_state"],
                "validation_metrics": str(metrics_path),
                "validation_metrics_sha256": file_sha256(metrics_path),
                "profile": profile,
                "lineage": complete["lineage"],
                "build_report": build_report,
                "microbatch": microbatch,
            }
        )
    assessment = evaluate_float_results(metric_objects).to_dict()
    git = _git_state()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else PROJECT_ROOT / "results/full35-j3-float20-seed0"
    )
    destination.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "schema_version": 1,
        "status": "completed_screening",
        "screening_only": True,
        "formal_split_used": False,
        "pose_executed": True,
        "handoff_revision": handoff["revision_id"],
        "winner_id": handoff["winner_id"],
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "training_yaml_sha256": file_sha256(config.path),
        "profiles_sha256": file_sha256(profile_path),
        "fixed_f1_thresholds": threshold,
        "batch_plan": batch_plan,
        "candidates": records,
    }
    _atomic_json(destination / "metrics.json", metrics_payload)
    transfer_payload = {
        "schema_version": 1,
        "status": "completed",
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "training_yaml_sha256": file_sha256(config.path),
        "candidates": {item["metrics"]["candidate_id"]: item["build_report"] for item in records},
    }
    transfer_path = destination / "transfer-reports.json"
    _atomic_json(transfer_path, transfer_payload)
    _atomic_json(destination / "selection.json", assessment)
    lineage = {
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "handoff": handoff,
        "training_yaml": str(config.path),
        "training_yaml_sha256": file_sha256(config.path),
        "matrix": str(matrix_path),
        "matrix_sha256": file_sha256(matrix_path),
        "profiles": str(profile_path),
        "profiles_sha256": file_sha256(profile_path),
        "transfer_reports": str(transfer_path),
        "transfer_reports_sha256": file_sha256(transfer_path),
        "git": git,
        "candidate_lineage": {item["metrics"]["candidate_id"]: item["lineage"] for item in records},
    }
    _atomic_json(destination / "lineage.json", lineage)
    rows = _csv_rows(
        config,
        handoff,
        records,
        profile_report=profile_report,
        git=git,
    )
    _write_csv(destination / "metrics.csv", rows)
    figures = _figures(destination, records, assessment)
    _atomic_text(
        destination / "REPORT.md",
        _markdown_report(
            config,
            handoff,
            records,
            assessment,
            profile_report,
            git,
        ),
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "selection_status": "pending_user_decision",
        "c_best": None,
        "output_dir": str(destination),
        "files": {
            str(path.relative_to(destination)): file_sha256(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
        "figures": [str(Path(path).relative_to(destination)) for path in figures],
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest
