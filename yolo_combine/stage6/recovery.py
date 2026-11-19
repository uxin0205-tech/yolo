"""有界的恢復候選 queue；完成後重驗、比較，仍停留第六階段。"""

from pathlib import Path

from .common import sha256, write_json
from .evaluate import COCO_KEYS, accepted, expected
from .policy import selection_score


def run_recovery(job, root, protocol):
    kind = protocol["kind"]
    if kind not in ("stage6-kd-recovery-v1", "stage6-pose-head-recovery-v1"):
        raise ValueError("unknown recovery protocol")
    parent, recipe = protocol["parent"], protocol["recipe"]
    calibration = None
    if kind == "stage6-kd-recovery-v1":
        probe = job(
            "kd-calibration",
            "calibrate-kd",
            parent=parent,
            recipe=recipe,
            kd_spec=protocol["kd"],
        )
        if probe["status"] != "passed":
            raise ValueError("KD calibration/smoke requires review")
        recipe = probe["recipe"]
        calibration = job("task-calibration", "calibrate", parent=parent, recipe=recipe)
        if calibration["status"] != "passed":
            raise ValueError("KD task calibration requires review")
    else:
        if recipe["task"] != "pose" or recipe["scope"] != "head" or "kd" in recipe:
            raise ValueError(
                "head recovery must freeze shared/Detect and use native Pose loss"
            )
        probe = job("smoke", "smoke-head-recovery", parent=parent, recipe=recipe)
        if probe["status"] != "passed":
            raise ValueError("Pose head recovery smoke requires review")
    trained = job(
        "train", "train", parent=parent, recipe=recipe, calibration=calibration
    )
    verified = job("verification", "verify-recovery", parent=trained["best"])
    base = expected()
    result = {
        "status": "candidate_complete"
        if trained["status"] == "complete"
        else "needs_review",
        "stage": 6,
        "stage6_complete": False,
        "next_stage_started": False,
        "best": trained["best"],
        "parent": parent,
        "recipe": recipe,
        "reason": trained["reason"],
        "verification": verified,
        "accepted": accepted(verified["metrics"]),
        "improved_balanced_score": selection_score(verified["metrics"], recipe)
        > selection_score(parent["metrics"], recipe),
        "deltas_from_parent": {
            k: v - parent["metrics"][k] for k, v in verified["metrics"].items()
        },
        "deltas_from_original": {
            k: v - base[k] for k, v in verified["metrics"].items()
        },
        "remaining_target_deficits": {
            k: max(0.0, base[k] - (0.005 if k in COCO_KEYS else 0.02) - v)
            for k, v in verified["metrics"].items()
        },
        "epochs_completed": len(trained["history"]),
        "retained_epoch_candidates": [
            h["candidate_checkpoint"]
            for h in trained["history"]
            if h["candidate_checkpoint"] and Path(h["candidate_checkpoint"]).is_file()
        ],
        "interpretation": "這是本輪恢復候選的結果，不宣稱整個第六階段已完成；保留各項精度取捨供接續分析。",
    }
    write_json(Path(root) / "result.json", result)
    lines = [
        f"# 第六階段 {recipe['name']} 恢復結果",
        "",
        result["interpretation"],
        "",
        f"完成 {result['epochs_completed']} 輪；停止原因：{result['reason']}。",
        f"完整重載驗證一致；原始精度目標通過：{result['accepted']}。",
        "",
        "| 指標 | 原獨立模型 | 本輪起點 | 本輪選定 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {k} | {base[k]:.6f} | {parent['metrics'][k]:.6f} | {v:.6f} |"
        for k, v in verified["metrics"].items()
    )
    lines.extend(
        [
            "",
            "Binary QK、Q10、完整 D1 PWL／reciprocal 契約不變，教師不進部署。",
            "單 seed、連續適應研究；本輪增益不能證明各方法的獨立因果貢獻。",
            "",
        ]
    )
    (Path(root) / "report.md").write_text("\n".join(lines))
    return result


def verify(output, parent):
    from . import checkpoint
    from .evaluate import pair

    if sha256(parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("recovery checkpoint SHA changed")
    model = checkpoint.load(parent["checkpoint"]).cuda()
    measured = pair(model, Path(output) / "evaluation", progress_root=output)["metrics"]
    if measured != parent["metrics"]:
        raise ValueError("reloaded recovery full AP differs")
    result = {
        "status": "passed",
        "metrics": measured,
        "sha256": parent["sha256"],
        "full_ap_reproduced": True,
        "teacher_exported": False,
    }
    write_json(Path(output) / "result.json", result)
    return result
