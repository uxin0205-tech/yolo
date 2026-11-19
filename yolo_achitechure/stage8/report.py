"""五組驗證完成後交付、整理小型證據，停止於 Stage8。"""

import csv
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .common import (
    CANDIDATES,
    HERE,
    ROOT,
    baseline,
    passed,
    protocol,
    score,
    sha256,
    write_json,
)


def original_gate(values):
    # optimize 原八項：COCO overall/person，加 BBAT overall/ball/bat box/pose。
    keys = (
        "coco_AP",
        "person_AP",
        "box_AP50_95",
        "pose_AP50_95",
        "ball_box_AP50_95",
        "ball_pose_AP50_95",
        "bat_box_AP50_95",
        "bat_pose_AP50_95",
    )
    base = baseline()
    return all(values[k] >= base[k] - 0.001 for k in keys) and (
        values["coco_AP"] >= base["coco_AP"] + 0.001
        or values["pose_AP50_95"] >= base["pose_AP50_95"] + 0.002
    )


def finalize(root, zeros, trained, selected, verified):
    from .packaging import bundle
    from .redraw import draw

    root = Path(root)
    resultdir = ROOT / protocol()["result_dir"]
    resultdir.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(root / "source.json")
    winner = max(
        CANDIDATES,
        key=lambda n: (passed(selected[n]["metrics"]), score(selected[n]["metrics"])),
    )
    recommend = (
        winner
        if passed(selected[winner]["metrics"])
        and score(selected[winner]["metrics"])
        >= score(baseline()) + protocol()["selection"]["promote_min_score_gain"]
        else "parent"
    )
    previous = protocol()["previous_result"]
    if passed(previous["metrics"]) and score(previous["metrics"]) >= score(baseline()) + protocol()["selection"]["promote_min_score_gain"]:
        if recommend == "parent" or score(previous["metrics"]) >= score(selected[winner]["metrics"]):
            recommend = "previous_rep17_20"
    entries = {
        n: {**verified[n], "balanced_score": score(selected[n]["metrics"])}
        for n in CANDIDATES
    }
    packaged = bundle(resultdir / "handoff", entries, recommend)
    candidates = {}
    multibranch = resultdir / "training"
    multibranch.mkdir(exist_ok=True)
    for name in CANDIDATES:
        source = Path(selected[name]["checkpoint"])
        target = multibranch / f"{name}-best-ema.pt"
        if not target.exists():
            source.rename(target)
            source.with_suffix(".json").rename(target.with_suffix(".json"))
            source.symlink_to(os.path.relpath(target, source.parent))
            source.with_suffix(".json").symlink_to(
                os.path.relpath(target.with_suffix(".json"), source.parent)
            )
        if sha256(target) != selected[name]["sha256"]:
            raise ValueError("多分支 EMA 搬移 SHA 改變")
        candidates[name] = {
            "best": selected[name],
            "training_checkpoint": str(target),
            "deploy": packaged["models"][name],
            "resume": [r["resume"] for r in trained[name]],
            "epochs": sum(len(r["history"]) for r in trained[name]),
            "relative_gate": passed(selected[name]["metrics"]),
            "reference_engineering_gate": original_gate(selected[name]["metrics"]),
            "original_independent_gate": verified[name]["original_independent_gate"],
            "float_reference_metrics": verified[name]["float_reference_metrics"],
        }
    result = {
        "status": "complete",
        "stage": 8,
        "completed_at": time.time(),
        "recommendation": recommend,
        "best_rep": winner,
        "models": candidates,
        "parent": protocol()["parent"],
        "previous_result": protocol()["previous_result"],
        "training_operators": protocol()["candidate_specs"],
        "selection_operators": "hardware qSiLU / D1",
        "training_metrics": {n: verified[n]["training_metrics"] for n in CANDIDATES},
        "deployment_minus_training": {n: verified[n]["deployment_minus_training"] for n in CANDIDATES},
        "source_lock_sha256": source_digest,
        "portable_validation": packaged["validation"],
        "next_stage_started": False,
        "training_statuses": {n: [r["status"] for r in trained[n]] for n in CANDIDATES},
        "limitation": "單 seed；未做 RTL/HLS／板端量測；其餘浮點與 head sigmoid 詳見算子盤點",
    }
    write_json(resultdir / "handoff.json", result)
    curves = {
        n: [
            [
                {**row, "balanced_score": score(row["metrics"])}
                for row in segment["history"]
            ]
            for segment in trained[n]
        ]
        for n in CANDIDATES
    }
    write_json(resultdir / "curves.json", curves)
    with (resultdir / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["candidate", *baseline(), "balanced_score", "relative_gate"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate": "parent",
                **baseline(),
                "balanced_score": score(baseline()),
                "relative_gate": True,
            }
        )
        writer.writerow({"candidate": "previous_rep17_20", **previous["metrics"], "balanced_score": score(previous["metrics"]), "relative_gate": passed(previous["metrics"])})
        for n in CANDIDATES:
            writer.writerow(
                {
                    "candidate": n,
                    **selected[n]["metrics"],
                    "balanced_score": score(selected[n]["metrics"]),
                    "relative_gate": passed(selected[n]["metrics"]),
                }
            )
    write_json(resultdir / "zero-evaluations.json", zeros)
    write_json(resultdir / "folding-and-verification.json", verified)
    for filename in (
        "protocol.json",
        "reference-review.json",
        "cpu-contracts.json",
        "cleanup-20261107.json",
    ):
        shutil.copyfile(HERE / filename, resultdir / filename)
    shutil.copyfile(HERE / "redraw.py", resultdir / "redraw.py")
    draw(resultdir)
    evidence = resultdir / "evidence"
    if not evidence.exists():
        evidence.symlink_to(os.path.relpath(root, resultdir), target_is_directory=True)
    link = ROOT / "yolo_achitechure/handoff"
    main_handoff = (resultdir / "handoff" if recommend in CANDIDATES else
                    ROOT / ("yolo_achitechure/result/stage8/handoff" if recommend == "previous_rep17_20" else "yolo_activation/result/stage7/handoff"))
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(os.path.relpath(main_handoff, link.parent), target_is_directory=True)
    lines = [
        "# Stage8 原生 RepConv 訓練與硬體部署結果",
        "",
        "訓練使用 Float PWL，四組保留 qSiLU，一組僅 Rep 使用 SiLU；表中 AP 全為折疊且轉回 qSiLU／D1 的部署結果。兩種算子的 AP 差距記錄於 handoff.json。",
        "",
        f"研究建議：**{recommend}**；五組最佳 Rep：`{winner}`。",
        "",
        "若建議為 parent／previous_rep17_20，維持既有較佳模型；根層 handoff 指向真正選定結果。採用規則在訓練前固定，未強制選 Rep。",
        "",
        "| 模型 | COCO AP | 球 AP | 球棒 AP | Pose box AP | Pose AP | 輪數 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, values, epochs in [
        ("parent", baseline(), 0),
        ("previous_rep17_20", previous["metrics"], 0),
        *[(n, selected[n]["metrics"], candidates[n]["epochs"]) for n in CANDIDATES],
    ]:
        lines.append(
            f"| {n} | {values['coco_AP']:.6f} | {values['sports_ball_AP']:.6f} | {values['baseball_bat_AP']:.6f} | {values['box_AP50_95']:.6f} | {values['pose_AP50_95']:.6f} | {epochs} |"
        )
    lines.extend(
        [
            "",
            "完整十項 AP 見 [metrics.csv](metrics.csv)，完整指標／血緣見 [handoff.json](handoff.json)。",
            "",
            (
                "五組從同一份 qSiLU E8 獨立開始；保留 B2 sign＋Hadamard、S1 Q10／signed16 bias、D1 整數 PWL exp＋LUT256 reciprocal。"
                "Rep／ACB 部署都是單 Conv3×3+bias → qSiLU，沒有分支或 BN；其他原生 ConvBN、head sigmoid 與浮點路徑詳見 evidence/verify-*/operators.json。"
            ),
            "",
            (
                "[可攜推論包](handoff/README.md) 已用獨立 CPU 程序比對 160／640 完整輸出。training/ 是最佳多分支 EMA；"
                "含 optimizer、EMA、scheduler、RNG 的 resume 位於 handoff.json 指定位置，兩種用途分開。"
            ),
            "",
            (
                "完整 D1 部署 AP 與 Float PWL 診斷分列；沒有將 Float 診斷路徑寫入部署模型。"
                "所有訓練均使用同一 BBAT5 v1，未重切資料。"
            ),
            "",
            "可用 `python redraw.py` 重畫 accuracy／curves，不需 GPU 或 checkpoint。CPU smoke、GPU 訓練、完整 AP、定點 reference 與未執行的硬體綜合／實板量測分開看待。",
            "",
            (
                "限制：單 seed、validation 選型，無獨立 test 與硬體量測；本輪沒有未改方法的 control 加訓。"
                "第九階段未啟動，長方形／二次冪特徵圖分析仍待指定。"
            ),
        ]
    )
    (resultdir / "README.md").write_text("\n".join(lines) + "\n")
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")
    marker = "## 自動完成紀錄"
    worklog = ROOT / "docs/worklogs/2026-11-07-repconv-native-training.md"
    existing = worklog.read_text().split(marker)[0]
    worklog.write_text(
        existing
        + f"\n{marker}\n\n{stamp}：五組完成，建議 `{recommend}`，最佳 Rep `{winner}`。"
        "folding、重載完整 AP 與可攜 CPU 驗證均通過；詳見 [正式結果](../../yolo_achitechure/result/stage8-retrain/README.md)。Stage9 未啟動。\n"
    )
    for filename in ("README.md", "docs/research/yolo-research-plan-2026-09-21.md"):
        path = ROOT / filename
        text = path.read_text().replace(
            "Stage8 新輪已啟動：五組 Rep／ACB 訓練，硬體部署 AP 驗收。",
            f"Stage8 已完成，研究建議 {recommend}；Stage9 待指定。",
        )
        text = text.replace(
            "| 8 | RepConv 訓練與硬體部署：R1～R3 | 新輪 queue 執行中 |",
            "| 8 | RepConv 訓練與硬體部署：R1～R3 | 新輪已完成，見 result/stage8-retrain |",
        )
        path.write_text(text)
    return result
