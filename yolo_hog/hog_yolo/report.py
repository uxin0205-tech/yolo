"""僅用小型數據重畫；正式採用與候選保留分開。"""

import csv
import shutil
from pathlib import Path

from yolo_attention.stage3_kd_training import metric_vector

from .common import PROJECT, ROOT, read, review, sha256, write_json


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    result = read(output / "training/result.json")
    history = result["history"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in ["AP50_95", "sports_ball_AP50_95", "baseball_bat_AP50_95"]:
        ax.plot(
            [h["epoch"] for h in history],
            [metric_vector(h["evaluation"])[key] for h in history],
            marker=".",
            label=key,
        )
    ax.set(xlabel="HOG adaptation epoch", ylabel="official AP50-95")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "curves.svg")
    plt.close(fig)


def summarize(output):
    output = Path(output)
    data = review()
    result = read(output / "training/result.json")
    baseline = metric_vector(read(data["parent_metrics"]))
    current = metric_vector(read(output / "verification/metrics.json"))
    candidate = output / "verification/deployment.pt"
    safe = all(current[k] >= baseline[k] - 0.001 for k in baseline)
    adopt = current["AP50_95"] > baseline["AP50_95"] and safe
    destination = PROJECT / "result/stage5"
    destination.mkdir(parents=True, exist_ok=False)
    delivered = destination / "candidate.pt"
    shutil.move(candidate, delivered)
    shutil.move(candidate.with_suffix(".json"), delivered.with_suffix(".json"))
    candidate.symlink_to(delivered)
    candidate.with_suffix(".json").symlink_to(delivered.with_suffix(".json"))
    handoff = {
        "schema": "hog-stage5-handoff-v1",
        "adopt_hog_recommendation": adopt,
        "checkpoint": str(delivered) if adopt else data["parent"],
        "sha256": sha256(delivered) if adopt else data["parent_sha256"],
        "candidate": str(delivered),
        "candidate_sha256": sha256(delivered),
        "parent": data["parent"],
        "parent_sha256": data["parent_sha256"],
        "parent_metrics": baseline,
        "candidate_metrics": current,
        "class_safety": safe,
        "best_epoch": result["best"]["epoch"],
        "completed_epochs": len(result["history"]) - 1,
        "stop_reason": result["stop_reason"],
        "source_run": str(output),
        "resume": str(output / "training/resume.pt"),
        "resume_sha256": sha256(output / "training/resume.pt"),
        "no_hog_inference_head": True,
        "masf_deferred_until_after_combine": True,
        "next_stage_started": False,
        "limitation": "single seed, no unchanged control training; cannot isolate HOG causal benefit",
    }
    write_json(destination / "handoff.json", handoff)
    write_json(output / "comparison.json", {"ready": True, **handoff})
    rows = [
        {"epoch": h["epoch"], **metric_vector(h["evaluation"])}
        for h in result["history"]
    ]
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(output)
    lines = [
        "# 第五階段 HOG 結果",
        "",
        "| 指標 | Parent | 最佳候選 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for key, value in baseline.items():
        lines.append(
            f"| {key} | {value:.9f} | {current[key]:.9f} | {current[key] - value:+.9f} |"
        )
    lines += [
        "",
        (
            f"建議採用 HOG 適應候選：{adopt}。最佳 epoch {result['best']['epoch']}，"
            f"實際完成 {len(result['history']) - 1} 輪。沒有未改方法 control 加訓，不能把全部差值歸因 HOG。"
        ),
        "",
        (
            "train-only HOG head 已移除，CPU FP32 輸出逐位等價，正式 COCO val5000 重載一致。"
            "B2／S1／完整 PWL exp／D1 reciprocal 與 activation 契約保留，部署不新增 HOG 運算。"
            "coverage 的 H4 提案另列，不自動啟動第二候選、Pose 或融合訓練。"
        ),
        "",
        "模型與完整 SHA 見 result/stage5/handoff.json；第六階段待使用者指定。",
    ]
    report = "\n".join(lines) + "\n"
    (output / "RESULTS.md").write_text(report)
    (destination / "README.md").write_text(
        "# 第五階段交接\n\n"
        + f"建議採用 HOG 適應候選：{adopt}。權重、八項 AP 與 parent／resume SHA 見 [handoff](handoff.json)。"
        "訓練輔助頭已移除，第六階段待指定。\n"
    )
    with (ROOT / "docs/worklogs/2026-10-21-hog-stage5.md").open("a") as f:
        f.write("\n\n## 背景流程完成\n\n" + report)
    removed = []
    for p in (output / "smoke").glob("*.pt"):
        removed.append({"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size})
        p.unlink()
    write_json(output / "smoke-cleanup.json", {"removed": removed})
