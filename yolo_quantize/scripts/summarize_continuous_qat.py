"""只彙整完成 QAT 的十六項搜尋指標，不使用 formal、不變更 queue。"""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(metrics, accepted, ptq):
    keys = sorted(k for k in accepted if k.endswith(("/map50", "/map50_95")))
    if len(keys) != 16 or any(
        not math.isfinite(source[k]) or not 0 <= source[k] <= 1
        for source in (metrics, accepted, ptq)
        for k in keys
    ):
        raise ValueError("必須具有十六項有效精度指標")
    total = {k: metrics[k] - accepted[k] for k in keys}
    worst = {
        suffix: min(total[k] for k in keys if k.endswith("/" + suffix))
        for suffix in ("map50", "map50_95")
    }
    return {
        "metrics": {k: metrics[k] for k in keys},
        "total_deltas": total,
        "qat_minus_ptq": {k: metrics[k] - ptq[k] for k in keys},
        "worst_total_deltas": worst,
        "passes_total_gate": worst["map50"] >= -0.015 and worst["map50_95"] >= -0.04,
    }


def main():
    queue_path = OUT / "selected-qat-jobs-v2.json"
    queue = json.loads(queue_path.read_text())
    ptq_path = OUT / "isolated-special-report.json"
    ptq_results = json.loads(ptq_path.read_text())["results"]
    jobs = []
    for job in queue["jobs"]:
        plan_path = Path(job["plan"]["path"])
        if record(plan_path) != job["plan"]:
            raise ValueError("queue plan hash 不一致")
        plan = json.loads(plan_path.read_text())
        run = (
            ROOT
            / plan["run_root"]
            / f"{plan['plan_id']}-qat-seed{plan['training']['seed']}"
        )
        completion_path = run / "qat-experiment.json"
        if not completion_path.exists():
            continue
        completion = json.loads(completion_path.read_text())
        if (
            completion["plan_sha256"] != job["plan"]["sha256"]
            or completion["epochs_completed"] != 5
        ):
            raise ValueError("QAT 完成證據不一致")
        accepted_path = Path(plan["sources"]["accepted_metrics"]["path"])
        if record(accepted_path) != plan["sources"]["accepted_metrics"]:
            raise ValueError("accepted metrics hash 不一致")
        accepted = json.loads(accepted_path.read_text())["metrics"]
        ptq = ptq_results[job["candidate_id"]]["all_search_metrics"]
        epochs = []
        for index in range(5):
            path = run / f"validation/epoch-{index:04d}/bittrue/metrics.json"
            metrics = json.loads(path.read_text())["metrics"]
            epochs.append(
                {
                    "epoch": index + 1,
                    "source": record(path),
                    **compare(metrics, accepted, ptq),
                }
            )
        jobs.append(
            {
                "job_id": job["job_id"],
                "completion": record(completion_path),
                "ptq": compare(ptq, accepted, ptq),
                "epochs": epochs,
            }
        )
    payload = {
        "status": "completed" if len(jobs) == len(queue["jobs"]) else "partial",
        "queue": record(queue_path),
        "ptq_source": record(ptq_path),
        "formal_validation": False,
        "jobs": jobs,
    }
    (OUT / "qat-recovery-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 同 parent 短 QAT 恢復結果（持續更新）",
        "",
        f"已完成 {len(jobs)}/{len(queue['jobs'])} 組；僅搜尋驗證，非 formal 或重新 export 驗證。",
        "",
        "所有下降為相對 accepted 模型的絕對百分點，包含 activation 替換；16 項全過才達標。",
        "PTQ 未達標不代表 QAT 無效；回升幅度須逐項比，不能相減不同指標的最差值當作恢復量。",
        "",
        "| 實驗 | 回合 | 最差 mAP50 下降（pp） | 最差 mAP50–95 下降（pp） | 搜尋雙門檻 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for job in jobs:
        for epoch in job["epochs"]:
            worst = epoch["worst_total_deltas"]
            lines.append(
                f"| {job['job_id']} | {epoch['epoch']} | {-100 * worst['map50']:.3f} | {-100 * worst['map50_95']:.3f} | {'通過' if epoch['passes_total_gate'] else '未通過'} |"
            )
    for job in jobs:
        lines.extend(
            [
                "",
                f"## {job['job_id']}：第 5 回合對 PTQ",
                "",
                "| 指標 | PTQ（%） | QAT（%） | QAT − PTQ（pp） |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        final = job["epochs"][-1]
        for key, value in final["metrics"].items():
            lines.append(
                f"| {key} | {100 * job['ptq']['metrics'][key]:.3f} | {100 * value:.3f} | {100 * final['qat_minus_ptq'][key]:+.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "完整五回合逐項數據及來源 SHA-256：`artifacts/queues/full-model-continuous-0907/qat-recovery-summary.json`。",
            "",
            "外部 sham 僅歷史參考，不宣稱同 parent 的 sham 因果對照。六組完成後才選累積混合配置；不得直接拼接各區獨立最優。",
        ]
    )
    (ROOT / "docs/reports/2026-09-07-continuous-qat-recovery-results.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(json.dumps({"completed_jobs": len(jobs), "status": payload["status"]}))


if __name__ == "__main__":
    main()
