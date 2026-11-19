"""獨立比較結論與 qSiLU 接續模型；小型數據可無 GPU 重畫。"""

import csv
import os
import shutil
from pathlib import Path

from .common import HERE, ROOT, baseline, passed, rank, read, sha256, write_json


def finalize(root, zero, training, sensitivity, selected, verifications):
    root = Path(root)
    output = ROOT / "yolo_activation/result/stage7"
    output.mkdir(parents=True, exist_ok=True)
    winner = max(selected, key=lambda n: rank(selected[n]["metrics"]))
    qsilu = selected["qsilu_pq"]
    ready = passed(qsilu["metrics"]) and training["qsilu_pq"]["status"] == "complete"
    rows = []
    for name, value in selected.items():
        rows.append(
            {
                "candidate": name,
                "phase": "zero-shot",
                **zero[name]["metrics"],
                "activation_gate": passed(zero[name]["metrics"]),
            }
        )
        rows.append(
            {
                "candidate": name,
                "phase": "selected",
                **value["metrics"],
                "activation_gate": passed(value["metrics"]),
            }
        )
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve = []
    for name, run in training.items():
        curve.extend(
            {
                "candidate": name,
                "epoch": row["epoch"],
                "seconds": row["seconds"],
                **row["metrics"],
            }
            for row in run.get("history", [])
        )
    write_json(output / "curves.json", curve)
    write_json(output / "sensitivity.json", sensitivity)
    write_json(output / "baseline.json", baseline())
    write_json(output / "comparison.json", rows)
    handoff_models = {}
    for name, value in selected.items():
        if verifications[name]["status"] != "passed":
            raise ValueError("不得交付未驗證模型")
        source = Path(value["checkpoint"])
        if sha256(source) != value["sha256"]:
            raise ValueError("交付 checkpoint SHA 改變")
        destination = output / "candidates" / f"{name}.pt"
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            if sha256(destination) != value["sha256"]:
                raise ValueError("拒絕覆寫不同交付權重")
        else:
            source.rename(destination)
            source.with_suffix(".json").rename(destination.with_suffix(".json"))
            source.symlink_to(os.path.relpath(destination, source.parent))
            source.with_suffix(".json").symlink_to(
                os.path.relpath(destination.with_suffix(".json"), source.parent)
            )
        if sha256(destination) != value["sha256"]:
            raise ValueError("搬移後 SHA 不符")
        from .model import load, manifest_for

        reloaded = load(destination)
        manifest_for(reloaded)
        del reloaded
        handoff_models[name] = {
            **value,
            "checkpoint": str(destination),
            "verification": str(root / f"verify-{name}/result.json"),
        }
    for name in (
        "protocol.json",
        "reference-review.json",
        "numerics.json",
        "cpu-contracts.json",
        "cleanup-20261102.json",
    ):
        shutil.copyfile(HERE / name, output / name)
    shutil.copyfile(HERE / "redraw.py", output / "redraw.py")
    result = {
        "status": "complete" if ready else "needs_review",
        "comparison_winner": winner,
        "repconv_parent": handoff_models["qsilu_pq"] if ready else None,
        "qsilu_candidate": handoff_models["qsilu_pq"],
        "models": handoff_models,
        "parent": read(HERE / "protocol.json")["parent"],
        "stage": 7,
        "next_stage_started": False,
        "hardware_contract_preserved": True,
        "repconv_installed": False,
        "original_independent_gate": {
            n: verifications[n]["original_independent_gate"] for n in selected
        },
        "limits": read(HERE / "protocol.json")["limits"],
        "source_lock": str(root / "source.json"),
    }
    write_json(output / "handoff.json", result)
    status = (
        "已通過 activation 相對門檻，可作下一階段共同 parent"
        if ready
        else "仍有未解精度或訓練診斷，須先檢視；不宣稱可直接接續"
    )
    lines = [
        "# 第 7 階段結果：Activation 與 qSiLU 主線",
        "",
        f"比較選定：`{winner}`。qSiLU 主線：{status}。",
        "",
        "所有候選從同一 Stage6 E1 出發。完整 COCO 5000／BBAT5 v1 683、FP32 E2E；不新增 SiLU control 訓練。",
        "比較 winner 與 RepConv parent 分開；RepConv 尚未開始。原獨立模型目標另外報告，activation 門檻通過不代表原目標皆恢復。",
        "",
        "| Activation | COCO AP | BBAT box AP | BBAT pose AP | 相對門檻 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, value in selected.items():
        m = value["metrics"]
        lines.append(
            f"| {name} | {m['coco_AP']:.6f} | {m['box_AP50_95']:.6f} | {m['pose_AP50_95']:.6f} | {'通過' if passed(m) else '未通過'} |"
        )
    lines.extend(
        [
            "",
            "完整十項 AP：`comparison.csv`；逐輪：`curves.json`；區域敏感度：`sensitivity.json`；設定／來源：`protocol.json`。",
            "重畫：`.venv/bin/python yolo_activation/result/stage7/redraw.py`。不需要 GPU 或 checkpoint。",
            "",
            "保留 B2／S1 Q10／D1 整數 PWL exp 與 LUT256 reciprocal。activation 固定係數可映射硬體，但任務模型 activation 仍為浮點參考；不是整網定點或硬體實測。",
            "Hardswish 固定 1/6 乘數、PolyQuality 一般固定乘數、PolyShift 四次資料乘法、qSiLU 一次共用平方等成本見 numerics.json。",
            "單 seed 與原四項融合目標缺口如實保留；後續矩形／2 冪次 feature shape 分析仍在最終階段。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    from .redraw import redraw

    redraw(output)
    # Completed resumes are replaceable optimizer state; preserve only qSiLU resume
    # if adaptation remains unresolved. Keep all small metrics and failure evidence.
    deleted = []
    for path in root.glob("train-*/resume.pt"):
        if not ready and "qsilu" in path.parent.name:
            continue
        deleted.append({"path": str(path), "bytes": path.stat().st_size})
        path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)
    write_json(
        output / "completion-cleanup.json",
        {
            "removed": deleted,
            "reason": "完整重載驗證後刪除不需續跑的 optimizer resume；未解 qSiLU 保留",
        },
    )
    worklog = ROOT / "docs/worklogs/2026-11-02-activation-stage7.md"
    with worklog.open("a") as stream:
        stream.write(
            f"\n## Queue 完成\n\n比較選定 `{winner}`；qSiLU 狀態：{status}。完整結果見 [Stage7 結果](../../yolo_activation/result/stage7/README.md)。所有完成候選經重載完整 AP 核對；每個 worker 與 600 秒監測記錄保留於 queue。未啟動 Stage8。\n"
        )
    for doc, link in (
        (ROOT / "README.md", "yolo_activation/result/stage7/README.md"),
        (
            ROOT / "docs/research/yolo-research-plan-2026-09-21.md",
            "../../yolo_activation/result/stage7/README.md",
        ),
    ):
        text = doc.read_text()
        notice = f"2026-11-02 Stage7 queue 已完成處理，狀態 `{result['status']}`；比較選定 `{winner}`，qSiLU 接續模型另列。[實際結果]({link})。Stage8 未啟動，仍待指定。\n\n"
        first, rest = text.split("\n", 1)
        doc.write_text(first + "\n\n" + notice + rest)
    return result
