"""獨立補充成果；現行主線與前 100E 證據保持。"""

import csv
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

from stage9.common import score
from .common import ROOT, HERE, CONFIG, RUN, RESULT, NAME, protocol, read, write, digest, guard


def finalize(zero, trained, layout):
    guard()
    p = protocol()
    previous = read(ROOT / p["dependency"]["prior_result"])
    landscape = previous["trained"]["rect512_768"]["best"]
    best = {**trained["best"], "candidate": NAME, "shape": [768, 512]}
    base = p["parent"]["metrics"]
    limits = p["selection"]
    violations = {k: best["metrics"][k] - v for k, v in base.items()
                  if best["metrics"][k] < v - limits["deployment_max_any_drop"]}
    quality_pass = not violations and score(best["metrics"]) >= score(base) + limits["min_score_gain"]
    RESULT.mkdir(parents=True, exist_ok=True)
    from stage9 import report as exporter
    building = RESULT / ".handoff-building"
    if building.exists():
        shutil.rmtree(building)
    old_run = exporter.RUN
    try:
        exporter.RUN = RUN
        validation = exporter.portable(best, building)
    finally:
        exporter.RUN = old_run
    if (RESULT / "handoff").exists():
        shutil.rmtree(RESULT / "handoff")
    building.rename(RESULT / "handoff")
    cpu = read(RUN / "preflight.json")
    cost = cpu["trace"]["conv_mac_plus_pv"] / 1e9
    rows = [{"candidate": "reference_rect640", "phase": "e0", "conv_pv_GMAC": "", **base},
            {"candidate": "rect512_768", "phase": "adapted_best", "conv_pv_GMAC": cost, **landscape["metrics"]},
            {"candidate": NAME, "phase": "e0", "conv_pv_GMAC": cost, **zero["metrics"]},
            {"candidate": NAME, "phase": "adapted_best", "conv_pv_GMAC": cost, **best["metrics"]}]
    with (RESULT / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for path in (CONFIG / "protocol.json", CONFIG / "reference-review.json", RUN / "source.json", RUN / "preflight.json"):
        shutil.copyfile(path, RESULT / path.name)
    epochs = len(trained["initial"]["history"]) + len((trained["low_lr"] or {}).get("history", []))
    result = {"status": "complete", "completed_at": datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
              "best": best, "quality_gate_passed": quality_pass, "violations": violations,
              "delta_vs_parent": {k: best["metrics"][k] - v for k, v in base.items()},
              "delta_vs_landscape": {k: best["metrics"][k] - v for k, v in landscape["metrics"].items()},
              "zero_shot": zero, "trained": trained, "layout": layout,
              "portable_validation": validation, "epochs": epochs, "epoch_cap": 20,
              "previous_100E_unchanged": True, "current_mainline_unchanged": True,
              "next_stage_started": False, "hardware_synthesis": False}
    write(RESULT / "result.json", result)
    keys = ["coco_AP", "person_AP", "sports_ball_AP", "baseball_bat_AP", "box_AP50_95", "pose_AP50_95"]
    lines = ["# Stage9 補充：768×512 與 512×768", "",
             f"已完成 {epochs}／20 輪與最佳重載、獨立 CPU 可攜驗證；本候選完成後停止。",
             "", "共同 parent 為 qSiLU Rep17＋20 E1，配方與先前相同，僅改 H×W。影像等比例縮放補邊，不旋轉。",
             "", "| 候選／段落 | COCO AP | person AP | COCO 球 AP | COCO 棒 AP | BBAT box AP | Pose AP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['candidate']}/{row['phase']} | " + " | ".join(f"{row[k]*100:.4f}%" for k in keys) + " |")
    lines += ["", f"原主線的全部採用門檻：{'通過' if quality_pass else '未通過'}；完整十項 AP 與逐項差異見 [result.json](result.json)。",
              "現行正式交接保持；本次候選可從 [獨立交接包](handoff/README.md) 使用，不覆蓋原主線。",
              "", f"雙任務 Conv＋P.V 分析成本 {cost:.6f} GMAC；與反向長寬的像素面積相同。",
              "GPU replay／同步時間見 result.json；它們不是 RTL／HLS 或板端量測。qSiLU／D1／B2／S1／folded17+20 保持。",
              "", "資料為完整 COCO／BBAT5 v1，沒有重切或換標註；單 seed、validation 選型，未增加相同原尺寸 control 加訓。",
              "", "來源見 [protocol.json](protocol.json)、[reference-review.json](reference-review.json)、[source.json](source.json)。",
              "", "清理只處理本候選已無用途的 optimizer resume 與被最佳取代的權重，清單見 [cleanup.json](cleanup.json)。"]
    (RESULT / "README.md").write_text("\n".join(lines) + "\n")
    protected = (ROOT / best["checkpoint"]).resolve()
    removed = []
    for path in RUN.rglob("*.pt"):
        if path.resolve() != protected and (path.name == "resume.pt" or path.name.startswith("best-e")):
            removed.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
            path.unlink(); path.with_suffix(".json").unlink(missing_ok=True)
    write(RESULT / "cleanup.json", {"removed": removed, "protected": str(protected), "no_archive": True})
    worklog = ROOT / "docs/worklogs/2026-11-12-actshape-portrait-training.md"
    with worklog.open("a") as stream:
        stream.write(f"\n## 背景 queue 完成\n\n{result['completed_at']}：{epochs}／20E 完成，"
                     f"採用門檻{'通過' if quality_pass else '未通過'}，完整結果位於 `yolo_actshape/result/stage9-portrait/`。"
                     "重載與可攜驗證通過；現行主線保持，無新階段。清理清單見該目錄 cleanup.json。\n")
    for doc in (ROOT / "README.md", HERE / "README.md", ROOT / "docs/research/yolo-research-plan-2026-09-21.md"):
        text = doc.read_text()
        text = text.replace("<!-- portrait-status -->補充候選 768×512：準備／執行中，新增 20E。<!-- /portrait-status -->",
                            f"<!-- portrait-status -->補充候選 768×512：已完成 {epochs}／20E，結果見 result/stage9-portrait；現行主線保持。<!-- /portrait-status -->")
        doc.write_text(text)
    return {"result": str(RESULT), "epochs": epochs, "quality_gate_passed": quality_pass,
            "portable_validation": validation, "mainline_unchanged": True}
