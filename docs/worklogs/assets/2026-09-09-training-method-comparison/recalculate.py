"""唯讀重算既有訓練紀錄；不載入模型、不啟動 GPU。"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
rows, sources = [], []


def events(path):
    raw = path.read_bytes()
    sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()})
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


opt = events(ROOT / "yolo_optimize/artifacts/direction1-20260908/native-parent-ema-control-adopted/progress.jsonl")
for epoch in range(2, 6):
    er = [r for r in opt if r.get("epoch") == epoch]
    macros = [r for r in er if r["kind"] == "macro"]
    start = next(r for r in er if r["kind"] == "epoch_start")
    end = next(r for r in er if r["kind"] == "epoch_complete")
    durations = []
    for kind in ("validation", "live_validation"):
        starts = [r for r in er if r["kind"] == kind + "_start"]
        ends = [r for r in er if r["kind"] == kind]
        assert len(starts) == len(ends)
        durations.extend(b["time_unix"] - a["time_unix"] for a, b in zip(starts, ends))
    assert len(macros) == 463
    rows.append(dict(project="optimize", run="native-parent-ema-control-adopted", epoch=epoch,
        macro_count=len(macros), macro_gap_median_s=statistics.median(b["time_unix"]-a["time_unix"] for a,b in zip(macros, macros[1:])),
        macro_engine_sum_s=sum(r["seconds"] for r in macros),
        macro_span_excluding_first_s=macros[-1]["time_unix"]-macros[0]["time_unix"],
        epoch_cycle_s=end["time_unix"]-start["time_unix"], validation_s=sum(durations),
        detect_images=sum(r["report"]["detect_images"] for r in macros),
        pose_images=sum(r["report"]["pose_images"] for r in macros),
        amp_retries=sum(r["report"]["amp_overflow_retries"] for r in macros)))

qat_root = ROOT / "yolo_quantize/artifacts/runs/qat/continuous-0907-special-v1"
for run in sorted(qat_root.iterdir()):
    if not run.is_dir():
        continue
    log = events(run / "logs/events.jsonl")
    previous_validation = None
    for epoch in range(5):
        macros = [r for r in log if r["kind"] == "macro" and r["context"]["epoch"] == epoch]
        end = next(r for r in log if r["kind"] == "epoch" and r["step"] == epoch)
        val = next(r for r in log if r["kind"] == "validation" and r["step"] == epoch)
        assert len(macros) == 463
        rows.append(dict(project="quantize", run=run.name, epoch=epoch+1,
            macro_count=len(macros), macro_gap_median_s=statistics.median(b["time_unix"]-a["time_unix"] for a,b in zip(macros, macros[1:])),
            macro_engine_sum_s=None,
            macro_span_excluding_first_s=macros[-1]["time_unix"]-macros[0]["time_unix"],
            epoch_cycle_s=None if previous_validation is None else val["time_unix"]-previous_validation,
            validation_s=val["time_unix"]-end["time_unix"],
            detect_images=end["values"]["images/detect"], pose_images=end["values"]["images/pose"],
            amp_retries=sum(r["values"]["amp/overflow_retries"] for r in macros)))
        previous_validation = val["time_unix"]

assert len(rows) == 34 and all(r["detect_images"] == 118287 for r in rows)
stable = {p: [r for r in rows if r["project"] == p and r["epoch"] >= 2] for p in ("optimize", "quantize")}
summary = {}
for project, records in stable.items():
    summary[project] = {key: {"mean": statistics.mean(r[key] for r in records), "min": min(r[key] for r in records), "max": max(r[key] for r in records)} for key in ("macro_gap_median_s", "epoch_cycle_s", "validation_s")}
summary["ratios"] = {key: summary["quantize"][key]["mean"] / summary["optimize"][key]["mean"] for key in ("macro_gap_median_s", "epoch_cycle_s")}
new_study = []
for run in ("a0-control-recovery-v1", "a0-qk-recovery-v1"):
    path = ROOT / "yolo_optimize/studies/pre-fusion-full35-b100/artifacts" / run / "progress.jsonl"
    for row in events(path):
        if row.get("kind") == "epoch_complete":
            new_study.append({"run": run, "epoch": row["epoch"], "elapsed_seconds_including_validation": row["elapsed_seconds"], "images": row["images"], "macros": row["macros"]})
(OUT / "timing-summary.json").write_text(json.dumps({"summary": summary, "sources": sources, "new_detect_study": new_study, "limitations": ["歷史紀錄比較，沒有單變因 GPU profiler；不分配各因素的因果百分比。", "QAT macro 使用相鄰日誌時間差；optimize 同樣以相鄰時間差比较。", "optimize epoch 為 start 至 complete；QAT 為相鄰 validation 完成，後者包含邊界存檔。", "QAT E1 無完整開始時間，epoch_cycle_s 留空；穩態比較均使用 E2–E5。", "optimize E1 沿用已完成前綴，不用延續程序時間冒稱完整五回合耗時。"]}, ensure_ascii=False, indent=2) + "\n")
with (OUT / "epoch-timings.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("已核對 34 個 epoch、7 組融合模型訓練與 2 組新 Detect 紀錄；全數融合 epoch 均為 118287 張 Detect。")
