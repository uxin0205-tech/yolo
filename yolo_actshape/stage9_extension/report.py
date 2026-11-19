"""合併既有與新增尺寸；精度選型、硬體取捨與診斷分開呈現。"""

import csv
import shutil
from pathlib import Path

from .common import ROOT, HERE, RUN, BASE_RUN, protocol, read, write, digest, guard, score


def mark_running():
    # The immutable initial controller writes its own completion notice. Reconcile that
    # notice only after it has released the lock, without changing its running source.
    root = ROOT / "README.md"
    lines = root.read_text().splitlines()
    lines = [line for line in lines if not line.startswith("Stage9 已完成並停止，形狀研究採用")]
    root.write_text("\n".join(lines) + "\n")
    plan = ROOT / "docs/research/yolo-research-plan-2026-09-21.md"
    lines = plan.read_text().splitlines()
    lines = [line for line in lines if not line.startswith("2026-11-08 啟動的 Stage9 queue 已完成：")]
    plan.write_text("\n".join(lines) + "\n")
    (HERE / "README.md").write_text(
        "# 特徵圖形狀研究\n\nStage9 初輪已完成，已授權的擴充 queue 正在接續；本階段尚未結束。\n\n"
        "新增 256×512、384×512、512×768，各 10E＋半 LR 10E；初輪40E＋擴充60E，總上限100E。"
        "七組 canvas 的分塊／記憶體排列 GPU replay 另列，保留原模型數學運算。\n\n"
        "執行 `./run_extension.sh status` 查看狀態；`queue-status.json` 是整體最新入口。"
        "STOP 檔：`artifacts/shape-expanded-20261108-v1/STOP`。每600秒監測、每30秒檢查資源。\n\n"
        "[初輪結果](result/stage9/README.md) · [擴充 protocol](extension/protocol.json) · "
        "[事前核對](extension/reference-review.json) · [硬體分析](analysis/HARDWARE_SHAPES.md)\n")


def choose(options, costs, p):
    base, limits = p["parent"]["metrics"], p["selection"]
    eligible = [o for o in options if o["shape"] and
                all(o["metrics"][k] >= v - limits["deployment_max_any_drop"] for k, v in base.items()) and
                score(o["metrics"]) >= score(base) + limits["min_score_gain"]]
    original = next(o for o in options if o["candidate"] == "reference_rect640")
    selected = max(eligible, key=lambda o: score(o["metrics"])) if eligible else original
    # A scalar-summary frontier is not a claim of per-class or hardware Pareto optimality.
    fixed = [o for o in options if o["candidate"] in costs]
    frontier = [o for o in fixed if not any(
        costs[q["candidate"]] <= costs[o["candidate"]] and score(q["metrics"]) >= score(o["metrics"])
        and (costs[q["candidate"]] < costs[o["candidate"]] or score(q["metrics"]) > score(o["metrics"]))
        for q in fixed)]
    return selected, frontier, bool(eligible)


def finalize(zero, trained, layouts):
    guard()
    p, folder = protocol(), HERE / "result/stage9"
    initial_path = folder / "initial-results.json"
    if not initial_path.exists():
        shutil.copyfile(folder / "result.json", initial_path)
        shutil.copyfile(folder / "metrics.csv", folder / "initial-metrics.csv")
    initial = read(initial_path)
    costs = {n: r["conv_mac_plus_pv"] for n, r in read(BASE_RUN / "cpu-audit.json")["candidates"].items()}
    costs.update({n.removeprefix("proposal_"): r["conv_mac_plus_pv"] for n, r in
                  read(HERE / "analysis/hardware-shapes-cpu.json")["proposals"].items()})
    base_protocol = read(HERE / "protocol.json")
    options = [{**p["parent"], "candidate": n, "shape": base_protocol["candidates"][n]["shape"],
                "metrics": r["metrics"], "phase": "e0", "epoch": 0} for n, r in initial["zero_shot"].items()]
    for name, result in initial["trained"].items():
        best = result["best"]
        # The initial finalizer may have removed non-selected weights. Their full
        # metrics remain in the merged CSV, but absent weights are not deliverable options.
        if (ROOT / best["checkpoint"]).exists():
            if digest(ROOT / best["checkpoint"]) != best["sha256"]:
                raise ValueError("初輪 best SHA 不符")
            options.append({**best, "candidate": name, "shape": base_protocol["candidates"][name]["shape"]})
    with (folder / "initial-metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for name, value in zero.items():
        option = {**p["parent"], "candidate": name, "shape": p["candidates"][name]["shape"],
                  "metrics": value["metrics"], "phase": "e0", "epoch": 0}
        options.append(option)
        rows.append({"candidate": name, "phase": "e0", "conv_pv_GMAC": costs[name] / 1e9, **value["metrics"]})
    for name, value in trained.items():
        option = {**value["best"], "candidate": name, "shape": p["candidates"][name]["shape"]}
        options.append(option)
        rows.append({"candidate": name, "phase": "adapted_best", "conv_pv_GMAC": costs[name] / 1e9,
                     **value["best"]["metrics"]})
    selected, frontier, promoted = choose(options, costs, p)
    from stage9 import report as base_report
    build = folder / ".handoff-building"
    if build.exists():
        shutil.rmtree(build)
    previous_run = base_report.RUN
    try:
        base_report.RUN = RUN
        validation = base_report.portable(selected, build)
    finally:
        base_report.RUN = previous_run
    if (folder / "handoff").exists():
        shutil.rmtree(folder / "handoff")
    build.rename(folder / "handoff")
    with (folder / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expansion = folder / "extension"
    expansion.mkdir(exist_ok=True)
    for source in (HERE / "extension/protocol.json", HERE / "extension/reference-review.json",
                   RUN / "source.json", RUN / "tests.json"):
        shutil.copyfile(source, expansion / source.name)
    write(expansion / "layout-results.json", layouts)
    write(expansion / "training-results.json", trained)
    with (expansion / "gpu-benchmark.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate", "height", "width", "device", "dtype", "batch",
                                                    "median_ms", "mean_ms", "peak_allocated_bytes", "incremental_peak_bytes"])
        writer.writeheader()
        for name, value in layouts.items():
            timing = value["whole_model_both_timing"]
            writer.writerow({"candidate": name, "height": value["shape"][0], "width": value["shape"][1],
                             "device": value["device"], "dtype": "FP32", "batch": 1,
                             **{k: timing[k] for k in ("median_ms", "mean_ms", "peak_allocated_bytes", "incremental_peak_bytes")}})
    reason = "完整十項 AP 門檻及綜合分數通過" if promoted else "沒有固定尺寸通過所有採用門檻，保留原 rect640 流程"
    result = {"selected": selected, "reason": reason, "initial_result": str(initial_path),
              "zero_shot": {**initial["zero_shot"], **zero}, "trained": {**initial["trained"], **trained},
              "available_options": options, "summary_score_vs_MAC_frontier": frontier,
              "frontier_limit": "僅十項AP聚合分數與Conv+PV MAC；非各類AP或目標硬體資源的完整Pareto",
              "layout_results": layouts, "portable_validation": validation, "stage_complete": True,
              "parent_preserved": True, "max_epoch_budget": 100, "hardware_measured": False, "GPU_measured": True}
    write(folder / "result.json", result)
    from stage9.redraw import redraw
    redraw(folder)
    lines = ["# Stage9 完整形狀研究結果", "", f"採用 **{selected['candidate']}**。{reason}。", "",
             "包含初輪及新增三組尺寸。Stage8 qSiLU Rep17＋20 最終 parent 保持。尺寸順序均為 H×W。", "",
             "| 候選／階段 | COCO AP | person AP | 球 AP | 棒 AP | BBAT box AP | Pose AP | Conv＋PV GMAC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [f"{float(row[k]):.6f}" for k in ("coco_AP", "person_AP", "sports_ball_AP", "baseball_bat_AP", "box_AP50_95", "pose_AP50_95")]
        lines.append(f"| {row['candidate']}/{row['phase']} | " + " | ".join(values) + f" | {row['conv_pv_GMAC']} |")
    lines += ["", "十項 AP、各尺寸最佳、來源、負結果與選型均見 result.json／metrics.csv。"
              "各尺寸最佳可作硬體成本取捨，不將所有 AP 壓成單一分數而隱去退化。",
              "", "分塊診斷見 extension/layout-results.json；FP32 batch1、同步 PyTorch 推論時間見 extension/gpu-benchmark.csv。"
              "分塊只作算子 replay，不替換正式任務模型。每個 query 分塊保留全部 keys；QK 分數矩陣仍完整物化。",
              "", "完整 AP 使用 COCO 5000／BBAT5 v1 683 張；每任務一張的 layout 診斷不代替 AP。"
              "tile 記憶體含補齊但不代表峰值 SRAM；Conv+PV MAC 不包含全部算子，也不等於 GPU 或板端加速。",
              "", "qSiLU／D1 PWL＋LUT256／B2／S1／folded17+20 保留。未改固定輸入下的內部 stride 架構、"
              "未做整網整數量化、RTL/HLS 或板端量測。單 seed、validation 選型，沒有原尺寸 control 加訓。",
              "", "[可直接使用的交接](handoff/README.md)。`python redraw.py` 可在無 GPU／checkpoint 下重畫。",
              "", "本階段全部完成並停止，未啟動下一階段。"]
    (folder / "README.md").write_text("\n".join(lines) + "\n")
    # Preserve all three new shape-best models as unique hardware trade-off evidence.
    keep = {(ROOT / v["best"]["checkpoint"]).resolve() for v in trained.values()}
    keep.add((ROOT / selected["checkpoint"]).resolve())
    removed = []
    for path in RUN.rglob("*.pt"):
        if path.name == "resume.pt" or (path.name.startswith("best-e") and path.resolve() not in keep):
            removed.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
            path.unlink()
            path.with_suffix(".json").unlink(missing_ok=True)
    write(expansion / "cleanup.json", {"removed": removed, "protected": sorted(map(str, keep)), "no_archive": True})
    message = f"Stage9 初輪與擴充全部完成，採用 **{selected['candidate']}**；[結果](yolo_actshape/result/stage9/README.md) 與 [交接](yolo_actshape/handoff/README.md)。階段完成後停止。"
    root = ROOT / "README.md"
    text = root.read_text()
    if message not in text:
        root.write_text(text.replace("# YOLO 研究工作區\n\n", "# YOLO 研究工作區\n\n" + message + "\n\n", 1))
    plan = ROOT / "docs/research/yolo-research-plan-2026-09-21.md"
    text = plan.read_text()
    if message not in text:
        first, rest = text.split("\n", 1)
        plan.write_text(first + "\n\n" + message + "\n" + rest)
    (HERE / "README.md").write_text("# 特徵圖形狀研究\n\n" + message.replace("yolo_actshape/", "") +
                                  "\n\n[硬體尺寸分析](analysis/HARDWARE_SHAPES.md) · [擴充配方](extension/protocol.json)\n")
    with (ROOT / "docs/worklogs/2026-11-08-actshape-expanded-gpu.md").open("a") as stream:
        stream.write(f"\n## Queue 完成\n\n{message} 完整 AP、GPU replay 與獨立 CPU 可攜驗證完成；清理見 extension/cleanup.json。"
                     "限制：單 seed、未做 RTL／板端量測；各候選失敗／停止原因保存在 result.json。\n")
    return {"selected": selected["candidate"], "result": str(folder), "portable_validation": validation}
