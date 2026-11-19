"""本階段結束後整理結果、獨立 CPU 可攜驗證與清理；不開始下一階段。"""

import copy
import csv
import os
import shutil
import subprocess
from pathlib import Path

from .common import ROOT, HERE, RUN, PARENT_ROOT, protocol, read, write, digest, score, guard


def portable(best, folder):
    import torch
    from .model import load as canonical_load
    from .portable import records
    parent = PARENT_ROOT / "handoff"
    folder.mkdir(parents=True, exist_ok=True)
    info = copy.deepcopy(read(parent / "manifest.json"))
    entry = info["models"]["rep17_20"]
    # Freeze only the required deployment runtime, not a second research implementation.
    for name in [*info["runtime_files"], entry["architecture"], "load_model.py", "stage7_contract.py"]:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target)
    for source, name in ((ROOT / best["checkpoint"], entry["checkpoint"]),
                         ((ROOT / best["checkpoint"]).with_suffix(".json"), entry["sidecar"])):
        target = folder / name
        if target.exists():
            target.unlink()
        os.link(source, target)
    shutil.copyfile(HERE / "stage9/portable.py", folder / "load_shape.py")
    shape = {"candidate": best["candidate"], "shape": best["shape"], "parent": protocol()["parent"],
             "selected": best, "preprocessing": protocol()["preprocessing"],
             "source_sha256": digest(RUN / "source.json"), "backend": protocol()["hardware"]}
    write(folder / "shape.json", shape)
    entry.update(sha256=digest(folder / entry["checkpoint"]), metrics=best["metrics"])
    info.update(stage=9, shape=best["shape"], selection=best, next_stage_started=False)
    files = [p for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"]
    info["files"] = {str(f.relative_to(folder)): {"sha256": digest(f), "bytes": f.stat().st_size} for f in files}
    write(folder / "manifest.json", info)
    model = canonical_load(best["checkpoint"]).cpu().eval()
    torch.set_num_threads(2)
    shapes = [[160, 320], best["shape"] or [640, 640]]
    output = []
    for index, hw in enumerate(shapes):
        torch.manual_seed(1700 + index)
        with torch.inference_mode():
            tensors = model(torch.rand(1, 3, *hw), task="both")
        output.append({"shape": hw, "seed": 1700 + index, "records": records(tensors)})
    write(folder / "validation/reference.json", {"outputs": output})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([str(ROOT / ".venv/bin/python"), "-B", str(folder / "load_shape.py")], env=env, check=True, cwd=folder)
    (folder / "README.md").write_text(
        "# Stage9 交接\n\n"
        f"選定 `{best['candidate']}`，固定 H×W：`{best['shape']}`（null 表示保留原 rect640 流程）。\n\n"
        "在獨立 Python 程序 `from load_shape import load, prepare`，`model = load()`；固定尺寸使用 "
        "`tensor, meta = prepare(bgr_image, model.shape_contract['shape'])`，再 `model(tensor, task='both')`。"
        "輸入 RGB BCHW FP32、0–1；bbox/keypoint 依 meta.ratio_pad 反算。原 rect640 請沿用來源 evaluator 的 batch letterbox。\n\n"
        "執行 `python -B load_shape.py` 可重驗矩形及選定尺寸的完整輸出、qSiLU/D1 契約。"
        "可攜包已獨立 CPU 重載驗證，完整 AP 見上層 README；未執行 ONNX/TensorRT/RTL/HLS/板端量測。\n")
    return read(folder / "validation/result.json")


def finalize(zero, trained):
    guard()
    p = protocol()
    folder = HERE / "result/stage9"
    folder.mkdir(parents=True, exist_ok=True)
    cpu = read(RUN / "cpu-audit.json")
    costs = {n: r["conv_mac_plus_pv"] for n, r in cpu["candidates"].items()}
    rows, options = [], []
    for name, result in zero.items():
        option = {**p["parent"], "candidate": name, "shape": p["candidates"][name]["shape"],
                  "metrics": result["metrics"], "phase": "e0", "epoch": 0}
        options.append(option)
        rows.append({"candidate": name, "phase": "e0", "conv_pv_GMAC": costs.get(name, 0) / 1e9 if name in costs else "",
                     **result["metrics"]})
    for name, result in trained.items():
        best = {**result["best"], "candidate": name, "shape": p["candidates"][name]["shape"]}
        options.append(best)
        rows.append({"candidate": name, "phase": "adapted_best", "conv_pv_GMAC": costs[name] / 1e9, **best["metrics"]})
    limits, base = p["selection"], p["parent"]["metrics"]
    eligible = [o for o in options if o["shape"] and
                all(o["metrics"][k] >= v - limits["deployment_max_any_drop"] for k, v in base.items()) and
                score(o["metrics"]) >= score(base) + limits["min_score_gain"]]
    original = next(o for o in options if o["candidate"] == "reference_rect640")
    selected = max(eligible, key=lambda o: score(o["metrics"])) if eligible else original
    cost_alternatives = [o for o in options if o["candidate"] in costs and
                        costs[o["candidate"]] <= costs["fixed640"] * (1 - limits["cost_alternative_min_mac_saving"]) and
                        all(o["metrics"][k] >= v - limits["cost_alternative_max_any_drop"] for k, v in base.items())]
    with (folder / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name in ("protocol.json", "reference-review.json"):
        shutil.copyfile(HERE / name, folder / name)
    for name in ("cpu-audit.json", "data-audit.json", "screening.json", "source.json"):
        shutil.copyfile(RUN / name, folder / name)
    shutil.copyfile(HERE / "stage9/redraw.py", folder / "redraw.py")
    validation = portable(selected, folder / "handoff")
    handoff = HERE / "handoff"
    if not handoff.exists():
        handoff.symlink_to("result/stage9/handoff", target_is_directory=True)
    result = {"selected": selected, "cost_alternatives": cost_alternatives, "zero_shot": zero,
              "trained": trained, "portable_validation": validation, "parent_preserved": True,
              "reason": "各項AP門檻及綜合分數通過" if eligible else "沒有候選通過採用門檻，保留原尺寸",
              "hardware_measured": False, "stage_complete": True}
    write(folder / "result.json", result)
    from .redraw import redraw
    redraw(folder)
    lines = ["# Stage9 形狀研究結果", "", f"採用：**{selected['candidate']}**。{result['reason']}。Stage8 qSiLU Rep17＋20 最終模型保持不變。", "",
             "| 候選／階段 | COCO AP | person AP | 球 AP | 球棒 AP | BBAT box AP | Pose AP | Conv＋PV GMAC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        vals = [f"{row[k]:.6f}" for k in ("coco_AP", "person_AP", "sports_ball_AP", "baseball_bat_AP", "box_AP50_95", "pose_AP50_95")]
        lines.append(f"| {row['candidate']}/{row['phase']} | " + " | ".join(vals) + f" | {row['conv_pv_GMAC']} |")
    lines += ["", "完整十項 AP、小物件 AP、兩站 score/PWL 診斷、補邊與有效縮放，見 result.json、metrics.csv、data-audit.json 與各 job 的 runtime-diagnostics.json。",
              "", "成本是固定 canvas、both routing 的 Conv＋P·V 計數，不等於 GPU latency。原 rect640 的成本依 batch canvas 改變，未以 640 正方形代替；省運算門檻只對 fixed640。二次冪有利於部分位址計算，不保證總成本較低。",
              "", "qSiLU、D1 PWL＋LUT256、B2、Q10／signed16 bias 與 folded Rep17＋20 保持。其餘浮點／除法與 head sigmoid 見 cpu-audit.json；未進行整網量化。",
              "", "[可直接使用的交接](handoff/README.md)。單 seed、validation 選型、無原形狀 control 加訓；未驗證 ONNX/TensorRT、無 RTL/HLS 或板端加速數據。可用 `python redraw.py` 無 GPU／checkpoint 重畫。",
              "", "本階段已完成並停止，未自動開始其他研究。"]
    (folder / "README.md").write_text("\n".join(lines) + "\n")
    # Drop only obsolete optimizer resumes after all final verification succeeds.
    protected = {str((ROOT / o["checkpoint"]).resolve()) for o in [selected, *cost_alternatives]}
    removed = []
    for path in RUN.rglob("*.pt"):
        if path.name == "resume.pt" or (path.name.startswith("best-e") and str(path.resolve()) not in protected):
            removed.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
            path.unlink()
            path.with_suffix(".json").unlink(missing_ok=True)
    write(folder / "cleanup.json", {"removed": removed, "protected": sorted(protected), "no_archive": True})
    worklog = ROOT / "docs/worklogs/2026-11-08-actshape-stage9.md"
    with worklog.open("a") as stream:
        stream.write(f"\n## Queue 完成\n\n採用 {selected['candidate']}；{result['reason']}。完整結果見 `yolo_actshape/result/stage9/README.md`。獨立 CPU 可攜驗證通過。安全清理清單見 cleanup.json，未啟動下一階段。\n")
    (HERE / "README.md").write_text("# 特徵圖形狀研究\n\nStage9 已完成並停止。" + f"採用 **{selected['candidate']}**。\n\n[完整結果](result/stage9/README.md) · [交接](handoff/README.md) · [來源與配方](reference-review.json)\n")
    root_readme = ROOT / "README.md"
    text = root_readme.read_text()
    message = f"Stage9 已完成並停止，形狀研究採用 **{selected['candidate']}**；[完整結果](yolo_actshape/result/stage9/README.md) 與 [交接](yolo_actshape/handoff/README.md)。Stage8 的使用者最終選定 qSiLU Rep17＋20 保留。\n\n"
    if message not in text:
        root_readme.write_text(text.replace("# YOLO 研究工作區\n\n", "# YOLO 研究工作區\n\n" + message, 1))
    research = ROOT / "docs/research/yolo-research-plan-2026-09-21.md"
    text = research.read_text()
    header = "\n2026-11-08 啟動的 Stage9 queue 已完成：採用 " + selected['candidate'] + "，詳見 `yolo_actshape/result/stage9/README.md`；本階段後停止，未啟動其他研究。\n"
    if header not in text:
        first, rest = text.split("\n", 1)
        research.write_text(first + "\n" + header + rest)
    return {"selected": selected["candidate"], "result": str(folder), "portable_validation": validation}
