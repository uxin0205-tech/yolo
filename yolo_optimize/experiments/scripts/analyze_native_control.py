#!/usr/bin/env python3
"""只用 CPU 分析已完成的 native5：AP、group drift、BN 與已記錄梯度。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from audit_recovery_states import _per_tensor_stats, _aggregate, _is_neck

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/direction1-20260908/native-parent-ema-control-adopted"
DIAGNOSTIC = RUN.parent / "ema-age-paired"
OUTPUT = RUN.parent / "native5-cpu-analysis.json"


def joint(metrics):
    return sum(weight * metrics[key] for key, weight in (
        ("coco/box/map50_95", .2), ("coco/person/box/map50_95", .2),
        ("bbat/box/map50_95", .2), ("bbat/pose/map50_95", .4)))


def distribution(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.mean(values)} if values else {"count": 0}


def group(name):
    if ".detect_head." in name or ".pose_head." in name:
        head = "detect" if ".detect_head." in name else "pose"
        branch = "one2one" if ".one2one_" in name else "one2many"
        if name.endswith((".running_mean", ".running_var")):
            return f"{head}_{branch}_bn_statistics"
        if name.endswith((".weight", ".bias")):
            return f"{head}_{branch}_parameters"
    if _is_neck(name) and "masf" not in name and ".attn." not in name:
        if name.endswith((".weight", ".bias")):
            return "neck_parameters"
    return None


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    summary = json.loads((RUN / "summary.json").read_text())
    parent_path = Path(summary["metadata"]["parent"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    parent_ap = summary["metadata"]["criteria_continuation"]["parent_metrics"]
    fixed = tuple(summary["metadata"]["ema"]["fixed_state_names"])
    grouped = {}
    for name in parent:
        category = group(name)
        if category is not None:
            grouped.setdefault(category, []).append(name)
    logs = {}
    for path in (DIAGNOSTIC / "progress.jsonl", RUN / "progress.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("kind") == "macro":
                key = (event["epoch"], event["macro"])
                if key in logs:
                    raise ValueError(f"重複 macro：{key}")
                logs[key] = event
    results = []
    for record in summary["epochs"]:
        epoch = record["epoch"]
        payload = torch.load(RUN / "checkpoints" / f"epoch-{epoch:04d}.pt",
                             map_location="cpu", weights_only=True, mmap=True)
        banks = {}
        for bank, field in (("live", "model_state"), ("ema", "ema_state")):
            state = {key.removeprefix("base."): value for key, value in payload[field].items()}
            if set(state) != set(parent):
                raise ValueError("snapshot state keys 與原 parent 不同")
            changed_fixed = [key for key in fixed if not torch.equal(
                payload[field][key], parent[key.removeprefix("base.")])]
            stats = {name: _per_tensor_stats(name, parent[name], state[name])
                     for names in grouped.values() for name in names}
            banks[bank] = {"changed_fixed_states": changed_fixed,
                "groups": {name: _aggregate(names, stats, top_n=3)
                           for name, names in grouped.items()}}
        events = [event for (e, _), event in sorted(logs.items()) if e == epoch]
        if len(events) != 463:
            raise ValueError(f"E{epoch} macro 紀錄不是完整 463")
        grad = [event["report"]["gradient_statistics"] for event in events
                if event["report"]["gradient_statistics"] is not None]
        cos = [item["cosine_similarity"] for item in grad]
        preclip = [event["report"]["clipped_gradient_norm"] for event in events]
        ema_ap, live_ap = record["metrics"]["bittrue"], record["live_metrics"]
        results.append({"epoch": epoch, "banks": banks, "ema_updates": payload["ema_updates"],
            "scheduler_step": payload["scheduler_state"]["current_step"],
            "criterion_state": payload["criteria_state"],
            "ema_joint": joint(ema_ap), "live_joint": joint(live_ap),
            "ema_delta": {key: ema_ap[key] - value for key, value in parent_ap.items()},
            "live_delta": {key: live_ap[key] - value for key, value in parent_ap.items()},
            "gradients": {"samples": grad, "cosine": distribution(cos),
                "negative_fraction": sum(value < 0 for value in cos) / len(cos) if cos else None,
                "pose_projection_fraction": distribution([max(-value, 0) for value in cos]),
                "pose_to_detect_norm_ratio": distribution([g["pose_norm"] / g["detect_norm"] for g in grad]),
                "preclip_norm": distribution(preclip),
                "clipped_fraction": sum(value > 10 for value in preclip) / len(preclip),
                "amp_retries": sum(event["report"]["amp_overflow_retries"] for event in events)}})
        print(json.dumps({"epoch": epoch, "live_joint": results[-1]["live_joint"],
                          "ema_joint": results[-1]["ema_joint"]}), flush=True)
        del payload
    result = {"status": summary["status"], "parent_joint": joint(parent_ap),
        "parent_metrics": parent_ap, "snapshot_count": len(results), "epochs": results,
        "notes": ["CPU checkpoint／已保存日誌分析，沒有重新推論，相關性不是根因證明。",
                  "clipped_gradient_norm 欄位是 clip_grad_norm_ 回傳的裁剪前 norm。",
                  "gradient samples 是定期抽取的既有 macro 統計，不是完整訓練母體。"]}
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(str(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
