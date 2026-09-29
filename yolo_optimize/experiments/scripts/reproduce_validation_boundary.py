#!/usr/bin/env python3
"""以真實 parent／materialization 在 CPU 重現驗證來源邊界錯誤；不訓練。

這是診斷 red-loop：來源 tensor、module.training 或 requires_grad 變化即非零退出。
沒有 mock eval、沒有 GPU、沒有資料集抽樣或修改。報告保留每一項實際差異。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_optimize import runtime
from yolo_optimize.training import SCOPE, TrainingModel, validation_boundary
from yolo_optimize.training_safety import FixedStateEMA
import torch
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.stage_policy import apply_stage


def signature(model):
    tensors = {}
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        record = {"dtype": str(value.dtype), "shape": list(value.shape),
                  "sha256": hashlib.sha256(raw).hexdigest()}
        tensors[name] = record
        digest.update(name.encode())
        digest.update(json.dumps(record, sort_keys=True).encode())
    return {"state_digest": digest.hexdigest(), "tensors": tensors,
            "training": {name: module.training for name, module in model.named_modules()},
            "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()}}


def differences(model, before, after):
    groups = {}
    modules = dict(model.named_modules())
    for group in ("tensors", "training", "requires_grad"):
        changed = {}
        for name in sorted(set(before[group]) | set(after[group])):
            old, new = before[group].get(name), after[group].get(name)
            if old != new:
                changed[name] = {"before": old, "after": new}
                if group == "training" and name in modules:
                    module = modules[name]
                    changed[name]["module_type"] = f"{type(module).__module__}.{type(module).__name__}"
                    changed[name]["all_alias_paths"] = [
                        path for path, candidate in model.named_modules(remove_duplicate=False)
                        if candidate is module]
        groups[group] = changed
    return {"state_digest_before": before["state_digest"], "state_digest_after": after["state_digest"],
            "counts": {name: len(value) for name, value in groups.items()}, **groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=runtime.FINAL_ROOT / "weights/combined/inference/best_joint.pt")
    parser.add_argument("--kind", choices=("float", "bittrue"), default="float")
    parser.add_argument("--source", choices=("live", "ema"), default="ema")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU repro 不允許事先初始化 CUDA")
    started = time.perf_counter()
    config = runtime.load_config(runtime.WORKSPACE / "artifacts/direction1-20260908")
    source, base, _factory, _loaded = runtime.load_model(config, args.checkpoint, "cpu")
    apply_stage(base, SCOPE)
    ema = FixedStateEMA(TrainingModel(base), decay=.9999, tau=2000)
    models = {"live": base, "ema": ema.ema.base}
    before = {name: signature(model) for name, model in models.items()}
    error = None
    try:
        with validation_boundary(*models.values()):
            materialized = build_graph_validation_models(models[args.source], source, kind=args.kind)
    except RuntimeError as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    after = {name: signature(model) for name, model in models.items()}
    changes = {name: differences(model, before[name], after[name]) for name, model in models.items()}
    red = any(any(item["counts"].values()) for item in changes.values())
    report = {"kind": "actual_materialization_cpu_validation_boundary_repro", "checkpoint": str(args.checkpoint),
              "backend": args.kind, "materialization_source": args.source, "seconds": time.perf_counter() - started,
              "cuda_initialized": torch.cuda.is_initialized(), "error": error, "red": red, "changes": changes}
    runtime.write_json(args.report, report)
    print(json.dumps({"red": red, "error": error, "seconds": report["seconds"],
                      "changes": {name: {"counts": item["counts"], "training": item["training"]}
                                  for name, item in changes.items()}}, ensure_ascii=False, indent=2))
    if error is not None:
        raise RuntimeError(error["message"])
    if red:
        raise AssertionError("actual materialization 修改了來源 state／flags")


if __name__ == "__main__":
    main()
