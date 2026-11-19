"""正式 AP 使用真正 folded 部署圖；Float PWL 只供診斷參考。"""

import gc
from pathlib import Path

import torch
from binaryqk.evaluate import evaluate as detect_evaluate
from masf_yolo.stage4.evaluate import evaluate_pose
from stage6.data import dataset
from stage6.evaluate import accepted, vector
from stage6.models import view
from stage7.diagnostics import RuntimeDiagnostics

from .common import event, passed, write_json
from .model import folded, manifest_for


def pair(model, output, progress_root=None, diagnostics=False, float_reference=False, operators="hardware"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_for(model)
    operators = model.stage8_manifest["operators"] if operators == "training" else operators
    deploy = folded(model, operators=operators).cuda().eval()
    if float_reference and operators != "hardware":
        raise ValueError("Float PWL 參考需以硬體部署為起點")
    if float_reference:
        for path in ("graph.model.10.m.0.attn", "graph.model.22.m.0.1.attn"):
            attention = deploy.get_submodule(path)
            attention.normalizer = attention.normalizer.proxy
    monitor = (
        RuntimeDiagnostics(deploy) if diagnostics and not float_reference and operators == "hardware" else None
    )
    callback = lambda e: event(progress_root or output, **e)
    try:
        ds, data, _ = dataset("detect", "val", 8)
        det = detect_evaluate(
            view(deploy, "detect"), ds, data, output / "detect", progress=callback
        )
        write_json(output / "detect.json", det)
        del ds
        ds, data, _ = dataset("pose", "val", 8)
        pose = evaluate_pose(
            view(deploy, "pose"), ds, data, output / "pose", progress=callback
        )
        write_json(output / "pose.json", pose)
    finally:
        if monitor:
            write_json(output / "runtime-diagnostics.json", monitor.close())
    values = vector(det, pose)
    result = {
        "metrics": values,
        "detect": det,
        "pose": pose,
        "rep_relative_gate": passed(values),
        "original_independent_gate": accepted(values),
        "folded": True,
        "backend": "float-pwl-diagnostic-only"
        if float_reference
        else ("D1-integer-pwl-reciprocal-lut256" if operators == "hardware" else "float-pwl-10-20-fp32"),
        "operators": operators,
        "qsilu_preserved": operators != "rep_silu_pwl",
    }
    manifest_for(model)
    write_json(output / "metrics.json", result)
    del ds, deploy
    gc.collect()
    torch.cuda.empty_cache()
    return result
