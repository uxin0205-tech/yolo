"""完整任務驗證；數值診斷只觀察，不介入部署輸出。"""

import gc
from pathlib import Path

import torch
from binaryqk.evaluate import evaluate as detect_evaluate
from masf_yolo.stage4.evaluate import evaluate_pose
from stage6.data import dataset
from stage6.evaluate import accepted, vector
from stage6.models import view

from .common import event, passed, write_json
from .diagnostics import RuntimeDiagnostics
from .model import manifest_for


def pair(model, output, progress_root=None, diagnostics=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    callback = lambda e: event(progress_root or output, **e)
    manifest_for(model)
    model.eval()
    monitor = RuntimeDiagnostics(model) if diagnostics else None
    try:
        ds, data, _ = dataset("detect", "val", 8)
        det = detect_evaluate(
            view(model, "detect"), ds, data, output / "detect", progress=callback
        )
        write_json(output / "detect.json", det)
        del ds
        ds, data, _ = dataset("pose", "val", 8)
        pose = evaluate_pose(
            view(model, "pose"), ds, data, output / "pose", progress=callback
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
        "activation_relative_gate": passed(values),
        "original_independent_gate": accepted(values),
    }
    manifest_for(model)
    write_json(output / "metrics.json", result)
    del ds
    gc.collect()
    torch.cuda.empty_cache()
    return result
