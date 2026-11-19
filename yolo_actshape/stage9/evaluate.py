"""沿用完整 COCO／BBAT evaluator，前處理 canvas 是唯一評估變因。"""

from pathlib import Path
import torch
from binaryqk.evaluate import evaluate as detect_evaluate
from masf_yolo.stage4.evaluate import evaluate_pose
from stage6.evaluate import vector
from stage6.models import view
from stage7.diagnostics import RuntimeDiagnostics
from .common import event, write
from .data import dataset
from .model import verify


def pair(model, name, shape, output, diagnostics=False):
    output = Path(output)
    verify(model)
    model.eval()
    monitor = RuntimeDiagnostics(model) if diagnostics else None
    try:
        ds, data, _ = dataset("detect", "val", shape)
        det = detect_evaluate(view(model, "detect"), ds, data, output / "detect",
                              progress=lambda e: event(output, candidate=name, task="detect", **e))
        ds, data, _ = dataset("pose", "val", shape)
        pose = evaluate_pose(view(model, "pose"), ds, data, output / "pose",
                              progress=lambda e: event(output, candidate=name, task="pose", **e))
    finally:
        if monitor:
            write(output / "runtime-diagnostics.json", monitor.close())
    result = {"status": "passed", "candidate": name, "shape": shape, "metrics": vector(det, pose),
              "detect": det, "pose": pose, "backend": "qSiLU/D1/B2/S1/folded17+20",
              "complete_data": True, "same_evaluator": True}
    verify(model)
    write(output / "metrics.json", result)
    torch.cuda.empty_cache()
    return result
