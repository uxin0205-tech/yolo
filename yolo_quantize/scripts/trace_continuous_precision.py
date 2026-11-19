"""在CPU追蹤兩任務的實際ATen精度，列出浮點實作邊界，不宣稱純整數。"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from yolo_quantize.progressive_preparation import LockedQATParentSpec
from yolo_quantize.qat_runtime import Full35QATRuntime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


class PrecisionTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        inputs = tuple(
            sorted(
                {
                    str(v.dtype)
                    for v in tree_leaves((args, kwargs))
                    if isinstance(v, torch.Tensor)
                }
            )
        )
        outputs = tuple(
            sorted(
                {
                    str(v.dtype)
                    for v in tree_leaves(result)
                    if isinstance(v, torch.Tensor)
                }
            )
        )
        self.counts[str(func), inputs, outputs] += 1
        return result


def main():
    torch.set_num_threads(4)
    parent = LockedQATParentSpec.from_yaml(OUT / "parent-manifest.json")
    preflight = json.loads((OUT / "parent-preflight.json").read_text())
    if preflight["forward_parity"]["status"] != "passed":
        raise ValueError("parent parity not passed")
    loaded = Full35QATRuntime.from_yaml(parent.plan_path).load_deployment_parent(
        parent.inference_checkpoint,
        checkpoint_sha256=parent.inference_sha256,
        full_resume_sha256=parent.full_resume_sha256,
        epoch=parent.selected_epoch,
    )
    loaded.model.eval()
    records = []
    with torch.inference_mode():
        image = torch.rand((1, 3, 64, 64), generator=torch.Generator().manual_seed(0))
        for task in ("detect", "pose"):
            trace = PrecisionTrace()
            with trace:
                output = loaded.model(image, task=task)
            tensors = [x for x in tree_leaves(output) if isinstance(x, torch.Tensor)]
            assert tensors and all(torch.isfinite(x).all() for x in tensors)
            records.extend(
                {
                    "task": task,
                    "operator": op,
                    "input_dtypes": list(inputs),
                    "output_dtypes": list(outputs),
                    "calls": count,
                }
                for (op, inputs, outputs), count in sorted(trace.counts.items())
            )
    payload = {
        "schema_version": 1,
        "status": "passed_diagnostic_precision_inventory",
        "parent_manifest_sha256": parent.config_sha256,
        "inference_sha256": parent.inference_sha256,
        "gpu_used": False,
        "pure_integer_deployment": False,
        "scope": "固定64x64兩任務路徑ATen dtype觀測；fake-quant及materialized權重的算子仍以浮點執行，非所有尺寸或資料分支的窮舉",
        "full_graph_modules": preflight["modules"],
        "protected_summary": preflight["catalog"],
        "operators": records,
        "required_deployment_work": [
            "實際整數Conv/Linear與scale傳遞",
            "add/concat requantization",
            "attention/BinaryQK/MASF及head邊界實作",
            "640與正式匯出backend擴充驗證",
        ],
    }
    destination = OUT / "operator-precision.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "operator_dtype_rows": len(records),
                "pure_integer": False,
            }
        )
    )


if __name__ == "__main__":
    main()
