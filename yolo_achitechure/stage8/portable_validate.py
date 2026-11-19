"""只靠交接包，在獨立 CPU 程序核對完整輸出及錯誤契約拒絕。"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from load_model import load_model, verify_contract


def records(value, prefix="output"):
    import torch

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("非有限輸出")
        return {
            prefix: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    value.cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            }
        }
    children = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, (tuple, list))
        else ()
    )
    return {
        k: v
        for name, child in children
        for k, v in records(child, f"{prefix}.{name}").items()
    }


def main():
    import torch

    torch.set_num_threads(2)
    reference = json.loads((HERE / "validation/reference.json").read_text())
    results = {}
    for candidate, entries in reference["models"].items():
        model = load_model(candidate=candidate)
        for size, expected in entries.items():
            torch.manual_seed(reference["seed"] + int(size))
            x = torch.rand(1, 3, int(size), int(size))
            with torch.inference_mode():
                outputs = model(x, task="both")
            if records(outputs) != expected:
                raise AssertionError(f"可攜輸出不等同 canonical：{candidate} {size}")
        index = model.stage8_manifest["layers"][0]
        model.graph.model[index].act = torch.nn.SiLU()
        try:
            verify_contract(model, model.stage8_manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("未拒絕 SiLU 回退")
        results[candidate] = {
            "sizes": list(entries),
            "bitwise_equal": True,
            "silu_rejected": True,
        }
        print(f"PASS portable {candidate}", flush=True)
    result = {
        "status": "passed",
        "models": results,
        "gpu_used": torch.cuda.is_initialized(),
    }
    if result["gpu_used"]:
        raise ValueError("交接驗證意外使用 GPU")
    (HERE / "validation/result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
