"""B2 成本分析；預設僅讀小型 JSON，--verify-models 才做 CPU 前向核對。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "yolo_binaryqk/artifacts/stage1/formal-20261018-v2"
REFERENCE = ROOT / "yolo_optimize/experiments/benchmark/artifacts/accounting-v2/fp.json"


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def verify_models():
    import torch
    from binaryqk.checkpoint import load
    from binaryqk.model import PARENT_SHA, SITES, load_parent, sha256
    from torch.utils._python_dispatch import TorchDispatchMode

    class MatrixMAC(TorchDispatchMode):
        # 依 optimize accounting.py；不使用 inference_mode 計矩陣 MAC。
        def __init__(self):
            super().__init__()
            self.floating = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if name in ("aten.mm.default", "aten.bmm.default", "aten.addmm.default"):
                a, b = args[-2:] if name == "aten.addmm.default" else args[:2]
                assert a.is_floating_point() and b.is_floating_point()
                self.floating += a.numel() * b.shape[-1]
            return func(*args, **(kwargs or {}))

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(x) for x in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(x) for x in value)
        return True

    x = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(50900912))

    def count(model):
        result = {"conv_mac": 0, "attention_conv_mac": 0, "sites": {}}
        hooks = []

        def conv(module, args, out, name):
            mac = out.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            result["conv_mac"] += mac
            if any(name.startswith(site + ".") for site in SITES):
                result["attention_conv_mac"] += mac

        def shape(module, args, name):
            result["sites"][name] = {
                "input_shape": list(args[0].shape), "heads": module.num_heads,
                "key_dim": module.key_dim, "value_dim": module.head_dim,
            }

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(lambda m, a, o, n=name: conv(m, a, o, n)))
            if name in SITES:
                hooks.append(module.register_forward_pre_hook(lambda m, a, n=name: shape(m, a, n)))
        counter = MatrixMAC()
        try:
            with torch.inference_mode(False), torch.no_grad(), counter:
                output = model.eval()(x)
            assert finite(output), "nonfinite model output"
        finally:
            for hook in hooks:
                hook.remove()
        result.update(float_matrix_mac=counter.floating,
                      dense_mac=result["conv_mac"] + counter.floating,
                      parameters=sum(p.numel() for p in model.parameters()), finite_output=True)
        return result

    native = count(load_parent())
    handoff = json.loads((OUTPUT / "handoff.json").read_text())
    assert sha256(handoff["checkpoint"]) == handoff["sha256"]
    model, manifest = load(handoff["checkpoint"])
    assert manifest["variant"] == "B2" and manifest["bases"] == ["sign", "hadamard"]
    b2 = count(model)
    assert native["sites"] == b2["sites"]
    assert native["conv_mac"] == b2["conv_mac"]
    assert native["parameters"] == b2["parameters"]
    reference = json.loads(REFERENCE.read_text())
    assert reference["checkpoint_sha256"] == PARENT_SHA
    for key in ("conv_mac", "float_matrix_mac"):
        assert native[key] == reference["cost"][key], key
    return {"device": "cpu", "input_shape": list(x.shape), "mode": "eval/no_grad/unfused",
            "parent_sha256": PARENT_SHA, "B2_sha256": handoff["sha256"],
            "native": native, "B2": b2, "matches_optimize_native_accounting": True}


def analyze(measured):
    qk = pv = pairs = channels = hadamard = 0
    for site in measured["native"]["sites"].values():
        batch, _, height, width = site["input_shape"]
        heads, dim, value = site["heads"], site["key_dim"], site["value_dim"]
        assert dim == 32, "32-bit packing estimate requires key_dim=32"
        n = height * width
        pairs += batch * heads * n * n
        channels += batch * heads * dim * n
        qk += batch * heads * n * n * dim
        pv += batch * heads * n * n * value
        hadamard += 2 * batch * heads * n * dim * int(math.log2(dim))
    native, b2 = measured["native"], measured["B2"]
    assert native["float_matrix_mac"] == qk + pv
    assert b2["float_matrix_mac"] == pv
    assert native["dense_mac"] - b2["dense_mac"] == qk
    added_float_ops = hadamard + 2 * pairs + pairs
    net_float_saved = 2 * qk + channels - added_float_ops
    sources = [REFERENCE, ROOT / "yolo_optimize/experiments/benchmark/accounting.py",
               ROOT / "yolo_binaryqk/binaryqk/basis.py", ROOT / "yolo_binaryqk/binaryqk/model.py",
               ROOT / "yolo_p2/ultralytics/nn/modules/block.py"]
    return {
        "scope": "每張 640x640，batch1，兩個20x20 sites，未 fuse 的本機推論前向；不是硬體量測",
        "measured": measured,
        "dense_mac_removed": qk, "dense_mac_reduction_percent": 100 * qk / native["dense_mac"],
        "attention_qk_pv_float_mac_reduction_percent": 100 * qk / (qk + pv),
        "attention_including_projection_float_mac_reduction_percent":
            100 * qk / (qk + pv + native["attention_conv_mac"]),
        "binary_bit_products": 2 * qk,
        "packed_32bit_estimate": {"xnor": 2 * pairs, "popcount": 2 * pairs,
                                  "integer_count_times2_minus32": 2 * pairs,
                                  "packing_and_memory_excluded": True, "implemented": False},
        "B2_added_float_ops": {"hadamard_add_sub": hadamard, "coefficient_multiplies": 2 * pairs,
                               "branch_additions": pairs},
        "native_removed_q_scale_multiplies": channels,
        "minimum_sign_comparisons": 4 * channels,
        "net_float_arithmetic_saved_after_B2_overhead": net_float_saved,
        "net_float_arithmetic_reduction_percent_subtotal":
            100 * net_float_saved / (2 * native["dense_mac"] + channels),
        "limits": ["MAC小計不含共同BN/activation/Softmax/decode/sort/pooling及記憶體搬移",
                   "位元運算、整數popcount與浮點MAC不可當等價運算相加",
                   "當前未打包布林reference另有重複sign比較、int64歸約與cast，無實測加速結論",
                   "以上為推論；訓練反向仍使用浮點代理，不可套用為訓練成本降幅"],
        "sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sources},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-models", action="store_true")
    args = parser.parse_args()
    measured_path = OUTPUT / "cost-model-check.json"
    if args.verify_models:
        write(measured_path, verify_models())
    result = analyze(json.loads(measured_path.read_text()))
    write(OUTPUT / "operation-cost.json", result)
    print(json.dumps({k: result[k] for k in (
        "dense_mac_removed", "dense_mac_reduction_percent", "binary_bit_products",
        "B2_added_float_ops", "net_float_arithmetic_reduction_percent_subtotal")}, indent=2))
