"""真實圖形狀、局部位寬與計算成本；分析值不冒稱硬體量測。"""

import math
import statistics
from pathlib import Path

import torch
from PIL import Image

from . import model as checkpoint
from .common import HERE, RUN, protocol, shape_for, write, read, ROOT
from .data import dataset, geometry


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def trace(model, shape):
    rows, handles = [], []
    conv_mac = 0

    def conv_hook(module, inputs, output):
        nonlocal conv_mac
        conv_mac += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        if name in {"graph.model.16", "graph.model.19", "graph.model.22", "graph.model.10.m.0.attn", "graph.model.22.m.0.1.attn"}:
            def hook(m, inputs, output, name=name):
                rows.append({"module": name, "input": list(inputs[0].shape), "output": list(output.shape)})
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, *shape), task="both")
        if not finite(output):
            raise FloatingPointError("非有限形狀前向")
    finally:
        for hook in handles:
            hook.remove()
    attention = []
    for row in rows:
        if row["module"].endswith(".attn"):
            h, w = row["input"][-2:]
            n = h * w
            if h > 32 or w > 32 or n > 1024:
                raise ValueError("超過已選 bias／D1 範圍")
            attention.append({"site": row["module"], "height": h, "width": w, "tokens": n,
                              "max_denominator_code": n * 32768, "required_unsigned_bits": (n * 32768).bit_length(),
                              "contract_bits": 26, "score_matrix_fp32_bytes": 4 * n * n * 4,
                              "pv_float_mac": 4 * 64 * n * n,
                              "binary_qk_bit_products": 2 * 4 * 32 * n * n,
                              "pow2_axis_padding_tokens": (1 << (h - 1).bit_length()) * (1 << (w - 1).bit_length()),
                              "pow2_padding_is_estimate_only": True})
    return {"input": list(shape), "shapes": rows, "attention": attention, "conv_mac": conv_mac,
            "conv_mac_plus_pv": conv_mac + sum(a["pv_float_mac"] for a in attention),
            "scope": "CPU FP32 both routing 實際 Conv 與 PV 計數；不含逐元素、搬移或 binary QK，非 latency"}


def cpu_audit():
    torch.set_num_threads(2)
    model = checkpoint.load().cpu().eval()
    rows = {}
    for name, candidate in protocol()["candidates"].items():
        if candidate["shape"] is not None:
            rows[name] = trace(model, candidate["shape"])
            write(RUN / "cpu-shapes-progress.json", rows)
    # Same model changes shape and then returns; head grid/cache must not remain stale.
    with torch.inference_mode():
        torch.manual_seed(5)
        x = torch.rand(1, 3, 160, 320)
        first = model(x, task="both")
        model(torch.zeros(1, 3, 192, 128), task="both")
        last = model(x, task="both")
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(compare(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(compare(x, y) for x, y in zip(a, b))
        return a == b
    if not compare(first, last):
        raise ValueError("跨 shape 後的 cache 不等價")
    norm = model.get_submodule("graph.model.10.m.0.attn.normalizer")
    numerical = []
    for n in [1, 240, 256, 288, 400, 512, 1024]:
        with torch.inference_mode():
            p = norm(torch.zeros(1, 4, 2, n))
        error = float((p.sum(-1) - 1).abs().max())
        if not finite(p) or error > .003 or p.min() < 0 or p.max() > 1:
            raise ValueError("新 token 數不符合 D1 數值契約")
        numerical.append({"tokens": n, "equal_score_row_sum_error": error})
    output = {"status": "passed", "parent_sha256": protocol()["parent"]["sha256"],
              "candidates": rows, "shape_cache_roundtrip_bitwise_equal": True,
              "D1_numerics": numerical, "GPU_used": False,
              "operators": read(ROOT / "yolo_achitechure/artifacts/stage8/repconv-variants-20261107/verify-rep17_20/operators.json"),
              "tile_padding": "只估算額外 buffer 容量，未向 Attention 注入未遮罩 token；沒有聲稱補零等價。",
              "export": "本輪驗證自訂 PyTorch 圖與可攜重載；未交付 ONNX/TensorRT/RTL，定點整數路徑需專用匯出驗證。"}
    write(RUN / "cpu-audit.json", output)
    return output


def data_audit():
    output = {}
    for task in ("detect", "pose"):
        source, _, _ = dataset(task, "val", None)
        dimensions = []
        for path in source.im_files:
            with Image.open(path) as image:
                w, h = image.size
            dimensions.append((h, w))
        conditions = {}
        for name, spec in protocol()["candidates"].items():
            if spec["shape"]:
                gs = [geometry(hw, spec["shape"]) for hw in dimensions]
                conditions[name] = {"mean_padding_fraction": statistics.mean(g["padding_fraction"] for g in gs),
                                    "mean_linear_gain": statistics.mean(min(g["gain"]) for g in gs)}
        # These are metadata diagnostics on the complete val membership, not a new split.
        output[task] = {"images": len(dimensions), "landscape": sum(w > h for h, w in dimensions),
                        "portrait": sum(h > w for h, w in dimensions),
                        "median_width_height": statistics.median(w / h for h, w in dimensions),
                        "conditions": conditions}
    write(RUN / "data-audit.json", output)
    return output
