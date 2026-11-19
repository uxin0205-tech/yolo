"""GPU/CPU 參考 replay：保留邏輯形狀與完整分母，不替換任務模型。"""

import math
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from stage9.data import dataset
from stage9.model import load, verify
from .common import protocol, event, write


def tile_roundtrip(x, tile):
    b, c, h, w = x.shape
    hp, wp = math.ceil(h / tile) * tile, math.ceil(w / tile) * tile
    packed = F.pad(x, (0, wp - w, 0, hp - h)).reshape(b, c, hp // tile, tile, wp // tile, tile)
    packed = packed.permute(0, 2, 4, 1, 3, 5).contiguous()
    restored = packed.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(b, c, hp, wp)[..., :h, :w]
    return restored, {"logical_hw": [h, w], "tile": tile, "capacity_hw": [hp, wp],
                      "valid_cell_utilization": h * w / (hp * wp), "packed_bytes": packed.numel() * packed.element_size()}


def pitch_roundtrip(x, alignment):
    b, c, h, w = x.shape
    pitch = math.ceil(w / alignment) * alignment
    storage = torch.full((b, c, h, pitch), float("nan"), device=x.device, dtype=x.dtype)
    storage[..., :w] = x
    return storage[..., :w].clone(), {"pitch_elements": pitch, "alignment_elements": alignment,
                                     "extra_capacity_fraction": pitch / w - 1}


def tiled_conv(x, conv, tile, activation):
    if conv.padding_mode != "zeros":
        raise ValueError("此參考只支援零padding Conv")
    ph, pw = conv.padding
    sh, sw = conv.stride
    kh = conv.dilation[0] * (conv.kernel_size[0] - 1) + 1
    kw = conv.dilation[1] * (conv.kernel_size[1] - 1) + 1
    h, w = x.shape[-2:]
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    source = F.pad(x, (pw, pw, ph, ph))
    output = x.new_empty((x.shape[0], conv.out_channels, oh, ow))
    for y in range(0, oh, tile):
        end_y = min(y + tile, oh)
        for z in range(0, ow, tile):
            end_z = min(z + tile, ow)
            patch = source[..., y * sh:(end_y - 1) * sh + kh, z * sw:(end_z - 1) * sw + kw]
            output[..., y:end_y, z:end_z] = F.conv2d(patch, conv.weight, conv.bias, conv.stride,
                                                    padding=0, dilation=conv.dilation, groups=conv.groups)
    return activation(output)


def tiled_normalization(normalizer, scores, values, tile):
    probabilities, products = [], []
    for start in range(0, scores.shape[-2], tile):
        # All K keys are present in every row. The final query block is not padded.
        p = normalizer(scores[..., start:start + tile, :])
        probabilities.append(p)
        products.append(values @ p.transpose(-2, -1))
    return torch.cat(probabilities, dim=-2), torch.cat(products, dim=-1)


def error(actual, expected, *, atol, rtol):
    finite = bool(torch.isfinite(actual).all())
    difference = (actual - expected).abs()
    passed = finite and bool(torch.all(difference <= atol + rtol * expected.abs()))
    return {"passed": passed, "max_abs_error": float(difference.max()),
            "rmse": float((difference.square().mean()).sqrt()), "atol": atol, "rtol": rtol}


def gpu_timing(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"warmup": warmup, "iterations": iterations, "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "incremental_peak_bytes": torch.cuda.max_memory_allocated() - baseline,
            "timing_scope": "GPU同步的PyTorch端到端wall time，包含Python launch成本，非RTL或custom kernel"}


@torch.inference_mode()
def run(spec, device="cuda"):
    cfg, shape, output = protocol()["layout"], spec["shape"], Path(spec["output"])
    model = load(spec["parent"]["checkpoint"]).to(device).eval()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    before = {n: v.detach().clone() for n, v in model.state_dict().items()}
    rows, first_image, images = [], None, {}
    for task in cfg["tasks"]:
        ds, _, _ = dataset(task, "val", shape)
        item = ds[0]  # Diagnostic only. Formal AP is evaluated on complete membership separately.
        image = item["img"][None].to(device, dtype=torch.float32) / 255
        images[task] = item["im_file"]
        first_image = image if first_image is None else first_image
        captures, handles = {}, []
        for index in (16, 19, 22):
            handles.append(model.graph.model[index].register_forward_hook(
                lambda m, args, out, index=index: captures.__setitem__(f"P{index}", out.detach().clone())))
        for index in (17, 20):
            handles.append(model.graph.model[index].register_forward_hook(
                lambda m, args, out, index=index: captures.__setitem__(f"rep{index}", (args[0].detach().clone(), out.detach().clone()))))
        for site in ("graph.model.10.m.0.attn", "graph.model.22.m.0.1.attn"):
            attn = model.get_submodule(site)
            handles.append(attn.normalizer.register_forward_hook(
                lambda m, args, out, site=site: captures.__setitem__(site + "/normalizer", (args[0].detach().clone(), out.detach().clone()))))
            handles.append(attn.aggregate.register_forward_hook(
                lambda m, args, out, site=site: captures.__setitem__(site + "/PV", (args[1].detach().clone(), out.detach().clone()))))
        try:
            model(image, task=task)
        finally:
            for handle in handles:
                handle.remove()
        for index in (16, 19, 22):
            x = captures[f"P{index}"]
            for tile in cfg["spatial_tiles"]:
                restored, info = tile_roundtrip(x, tile)
                pitched, pitch_info = pitch_roundtrip(x, tile)
                passed = torch.equal(restored, x) and torch.equal(pitched, x)
                rows.append({"task": task, "module": f"graph.model.{index}", "kind": "layout", "passed": passed,
                             **info, **pitch_info})
        for index in (17, 20):
            x, expected = captures[f"rep{index}"]
            block = model.graph.model[index]
            for tile in cfg["spatial_tiles"]:
                actual = tiled_conv(x, block.conv, tile, block.act)
                rows.append({"task": task, "module": f"graph.model.{index}", "kind": "conv_halo_qsilu", "tile": tile,
                             **error(actual, expected, atol=cfg["conv_atol"], rtol=cfg["conv_rtol"])})
        for site in ("graph.model.10.m.0.attn", "graph.model.22.m.0.1.attn"):
            scores, expected_p = captures[site + "/normalizer"]
            values, expected_pv = captures[site + "/PV"]
            for tile in cfg["query_tiles"]:
                probability, pv = tiled_normalization(model.get_submodule(site).normalizer, scores, values, tile)
                comparison = error(pv, expected_pv, atol=cfg["PV_atol"], rtol=cfg["PV_rtol"])
                rows.append({"task": task, "module": site, "kind": "query_rows_full_keys", "tile": tile,
                             "keys": scores.shape[-1], "probability_bitwise_equal": torch.equal(probability, expected_p),
                             **comparison, "passed": comparison["passed"] and torch.equal(probability, expected_p)})
        if not all(r["passed"] for r in rows):
            write(output / "failure.json", rows)
            raise ValueError("分塊replay未達等價門檻，不能默默換算子")
        event(output, kind="layout_diagnostics", candidate=spec["candidate"], task=task, cases=len(rows), image=item["im_file"])
        del captures, ds
    unchanged = all(torch.equal(before[n], v) for n, v in model.state_dict().items())
    if not unchanged:
        raise ValueError("replay 改變了模型 state")
    del before
    del x, restored, pitched, expected, actual, scores, expected_p, values, expected_pv, probability, pv, image, item
    verify(model)
    timing = gpu_timing(lambda: model(first_image, task="both"), cfg["GPU_benchmark_warmup"],
                        cfg["GPU_benchmark_iterations"]) if device == "cuda" else None
    result = {"status": "passed", "candidate": spec["candidate"], "shape": shape,
              "parent": spec["parent"], "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
              "dtype": "FP32", "batch": 1, "rows": rows, "diagnostic_images": images,
              "task_AP": "另以完整5000/683張驗證；此處每任務一張僅供算子replay",
              "state_unchanged": unchanged, "operators_verified": True,
              "whole_model_both_timing": timing, "limitations": cfg["limitations"],
              "hardware_synthesis": False}
    write(output / "result.json", result)
    return result
