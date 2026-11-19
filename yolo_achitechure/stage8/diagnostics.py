"""CPU 真實模型驗證與算子盤點；不以零分支 folding 充當全部證據。"""

import copy
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from activation_lab.activations import ShiftPiecewiseQuadraticSiLU
from stage7 import model as parent_model
from stage7.prepare import tensors
from torch.utils._python_dispatch import TorchDispatchMode

from . import model as checkpoint
from .common import ARTIFACTS, CANDIDATES, HERE, protocol, sha256, write_json
from .rep import FoldedRepConv, branches
from .operators import set_operators


def compare(outputs, reference, atol=2e-4, rtol=1e-5):
    """Raw anchor 對齊；decoded 允許 top-k 排列但必須完整一對一配對。"""
    maxima = {}
    if set(outputs) != set(reference):
        raise AssertionError("任務輸出不同")
    for task in outputs:
        a, ar = outputs[task]
        b, br = reference[task]
        raw_max = 0.0
        for x, y in zip(tensors(ar), tensors(br), strict=True):
            torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
            raw_max = max(raw_max, float((x.float() - y.float()).abs().max()))
        decoded_max = 0.0
        for x, y in zip(a, b, strict=True):
            if x.shape != y.shape:
                raise AssertionError("decoded shape 不同")
            if torch.allclose(x, y, atol=atol, rtol=rtol):
                decoded_max = max(
                    decoded_max, float((x.float() - y.float()).abs().max())
                )
                continue
            compatible = torch.isclose(
                x[:, None, :], y[None, :, :], atol=atol, rtol=rtol
            ).all(-1)
            neighbors = [row.nonzero().flatten().tolist() for row in compatible]
            matched = {}

            def augment(i, seen, neighbors=neighbors, matched=matched):
                for j in neighbors[i]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if j not in matched or augment(matched[j], seen):
                        matched[j] = i
                        return True
                return False

            if not all(
                augment(i, set())
                for i in sorted(range(len(x)), key=lambda i: len(neighbors[i]))
            ):
                raise AssertionError(f"{task} decoded 無法在原容差內一對一配對")
            delta = (x[list(matched.values())].float() - y[list(matched)].float()).abs()
            decoded_max = max(decoded_max, float(delta.max()))
        maxima[task] = {"raw_max_abs": raw_max, "matched_decoded_max_abs": decoded_max}
    return maxima


def fold_check(model, size=160):
    """已訓權重同樣使用此檢查；不改 live／EMA。"""
    live = copy.deepcopy(model).cpu().float().eval()
    deploy = checkpoint.folded(live)
    torch.manual_seed(8170 + size)
    x = torch.rand(1, 3, size, size)
    with torch.inference_mode():
        result = compare(deploy(x, task="both"), live(x, task="both"))
    return result


def audit(model):
    checkpoint.manifest_for(model)
    hardware = model.stage8_manifest["operators"] == "hardware"
    stack, operations, counts, hooks = [], defaultdict(Counter), Counter(), []
    for name, module in model.named_modules():

        def before(_module, _args, name=name):
            stack.append(name)
            counts[name] += 1

        def after(_module, _args, _out):
            stack.pop()

        hooks.append(module.register_forward_pre_hook(before))
        hooks.append(module.register_forward_hook(after, always_call=True))

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operations[stack[-1] if stack else "<functional>"][str(func)] += 1
            return func(*args, **(kwargs or {}))

    try:
        with torch.inference_mode(), Observe():
            output = model(torch.rand(1, 3, 160, 160), task="both")
        if not all(torch.isfinite(t).all() for t in tensors(output)):
            raise ValueError("算子盤點輸出非有限")
    finally:
        for hook in hooks:
            hook.remove()
    normalizers = [
        p for p, m in model.named_modules() if type(m).__name__ == ("CompletePWLSoftmax" if hardware else "PiecewiseLinearSoftmax")
    ]
    qsilu_calls = sum(
        counts[p]
        for p, m in model.named_modules()
        if isinstance(m, ShiftPiecewiseQuadraticSiLU)
    )
    silu_calls = sum(counts[p] for p,m in model.named_modules() if isinstance(m, torch.nn.SiLU))
    if (model.stage8_manifest["operators"] == "rep_silu_pwl") != (silu_calls > 0):
        raise ValueError("SiLU 執行範圍與模式不符")
    if (
        len(normalizers) != 2
        or not all(counts[p] > 0 for p in normalizers)
        or qsilu_calls == 0
    ):
        raise ValueError("qSiLU／D1 未實際執行")
    forbidden = []
    remaining = []
    for path, ops in operations.items():
        for op, count in ops.items():
            if hardware and (".attn.normalizer" in path or path.endswith(".act")) and any(
                k in op for k in ("aten.exp.", "aten._softmax.", "aten.silu.")
            ):
                forbidden.append([path, op, count])
            if any(k in op for k in ("sigmoid", "div", "sqrt", "batch_norm")):
                remaining.append({"path": path, "op": op, "count": count})
    if forbidden:
        raise ValueError(f"已簡化算子回退：{forbidden}")
    return {
        "status": "passed",
        "input": [1, 3, 160, 160],
        "device": "cpu",
        "dtype": "FP32",
        "normalizer_call_sites": {p: counts[p] for p in normalizers},
        "qsilu_calls": qsilu_calls,
        "silu_calls": silu_calls,
        "operators": model.stage8_manifest["operators"],
        "operations_by_owner": {p: dict(v) for p, v in operations.items()},
        "remaining_operations": remaining,
        "interpretation": "head confidence／visibility sigmoid 保留原語意；整數 div／移位與浮點除法需按 owner 分開。Rep BN 已 fold，其餘 ConvBN 可離線融合，未在本輪擴張全網格式。無 RTL/HLS 實測。",
    }


def run():
    torch.set_num_threads(2)
    protocol(verify=True)
    torch.manual_seed(8170)
    base = parent_model.load(protocol()["parent"]["checkpoint"])
    inputs = torch.rand(1, 3, 160, 160)
    with torch.inference_mode():
        hardware_expected = base(inputs, task="both")

    # 比較器確實拒絕內容錯誤，僅允許合法排序交換。
    small = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 0.01, 1.0], [3.0, 4.0, 5.0, 6.0, 0.02, 2.0]]]
    )
    compare(
        {"detect": (small.flip(1), [torch.ones(2)])},
        {"detect": (small, [torch.ones(2)])},
    )
    bad = small.clone()
    bad[0, 0, 5] += 1
    try:
        compare(
            {"detect": (bad, [torch.ones(2)])}, {"detect": (small, [torch.ones(2)])}
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("比較器未拒絕類別改變")
    results = {}
    for name in CANDIDATES:
        model = checkpoint.create(name)
        set_operators(base, model.stage8_manifest["operators"], model.stage8_manifest["layers"])
        with torch.inference_mode():
            expected = base(inputs, task="both")
            initial = compare(model(inputs, task="both"), expected)
        with torch.inference_mode():
            initial_hardware = compare(checkpoint.folded(model, operators="hardware")(inputs, task="both"), hardware_expected)
        native_ops = audit(checkpoint.folded(model))
        write_json(HERE / f"operators-training-{name}.json", native_ops)
        nonzero = copy.deepcopy(model).eval()
        unit = {}
        for index in model.stage8_manifest["layers"]:
            rep = nonzero.graph.model[index]
            with torch.no_grad():
                for branch in branches(rep)[1:]:
                    branch.bn.weight.fill_(0.1)
                    branch.bn.bias.uniform_(-0.02, 0.02)
                    branch.bn.running_mean.uniform_(-0.1, 0.1)
                    branch.bn.running_var.uniform_(0.8, 1.2)
                    branch.conv.weight.add_(torch.randn_like(branch.conv.weight) * 0.001)
            deploy = FoldedRepConv(rep)
            x = torch.randn(2, rep.conv1.conv.in_channels, 13, 18)
            errors = {}
            for dtype in (torch.float32, torch.float16):
                a, b = copy.deepcopy(rep).to(dtype), copy.deepcopy(deploy).to(dtype)
                with torch.inference_mode():
                    y, z = sum(branch(x.to(dtype)) for branch in branches(a)), b.conv(x.to(dtype))
                    atol, rtol = (
                        (2e-4, 1e-5) if dtype == torch.float32 else (0.03, 0.02)
                    )
                    torch.testing.assert_close(y, z, atol=atol, rtol=rtol)
                    torch.testing.assert_close(
                        a(x.to(dtype)), b(x.to(dtype)), atol=atol, rtol=rtol
                    )
                errors[str(dtype)] = {
                    "linear_max_abs": float((y.float() - z.float()).abs().max()),
                    "atol": atol,
                    "rtol": rtol,
                }
            unit[str(index)] = errors
        full_nonzero = fold_check(nonzero)
        full640 = fold_check(nonzero, 640)
        deploy = checkpoint.folded(nonzero)
        roundtrip = {}
        with tempfile.TemporaryDirectory(prefix="stage8-cpu-") as temp:
            for mode, candidate in (("multibranch-training-operators", nonzero), ("folded-training-operators", deploy),
                                    ("folded-hardware", checkpoint.folded(nonzero, operators="hardware"))):
                path = Path(temp) / f"{mode}.pt"
                checkpoint.save(path, candidate)
                loaded = checkpoint.load(path)
                with torch.inference_mode():
                    roundtrip[mode] = compare(
                        loaded(inputs, task="both"),
                        candidate(inputs, task="both"),
                        atol=0,
                        rtol=0,
                    )
                for task in ("detect", "pose"):
                    with torch.inference_mode():
                        compare(
                            loaded(inputs, task=task),
                            {task: loaded(inputs, task="both")[task]},
                            atol=0,
                            rtol=0,
                        )
        for path, replacement in (("graph.model.10.m.0.attn.normalizer", torch.nn.Softmax(-1)),
                                  ("graph.model.0.act", torch.nn.SiLU())):
            old = nonzero.get_submodule(path)
            nonzero.set_submodule(path, replacement)
            try:
                checkpoint.manifest_for(nonzero)
            except ValueError:
                pass
            else:
                raise AssertionError("未拒絕算子／alias 變更")
            nonzero.set_submodule(path, old)
        deploy = checkpoint.folded(nonzero, operators="hardware")
        original = deploy.graph.model[model.stage8_manifest["layers"][0]].act
        deploy.graph.model[model.stage8_manifest["layers"][0]].act = torch.nn.SiLU()
        try:
            checkpoint.manifest_for(deploy)
        except ValueError:
            pass
        else:
            raise AssertionError("未拒絕 SiLU 回退")
        deploy.graph.model[model.stage8_manifest["layers"][0]].act = original
        checkpoint.manifest_for(deploy)
        ops = audit(deploy)
        write_json(HERE / f"operators-{name}.json", ops)
        initial_path = ARTIFACTS / protocol()["run_id"] / "initial" / f"{name}.pt"
        if initial_path.exists():
            saved = checkpoint.load(initial_path)
            assert all(
                torch.equal(v, saved.state_dict()[k])
                for k, v in model.state_dict().items()
            )
        else:
            checkpoint.save(initial_path, model)
        results[name] = {
            "initial_equivalence": initial,
            "initial_hardware_equivalence_to_stage7": initial_hardware,
            "fold_equivalence_operators": model.stage8_manifest["operators"],
            "training_operators": str(HERE / f"operators-training-{name}.json"),
            "nonzero_fold": full_nonzero,
            "nonzero_fold_640": full640,
            "linear_and_activation_dtypes": unit,
            "roundtrip": roundtrip,
            "fallback_rejected": True,
            "initial_checkpoint": str(initial_path),
            "sha256": sha256(initial_path),
            "train_parameters": sum(p.numel() for p in model.parameters()),
            "folded_parameters": sum(p.numel() for p in deploy.parameters()),
            "operators": str(HERE / f"operators-{name}.json"),
        }
        print(
            f"PASS CPU {name}: 初始／非零fold／兩dtype單元／160及640完整輸出／雙格式重載／算子",
            flush=True,
        )
        del model, nonzero, deploy
    if torch.cuda.is_initialized():
        raise ValueError("CPU 前置意外初始化 CUDA")
    result = {
        "status": "passed",
        "gpu_used": False,
        "results": results,
        "comparison": "FP32 raw anchor-aligned; decoded one-to-one same tolerance; FP16 block-only, task deployment FP32",
        "next_stage_started": False,
    }
    write_json(HERE / "cpu-contracts.json", result)
    print(
        json.dumps(
            {"status": "passed", "candidates": list(results)}, ensure_ascii=False
        )
    )
    return result


if __name__ == "__main__":
    run()
