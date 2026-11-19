"""數值、alias、嚴格載入與實際 Attention score 的診斷。"""

import gc
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import torch
from activation_lab.activations import build_activation
from activation_lab.hardware import (
    FixedPointConfig,
    emulate_poly_shift_int,
    emulate_qsilu_pq_int,
    hardware_proxy,
)
from torch import nn
from yolo_attention.stage3_checkpoint import SITES

from . import model as checkpoint
from .common import CANDIDATES, HERE, write_json
from .prepare import assert_nested_equal, tensors


def numerics():
    torch.set_num_threads(2)
    result = {}
    for name in CANDIDATES:
        act = build_activation(name)
        x = torch.linspace(-16, 16, 32769, dtype=torch.float64, requires_grad=True)
        y = act(x)
        grad = torch.autograd.grad(y.sum(), x)[0]
        ref = nn.functional.silu(x.detach())
        target = torch.sigmoid(x.detach()) * (
            1 + x.detach() * (1 - torch.sigmoid(x.detach()))
        )
        dtype_rows = {}
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            v = x.detach().to(dt).requires_grad_(True)
            output = act(v)
            derivative = torch.autograd.grad(output.sum(), v)[0]
            tail = torch.tensor(
                [-60000.0, -1000.0, -8.0, 0.0, 8.0, 1000.0, 60000.0],
                dtype=dt,
                requires_grad=True,
            )
            ty = act(tail)
            tg = torch.autograd.grad(ty.sum(), tail)[0]
            finite = bool(
                torch.isfinite(output).all()
                and torch.isfinite(derivative).all()
                and torch.isfinite(ty).all()
                and torch.isfinite(tg).all()
            )
            if not finite:
                raise FloatingPointError(f"{name}/{dt} 前向或梯度非有限")
            dtype_rows[str(dt)] = {
                "finite": finite,
                "max_error_vs_float64": float(
                    (output.detach().double() - act(v.detach().double())).abs().max()
                ),
                "max_derivative_error_vs_float64_grid": float(
                    (derivative.double() - grad).abs().max()
                ),
                "tails": ty.detach().tolist(),
                "tail_derivatives": tg.tolist(),
            }
        knots = (
            [0, 1, 2, 4, 8]
            if name == "qsilu_pq"
            else [0, 3]
            if name == "hardswish"
            else [0, act.profile.threshold]
        )
        boundaries = []
        for knot in sorted(set(knots + [-k for k in knots])):
            xx = torch.tensor(
                [knot - 1e-7, knot + 1e-7], dtype=torch.float64, requires_grad=True
            )
            yy = act(xx)
            gg = torch.autograd.grad(yy.sum(), xx)[0]
            boundaries.append(
                {
                    "x": knot,
                    "value_jump": float(abs(yy[1] - yy[0]).detach()),
                    "derivative_jump": float(abs(gg[1] - gg[0])),
                }
            )
        if name != "hardswish" and any(
            r["value_jump"] > 1e-5 or r["derivative_jump"] > 1e-5 for r in boundaries
        ):
            raise ValueError(f"{name} C1 邊界不連續")
        fixed = None
        if name in ("qsilu_pq", "poly_shift"):
            cfg = FixedPointConfig()
            codes = torch.arange(cfg.qmin, cfg.qmax + 1, dtype=torch.int64)
            emulator = (
                emulate_qsilu_pq_int if name == "qsilu_pq" else emulate_poly_shift_int
            )
            actual = emulator(codes, cfg).double() / cfg.scale
            floating = act(codes.double() / cfg.scale)
            fixed = {
                "config": asdict(cfg),
                "exhaustive_codes": len(codes),
                "max_abs_error": float((actual - floating).abs().max()),
                "mean_abs_error": float((actual - floating).abs().mean()),
                "scope": "activation 局部 reference；任務模型仍浮點 activation",
                "rounding": "canonical hardware.py signed magnitude round-shift; final saturation",
                "container": "int64",
            }
        result[name] = {
            "profile": checkpoint.definition(name),
            "max_error_vs_silu": float((y.detach() - ref).abs().max()),
            "rmse_vs_silu": float((y.detach() - ref).square().mean().sqrt()),
            "max_derivative_error_vs_silu": float((grad - target).abs().max()),
            "boundaries": boundaries,
            "dtype": dtype_rows,
            "fixed_point": fixed,
            "cost_proxy": asdict(hardware_proxy(name)),
        }
    write_json(
        HERE / "numerics.json",
        {"status": "passed", "candidates": result, "gpu_used": False},
    )
    return result


def cpu_contracts():
    torch.set_num_threads(2)
    torch.manual_seed(20261003)
    x = torch.rand(1, 3, 160, 160)
    rows = []
    for name in CANDIDATES:
        model = checkpoint.create(name)
        manifest = checkpoint.manifest_for(model)
        with torch.inference_mode():
            outputs = model(x, task="both")
            for task in ("detect", "pose"):
                assert_nested_equal(outputs[task], model(x, task=task)[task])
        monitor = RuntimeDiagnostics(model, max_calls=1)
        try:
            with torch.inference_mode():
                observed = model(x, task="both")
            assert_nested_equal(outputs, observed)
        finally:
            diagnostics = monitor.close()
        if len(diagnostics["sites"]) != 163 or len(diagnostics["attention"]) != 2:
            raise AssertionError("runtime hook call-site coverage differs")
        if not all(torch.isfinite(t).all() for t in tensors(outputs)):
            raise FloatingPointError("CPU 輸出非有限")
        with tempfile.TemporaryDirectory(
            prefix="stage7-contract-", dir="/tmp"
        ) as folder:
            path = Path(folder) / "model.pt"
            checkpoint.save(path, model)
            restored = checkpoint.load(path)
            with torch.inference_mode():
                assert_nested_equal(outputs, restored(x, task="both"))
        path = next(iter(manifest["activation"]["sites"]))
        old = model.get_submodule(path)
        model.set_submodule(path, nn.SiLU())
        try:
            checkpoint.manifest_for(model)
        except ValueError:
            pass
        else:
            raise AssertionError("未拒絕 activation fallback")
        model.set_submodule(path, old)
        if name == "qsilu_pq":
            original = old.profile
            old.profile = replace(original, threshold=9.0)
            try:
                checkpoint.manifest_for(model)
            except ValueError:
                pass
            else:
                raise AssertionError("未拒絕 qSiLU 係數改變")
            old.profile = original
        normalizer = model.get_submodule("graph." + SITES[0]).normalizer
        table = normalizer.reciprocal.table
        with torch.no_grad():
            table[0] += 1
        try:
            checkpoint.manifest_for(model)
        except ValueError:
            pass
        else:
            raise AssertionError("未拒絕 reciprocal 常數改變")
        with torch.no_grad():
            table[0] -= 1
        checkpoint.manifest_for(model)
        rows.append(
            {
                "candidate": name,
                "registered_sites": len(manifest["activation"]["sites"]),
                "alias_groups": len(manifest["activation"]["alias_groups"]),
                "parameters": sum(p.numel() for p in model.parameters()),
                "strict_reload_equal": True,
                "runtime_hooks_preserve_outputs": True,
                "observed_deployment_sites": len(diagnostics["sites"]),
                "observed_attention_sites": len(diagnostics["attention"]),
                "task_routing_equal": True,
                "reject_activation_or_reciprocal_fallback": True,
            }
        )
        del model, restored, outputs
        gc.collect()
    model = checkpoint.create("poly_shift", "neck")
    checkpoint.manifest_for(model)
    # A selective policy must keep every other site native SiLU, including flow.
    assert all(
        (d["name"] == "poly_shift") == (d["region"] == "neck")
        for d in model.stage7_manifest["activation"]["sites"].values()
    )
    write_json(
        HERE / "cpu-contracts.json",
        {
            "status": "passed",
            "cases": rows,
            "selective_policy": "passed",
            "gpu_used": False,
        },
    )
    return rows


class RuntimeDiagnostics:
    """Observe early real validation batches; use owner context for shared activations."""

    def __init__(self, model, max_calls=16):
        self.handles, self.site_rows, self.attention_rows = [], {}, {}
        self.context = []
        self.max_calls = max_calls
        policies = model.stage7_manifest["activation"]["sites"]
        unique = {}
        for path in policies:
            module = model.get_submodule(path)
            unique[id(module)] = module
            if path.endswith(".act"):
                owner = model.get_submodule(path.rsplit(".", 1)[0])
                self.handles.append(
                    owner.register_forward_pre_hook(
                        lambda m, args, path=path: self.context.append(path)
                    )
                )
                self.handles.append(owner.register_forward_hook(self.leave_owner))
        for module in unique.values():
            self.handles.append(module.register_forward_pre_hook(self.observe_input))
        for site in SITES:
            module = model.get_submodule("graph." + site).normalizer
            self.handles.append(
                module.register_forward_hook(
                    lambda m, args, out, site=site: self.observe_attention(
                        site, args[0], out
                    )
                )
            )

    def leave_owner(self, module, inputs, output):
        self.context.pop()
        # Forward hooks must return None so they never replace model outputs.

    def observe_input(self, module, inputs):
        if not self.context:
            return
        path = self.context[-1]
        row = self.site_rows.setdefault(
            path,
            {
                "calls": 0,
                "samples": 0,
                "min": float("inf"),
                "max": -float("inf"),
                "squared_error_sum": 0.0,
                "max_abs_error": 0.0,
            },
        )
        if row["calls"] >= self.max_calls:
            return
        x = inputs[0].detach().float().flatten()
        x = x[:: max(1, x.numel() // 2048)]
        with torch.no_grad():
            # forward directly avoids triggering this observation hook again.
            error = module.forward(x) - nn.functional.silu(x)
        row["calls"] += 1
        row["samples"] += x.numel()
        row["min"], row["max"] = (
            min(row["min"], float(x.min())),
            max(row["max"], float(x.max())),
        )
        row["squared_error_sum"] += float(error.square().sum())
        row["max_abs_error"] = max(row["max_abs_error"], float(error.abs().max()))

    def observe_attention(self, site, scores, probability):
        row = self.attention_rows.setdefault(
            site,
            {
                "calls": 0,
                "tokens": [],
                "score_min": float("inf"),
                "score_max": -float("inf"),
                "shifted_min": 0.0,
                "max_row_sum_error": 0.0,
                "tail_clamped": 0,
                "elements": 0,
            },
        )
        if row["calls"] >= self.max_calls:
            return
        s, p = scores.detach().float(), probability.detach().float()
        if (
            not torch.isfinite(s).all()
            or not torch.isfinite(p).all()
            or p.min() < 0
            or p.max() > 1
            or s.shape[-1] > 1024
        ):
            raise ValueError("真實 Attention 數值契約失效")
        shifted = s - s.amax(-1, keepdim=True)
        row["calls"] += 1
        row["tokens"] = sorted(set(row["tokens"] + [s.shape[-1]]))
        row["score_min"], row["score_max"] = (
            min(row["score_min"], float(s.min())),
            max(row["score_max"], float(s.max())),
        )
        row["shifted_min"] = min(row["shifted_min"], float(shifted.min()))
        row["max_row_sum_error"] = max(
            row["max_row_sum_error"], float((p.sum(-1) - 1).abs().max())
        )
        row["tail_clamped"] += int((shifted < -10).sum())
        row["elements"] += s.numel()
        if row["max_row_sum_error"] > 0.01:
            raise ValueError("D1 probability row-sum 超過已知誤差範圍")

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self.context:
            raise ValueError("activation owner context 未還原")
        return {
            "sites": self.site_rows,
            "attention": self.attention_rows,
            "sampling": "每個 owner 前 16 次完整 val forward，每次最多約 2048 elements；診斷不改 evaluator split",
            "fixed_normalization_preserved": True,
        }
