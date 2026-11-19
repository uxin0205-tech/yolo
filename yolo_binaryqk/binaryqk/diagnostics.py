"""CPU 真實 Detect loss 的一步梯度與嚴格重載驗證；不是任務 AP。"""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics.cfg import DEFAULT_CFG_DICT

from . import checkpoint
from .basis import BASES
from .model import SITES, convert, frozen_hashes, load_parent, sha256, train_qk_only


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensors(v)]
    return []


def smoke(calibration_path, calibration):
    torch.manual_seed(0)
    parent = load_parent()
    image = torch.rand(2, 3, 96, 128)
    batch = {"img": image, "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.tensor([[32.], [34.]]),
             "bboxes": torch.tensor([[.5, .5, .3, .3], [.6, .4, .2, .5]])}
    report = {"device": "cpu", "dtype": "float32", "kind": "synthetic-Detect-loss-smoke-not-AP",
              "image_shape": list(image.shape), "candidates": {}}
    for variant in BASES:
        model = convert(copy.deepcopy(parent), variant, calibration["coefficients"][variant])
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        parameters = train_qk_only(model)
        before = frozen_hashes(model)
        train_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        counts, handles = {}, []
        for site in SITES:
            for name in ("normalizer", "aggregate"):
                key = f"{site}.{name}"

                def count_hook(module, args, result, key=key, counts=counts):
                    counts[key] = counts.get(key, 0) + 1

                handles.append(model.get_submodule(key).register_forward_hook(count_hook))
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.)
        loss, components = model.loss(batch)
        loss.sum().backward()
        for handle in handles:
            handle.remove()
        if counts != {f"{s}.{name}": 1 for s in SITES for name in ("normalizer", "aggregate")}:
            raise AssertionError(f"normalization/PV call count mismatch: {counts}")
        gradients = {n: float(p.grad.abs().max()) for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None}
        if set(gradients) != set(train_before) or any(not torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise AssertionError("missing/nonfinite Q/K gradients")
        for site in SITES:
            for part in ("q", "k"):
                if gradients[f"{site}.qkv.{part}.conv.weight"] <= 0:
                    raise AssertionError("Q/K projection gradient is zero")
        optimizer.step()
        changes = {n: float((p.detach() - train_before[n]).abs().max())
                   for n, p in model.named_parameters() if p.requires_grad}
        for site in SITES:
            for part in ("q", "k"):
                if changes[f"{site}.qkv.{part}.conv.weight"] <= 0:
                    raise AssertionError("Q/K projection did not update")
        if before != frozen_hashes(model):
            raise AssertionError("frozen parameters/BN buffers changed")
        model.eval()
        manifest = checkpoint.manifest_for(model, sha256(calibration_path))
        with tempfile.TemporaryDirectory(prefix="binaryqk-smoke-") as temporary:
            path = Path(temporary) / "diagnostic.pt"
            checkpoint.save(path, model, manifest)
            loaded, _ = checkpoint.load(path)
            with torch.inference_mode():
                expected, actual = tensors(model(image)), tensors(loaded(image))
            if len(expected) != len(actual):
                raise AssertionError("checkpoint output structure changed")
            reload_error = max(float((a - b).abs().max()) for a, b in zip(expected, actual))
            if reload_error != 0:
                raise AssertionError(f"CPU FP32 reload not exact: {reload_error}")
        report["candidates"][variant] = {
            "loss": float(loss.detach().sum()), "loss_components": components.detach().tolist(),
            "scale_slots": manifest["scale_slots"], "calls": counts,
            "gradient_abs_max": gradients, "update_abs_max": changes,
            "frozen_tensors_bitwise_equal": True, "reload_abs_max": reload_error,
        }
        print(f"{variant}: Q/K gradients, frozen tensors, single Softmax/PV, strict reload PASS", flush=True)
    return report
