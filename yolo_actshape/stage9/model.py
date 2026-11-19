"""沿用已驗證的 folded Rep 模型；形狀只作獨立 manifest，不更動 Attention。"""

import torch
from binaryqk.train import atomic_torch, state_cpu
from stage8 import model as upstream
from .common import ROOT, protocol, read, digest, write


def load(path=None):
    path = ROOT / (path or protocol()["parent"]["checkpoint"])
    model = upstream.load(path)
    if model.stage8_manifest["candidate"] != "rep17_20" or model.stage8_manifest["mode"] != "folded":
        raise ValueError("必須繼承選定的 folded qSiLU Rep17＋20")
    if model.stage8_manifest["operators"] != "hardware":
        raise ValueError("D1 不得回退 Float PWL")
    sidecar = read(path.with_suffix(".json"))
    model.stage9_manifest = sidecar.get("stage9_manifest")
    if model.stage9_manifest is not None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("stage9_manifest") != model.stage9_manifest:
            raise ValueError("shape manifest 與權重封裝不符")
        if model.stage9_manifest["preprocessing"] != protocol()["preprocessing"]:
            raise ValueError("shape 前處理契約改變")
    return model


def verify(model):
    manifest = upstream.manifest_for(model)
    if manifest["candidate"] != "rep17_20" or manifest["mode"] != "folded" or manifest["operators"] != "hardware":
        raise ValueError("Stage9 必須保持 qSiLU Rep17＋20 的部署契約")
    return manifest


def save(path, model, shape_manifest):
    path = type(ROOT)(path)
    manifest = verify(model)
    weights = state_cpu(model)
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()) for v in weights.values()):
        raise ValueError("Stage9 權重必須有限 FP32")
    atomic_torch(path, {"manifest": manifest, "state_dict": weights, "stage9_manifest": shape_manifest})
    write(path.with_suffix(".json"), {"sha256": digest(path), "manifest": manifest, "stage9_manifest": shape_manifest})


def role(name):
    if ".attn." in name or ".dfl." in name:
        return "fixed"
    if ".detect_head." in name:
        return "detect_head"
    if ".pose_head." in name:
        return "pose_head"
    if name.startswith("graph.model.") and int(name.split(".")[2]) >= 11:
        return "neck"
    return "fixed"


def configure(model):
    verify(model)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad_(role(name) != "fixed")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    return set()


def groups(model):
    result = {"shared": [], "detect_head": [], "pose_head": []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            group = role(name)
            result[group if group in ("detect_head", "pose_head") else "shared"].append(param)
    return result


def optimizer(model, recipe):
    grouped = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grouped.setdefault((role(name), param.ndim >= 2), []).append(param)
    return torch.optim.AdamW([
        {"params": ps, "lr": recipe["lrs"][name], "weight_decay": .00027 if decay else 0.,
         "group_name": f"{name}-{int(decay)}"} for (name, decay), ps in grouped.items()
    ], betas=(.948, .999))
