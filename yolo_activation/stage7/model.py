"""自包含 activation manifest；先驗證替換，再驗證原 Attention 契約。"""

import copy
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from activation_lab.activations import build_activation, get_profile
from binaryqk.checkpoint import activations
from binaryqk.train import atomic_torch, state_cpu
from stage6 import checkpoint as previous
from stage6.models import from_manifest as base_model
from stage6.models import role, view
from torch import nn

from yolo_attention import stage3_checkpoint as stage3

from .common import CANDIDATES, HERE, REGIONS, protocol, read, sha256, write_json


def region(path):
    if ".flow_model." in path:
        return "pose_flow_training_only"
    if ".attn." in path:
        return "attention"
    if ".pose_head." in path:
        return "pose_head"
    if ".detect_head." in path:
        return "detect_head"
    index = int(path.split(".")[2])
    return (
        "backbone_early" if index <= 4 else "backbone_late" if index <= 10 else "neck"
    )


def definition(name):
    module = build_activation(name)
    profile = get_profile(name)
    # JSON canonical form keeps tuple coefficients identical before/after loading.

    return json.loads(
        json.dumps(
            {
                "name": name,
                "class": f"{type(module).__module__}.{type(module).__qualname__}",
                "profile": asdict(profile) if profile else None,
                "inplace": getattr(module, "inplace", None),
            }
        )
    )


def aliases(model, paths):
    groups = defaultdict(list)
    for path in paths:
        groups[id(model.get_submodule(path))].append(path)
    return sorted(sorted(g) for g in groups.values())


def install(model, name, selected_region=None):
    if name not in CANDIDATES or (
        selected_region is not None and selected_region not in REGIONS
    ):
        raise ValueError("非本輪候選")
    sites = [
        n
        for n, m in model.named_modules(remove_duplicate=False)
        if isinstance(m, nn.SiLU)
    ]
    expected = read(HERE / "preflight.json")["registered_silu_references"]
    if sites != expected:
        raise ValueError("parent activation 路徑改變")
    original_aliases = aliases(model, sites)
    assignments = {
        n: name if selected_region is None or region(n) == selected_region else "silu"
        for n in sites
    }
    cache = {}
    for path in sites:
        old = model.get_submodule(path)
        target = assignments[path]
        # Preserve every shared module alias within a given activation policy.
        key = (id(old), target)
        if key not in cache:
            cache[key] = old if target == "silu" else build_activation(target)
        model.set_submodule(path, cache[key])
    inventory = {n: {"region": region(n), **definition(assignments[n])} for n in sites}
    # Original SiLU may use inplace=True; record actual property for retained sites.
    for n in sites:
        inventory[n]["inplace"] = getattr(model.get_submodule(n), "inplace", None)
    return {
        "candidate": name,
        "region": selected_region,
        "sites": inventory,
        "original_alias_groups": original_aliases,
        "alias_groups": aliases(model, sites),
        "call_sites": read(HERE / "preflight.json")["executed_call_sites"],
        "precision": "canonical native input dtype; FP32 validation, AMP FP16 training",
        "repconv": None,
    }


def create(name, selected_region=None):
    p = protocol(verify=True)
    model = previous.load(p["parent"]["checkpoint"])
    activation = install(model, name, selected_region)
    model.stage7_manifest = {
        "schema": "activation-stage7-v1",
        "parent": p["parent"],
        "parent_manifest": copy.deepcopy(model.stage6_manifest),
        "activation": activation,
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "training_lineage": [],
    }
    manifest_for(model)
    return model.eval()


def manifest_for(model):
    data = model.stage7_manifest
    if (
        data["schema"] != "activation-stage7-v1"
        or model.stage6_manifest != data["parent_manifest"]
    ):
        raise ValueError("缺少 Stage7 manifest 或上游被改寫")
    policy = data["activation"]
    if policy["candidate"] not in CANDIDATES or policy["repconv"] is not None:
        raise ValueError("不合法 activation／未授權 RepConv")
    paths = policy["sites"]
    actual = {
        n
        for n, m in model.named_modules(remove_duplicate=False)
        if isinstance(m, (nn.SiLU, nn.Hardswish))
        or type(m).__module__ == "activation_lab.activations"
    }
    if actual != set(paths) or aliases(model, paths) != policy["alias_groups"]:
        raise ValueError("activation call-site／alias 改變")
    for path, desc in paths.items():
        target = (
            policy["candidate"]
            if policy["region"] is None or region(path) == policy["region"]
            else "silu"
        )
        expected = definition(target)
        if {k: desc[k] for k in ("name", "class", "profile")} != {
            k: expected[k] for k in ("name", "class", "profile")
        }:
            raise ValueError(f"activation manifest 與 policy 不符：{path}")
        module = model.get_submodule(path)
        if type(module) is not type(build_activation(target)) or getattr(
            module, "profile", None
        ) != get_profile(target):
            raise ValueError(f"activation 類別／固定係數不符：{path}")
        if getattr(module, "inplace", None) != desc["inplace"]:
            raise ValueError(f"activation inplace 改變：{path}")
    base = data["parent_manifest"]
    if model.contract() != base["graph"] or base["repconv"] is not None:
        raise ValueError("共享圖或 RepConv 改變")
    if any("masf" in n.lower() or "hog_aux" in n for n, _ in model.named_modules()):
        raise ValueError("不允許加入未選 MASF/HOG")
    # This shallow task view owns metadata only. Original Stage6 manifest is immutable.
    detect = view(model, "detect")
    detect.stage3_parent_manifest = copy.deepcopy(base["upstream"]["parent_manifest"])
    detect.stage3_parent_manifest["activation_calls"] = activations(detect)
    validated = stage3.manifest_for(detect)
    validated["activation_calls"] = base["upstream"]["activation_calls"]
    validated["parent_manifest"]["activation_calls"] = base["upstream"][
        "parent_manifest"
    ]["activation_calls"]
    if validated != base["upstream"]:
        raise ValueError("B2／Q10 scale／bias／完整 PWL reciprocal 契約改變")
    return copy.deepcopy(data)


def save(path, model):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("不得覆寫既有 checkpoint")
    manifest = manifest_for(model)
    state = state_cpu(model)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in state.values()
    ):
        raise ValueError("checkpoint 必須為有限 FP32")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch(path, {"manifest": manifest, "state_dict": state})
    write_json(
        path.with_suffix(".json"), {"sha256": sha256(path), "manifest": manifest}
    )


def load(path):
    path = Path(path)
    sidecar = read(path.with_suffix(".json"))
    if sha256(path) != sidecar["sha256"]:
        raise ValueError("checkpoint SHA 不符")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    data = payload["manifest"]
    if json.loads(json.dumps(data)) != sidecar["manifest"]:
        raise ValueError("checkpoint sidecar manifest 不符")
    model = base_model(data["parent_manifest"])
    installed = install(
        model, data["activation"]["candidate"], data["activation"]["region"]
    )
    if installed != data["activation"]:
        raise ValueError("activation 重建與 manifest 不符")
    model.stage7_manifest = copy.deepcopy(data)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in model.state_dict().values()
    ):
        raise ValueError("checkpoint state 非有限 FP32")
    manifest_for(model)
    return model.eval()


def configure(model, recipe):
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(role(name) != "fixed")
    mutable = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            is_head = ".pose_head." in name or ".detect_head." in name
            module.train(is_head)
            if is_head:
                mutable.update(f"{name}.{k}" for k, _ in module.named_buffers())
    return mutable
