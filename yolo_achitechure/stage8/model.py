"""Rep 的 training／folded 格式嚴格重建；原 Stage7 metadata 保持不可變。"""

import copy
import json
from pathlib import Path

import torch
from binaryqk.train import atomic_torch, state_cpu
from stage6.models import from_manifest
from stage7 import model as previous
from torch import nn

from .common import protocol, read, sha256, write_json
from .rep import FoldedRepConv, RepConv, AsymmetricRepConv, from_conv, branches
from .operators import set_operators, validate_operators


def install(model, candidate, mode):
    for index in protocol()["layers"][candidate]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1702 + index)
            factory = AsymmetricRepConv if protocol()["candidate_specs"][candidate]["variant"] == "acb" else from_conv
            rep = factory(model.graph.model[index]).eval()
            model.graph.model[index] = (
                rep if mode == "training" else FoldedRepConv(rep)
            )


def create(candidate):
    p = protocol(verify=True)
    if candidate not in p["candidates"]:
        raise ValueError(candidate)
    model = previous.load(p["parent"]["checkpoint"])
    spec = p["candidate_specs"][candidate]
    policy = set_operators(model, spec["operators"], p["layers"][candidate])
    contract = {}
    for index in p["layers"][candidate]:
        old = model.graph.model[index]
        c = old.conv
        contract[str(index)] = {
            "in_channels": c.in_channels,
            "out_channels": c.out_channels,
            "kernel_size": list(c.kernel_size),
            "stride": list(c.stride),
            "padding": list(c.padding),
            "groups": c.groups,
            "bn_eps": old.bn.eps,
            "bn_momentum": old.bn.momentum,
        }
    install(model, candidate, "training")
    model.stage8_manifest = {
        "schema": "repconv-stage8-v3",
        "candidate": candidate,
        "layers": p["layers"][candidate],
        "mode": "training",
        "operators": spec["operators"],
        "variant": spec["variant"],
        "activation_policy": policy,
        "operator_contract": copy.deepcopy(p["operator_policy"]),
        "parent": p["parent"],
        "stage7_manifest": copy.deepcopy(model.stage7_manifest),
        "layer_contract": contract,
        "training_lineage": [],
    }
    manifest_for(model)
    return model.eval()


def manifest_for(model):
    data = model.stage8_manifest
    p = protocol()
    if (
        data["schema"] != "repconv-stage8-v3"
        or data["candidate"] not in p["candidates"]
        or data["mode"] not in ("training", "folded")
        or data["parent"] != p["parent"]
        or data["layers"] != p["layers"][data["candidate"]]
    ):
        raise ValueError("Rep manifest／parent 不符")
    if model.stage7_manifest != data["stage7_manifest"] or data["operator_contract"] != p["operator_policy"]:
        raise ValueError("上游 manifest 或訓練／部署算子契約改變")
    spec = p["candidate_specs"][data["candidate"]]
    if data["variant"] != spec["variant"] or data["operators"] not in (spec["operators"], "hardware"):
        raise ValueError("候選結構／算子模式不符")
    validate_operators(model, data["operators"])
    expected = (AsymmetricRepConv if data["variant"] == "acb" else RepConv) if data["mode"] == "training" else FoldedRepConv
    paths = {
        n
        for n, m in model.named_modules()
        if isinstance(m, (RepConv, AsymmetricRepConv, FoldedRepConv))
    }
    if paths != {f"graph.model.{i}" for i in data["layers"]}:
        raise ValueError("Rep 位置改變")
    for path in paths:
        block = model.get_submodule(path)
        if type(block) is not expected:
            raise ValueError("Rep training／folded 結構不符")
        desc = data["layer_contract"][path.split(".")[-1]]
        conv = block.conv1.conv if data["mode"] == "training" else block.conv
        for key in (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "groups",
        ):
            value = getattr(conv, key)
            if (list(value) if isinstance(value, tuple) else value) != desc[key]:
                raise ValueError(f"Rep Conv 設定改變：{key}")
        if data["mode"] == "training":
            if block.bn is not None or any(type(b.act) is not nn.Identity for b in branches(block)):
                raise ValueError("Rep 分支必須線性且不含 identity BN")
            expected_shapes = [((3,3), (1,1)), ((1,3), (0,1)), ((3,1), (1,0))] if data["variant"] == "acb" else [((3,3),(1,1)), ((1,1),(0,0))]
            for branch, (kernel, padding) in zip(branches(block), expected_shapes, strict=True):
                c = branch.conv
                if (c.kernel_size != kernel or c.padding != padding or c.stride != conv.stride
                        or c.groups != conv.groups or c.in_channels != conv.in_channels
                        or c.out_channels != conv.out_channels or c.bias is not None
                        or branch.bn.eps != desc["bn_eps"] or branch.bn.momentum != desc["bn_momentum"]
                        or branch.bn.training):
                    raise ValueError("Rep 分支 geometry／BN 契約改變")
        elif any(
            isinstance(m, nn.modules.batchnorm._BatchNorm) for m in block.modules()
        ):
            raise ValueError("folded Rep 仍含 BN")
    return copy.deepcopy(data)


def folded(model, operators=None):
    """operators=None 保持原算子，只驗證線性折疊；hardware 才做部署替換。"""
    manifest_for(model)
    result = copy.deepcopy(model).cpu().float().eval()
    if result.stage8_manifest["mode"] == "training":
        for index in result.stage8_manifest["layers"]:
            result.graph.model[index] = FoldedRepConv(result.graph.model[index])
        result.stage8_manifest["mode"] = "folded"
    if operators is not None:
        result.stage8_manifest["activation_policy"] = set_operators(result, operators)
        result.stage8_manifest["operators"] = operators
    manifest_for(result)
    return result


def save(path, model):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    manifest = manifest_for(model)
    state = state_cpu(model)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in state.values()
    ):
        raise ValueError("checkpoint 需有限 FP32")
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
    data = torch.load(path, map_location="cpu", weights_only=True)
    m = data["manifest"]
    if json.loads(json.dumps(m)) != sidecar["manifest"]:
        raise ValueError("sidecar 不符")
    model = from_manifest(m["stage7_manifest"]["parent_manifest"])
    installed = previous.install(model, "qsilu_pq")
    if installed != m["stage7_manifest"]["activation"]:
        raise ValueError("qSiLU 重建不符")
    model.stage7_manifest = copy.deepcopy(m["stage7_manifest"])
    policy = set_operators(model, m["operators"], m["layers"])
    if policy != m["activation_policy"]:
        raise ValueError("activation policy 重建失敗")
    install(model, m["candidate"], m["mode"])
    model.stage8_manifest = copy.deepcopy(m)
    model.load_state_dict(data["state_dict"], strict=True)
    model.eval()
    manifest_for(model)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in model.state_dict().values()
    ):
        raise ValueError("非有限 FP32 權重")
    return model


def role(name, layers):
    if ".attn." in name or ".dfl." in name:
        return "fixed"
    if ".detect_head." in name:
        return "detect_head"
    if ".pose_head." in name:
        return "pose_head"
    return (
        "rep" if any(name.startswith(f"graph.model.{i}.") for i in layers) else "fixed"
    )


def configure(model, recipe):
    if model.stage8_manifest["mode"] != "training" or model.stage8_manifest["operators"] != recipe["operators"]:
        raise ValueError("folded 模型不得當多分支 resume")
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(role(name, model.stage8_manifest["layers"]) != "fixed")
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return set()
