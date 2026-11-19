"""一個 shared trunk；兩個原生 head；強制繼承 D1 完整數值契約。"""

import copy

import torch
from binaryqk.checkpoint import activations
from torch import nn
from ultralytics.nn.tasks import DetectionModel, PoseModel
from yolo_combine.fusion_model import assemble_graph_shared_model

from yolo_attention import stage3_checkpoint as stage3

from .common import REVIEW, review, sha256


def native_pose():
    data = review()
    payload = torch.load(data["pose_parent"], map_location="cpu", weights_only=False)
    model = (payload.get("ema") or payload["model"]).float().eval()
    if hasattr(model, "criterion"):
        del model.criterion
    if not isinstance(model, PoseModel) or model.model[-1].kpt_shape != [2, 3]:
        raise ValueError("expected trained BBAT ball/bat Pose26")
    return model


def view(model, task, *, clone=False):
    """淺層任務 view 不新增參數；驗證可選擇獨立 clone。"""
    graph = copy.copy(model.graph)
    graph._modules = model.graph._modules.copy()
    head = model.detect_head if task == "detect" else model.pose_head
    graph.model = nn.Sequential(*model.trunk_layers, head)
    if task == "pose":
        graph.__class__ = PoseModel
        graph.names = copy.deepcopy(model.pose_names)
        graph.yaml = copy.deepcopy(model.stage6_manifest["pose_architecture"])
        graph.kpt_shape = list(head.kpt_shape)
    elif task != "detect":
        raise ValueError(task)
    return copy.deepcopy(graph) if clone else graph


def assemble(detect, pose):
    # audit_task_pair 要求相同結構；Pose 原獨立 trunk 另外保留作參考。
    # 此處已明確選定共享 Detect trunk，不宣稱保留獨立 Pose 的輸出。
    for index in range(23):
        pose.model[index] = copy.deepcopy(detect.model[index])
    model, audit = assemble_graph_shared_model(detect, pose)
    if not audit.complete:
        raise ValueError("incomplete graph assembly")
    return model


def build():
    data = review()
    detect, upstream = stage3.load(data["detect_parent"])
    pose = native_pose()
    pose_architecture, pose_names = copy.deepcopy(pose.yaml), copy.deepcopy(pose.names)
    pose_activations = activations(pose)
    independent_parameters = sum(
        p.numel() for m in (detect, pose) for p in m.parameters()
    )
    if stage3.manifest_for(detect) != upstream:
        raise ValueError("D1 deployment contract changed")
    model = assemble(detect, pose)
    model.stage6_manifest = {
        "schema": "combine-stage6-rebuild-v2",
        "detect_parent": data["detect_parent"],
        "detect_parent_sha256": data["detect_parent_sha256"],
        "pose_parent": data["pose_parent"],
        "pose_parent_sha256": data["pose_parent_sha256"],
        "upstream": upstream,
        "pose_architecture": pose_architecture,
        "pose_names": pose_names,
        "pose_head_activations": {
            k: v for k, v in pose_activations.items() if k.startswith("model.23.")
        },
        "role": "pose-adaptation-on-detect-trunk",
        "rho": None,
        "mixing": data["protocol"]["mixing"],
        "graph": model.contract(),
        "independent_parameters": independent_parameters,
        "shared_parameters": sum(p.numel() for p in model.parameters()),
        "reference_review_sha256": sha256(REVIEW),
        "masf": None,
        "hog": None,
        "repconv": None,
        "dataset": data["protocol"]["data"],
        "evaluator": data["protocol"]["evaluator"],
    }
    manifest_for(model)
    return model.eval()


def mix(anchor, pose_branch, rho, *, parent):
    """只混合同血緣適應後的 trunk；原 Detect anchor 保持不動。"""
    if rho not in review()["protocol"]["rho"]:
        raise ValueError("unplanned trunk ratio")
    if anchor.stage6_manifest["upstream"] != pose_branch.stage6_manifest["upstream"]:
        raise ValueError("incompatible upstream contract")
    lineage = pose_branch.stage6_manifest.get("training_lineage", [])
    if not any(row["recipe"]["scope"] == "pose_full" for row in lineage):
        raise ValueError("mixing requires completed compatible full Pose adaptation")
    result = copy.deepcopy(anchor).cpu().eval()
    ds, ps = result.state_dict(), pose_branch.state_dict()
    mixed = []
    for name, value in ds.items():
        other = ps[name].detach().cpu()
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"incompatible tensor {name}")
        if ".pose_head." in name:
            ds[name] = other.clone()
        elif ".detect_head." in name:
            if not torch.equal(value, other):
                raise ValueError("Pose branch changed frozen Detect head")
        elif ".attn." in name:
            if not torch.equal(value, other):
                raise ValueError("Pose branch changed Attention")
        elif value.is_floating_point():
            ds[name] = value + rho * (other - value)
            mixed.append(name)
        elif not torch.equal(value, other):
            raise ValueError("Pose branch changed frozen shared integer buffer")
    result.load_state_dict(ds, strict=True)
    result.stage6_manifest.update(
        role="joint-candidate",
        rho=rho,
        mixed_tensor_names=mixed,
        pose_adaptation_parent=copy.deepcopy(parent),
        pose_adaptation_lineage=copy.deepcopy(lineage),
    )
    manifest_for(result)
    return result


def manifest_for(model):
    data = model.stage6_manifest
    if (
        data["schema"] != "combine-stage6-rebuild-v2"
        or model.contract() != data["graph"]
    ):
        raise ValueError("shared graph contract changed")
    if stage3.manifest_for(view(model, "detect")) != data["upstream"]:
        raise ValueError("inherited Attention/scale/activation fallback")
    pose_acts = activations(view(model, "pose"))
    if {k: v for k, v in pose_acts.items() if k.startswith("model.23.")} != data[
        "pose_head_activations"
    ]:
        raise ValueError("Pose activation changed")
    if any("masf" in n.lower() or "hog_aux" in n for n, _ in model.named_modules()):
        raise ValueError("unselected MASF/HOG module")
    return copy.deepcopy(data)


def from_manifest(data):
    from binaryqk.model import convert
    from binaryqk.scale_checkpoint import install as install_affine

    if data.get("schema") != "combine-stage6-rebuild-v2":
        raise ValueError("missing fusion manifest")
    upstream = data["upstream"]
    base = upstream["parent_manifest"]
    detect = DetectionModel(
        copy.deepcopy(base["architecture"]), ch=3, nc=80, verbose=False
    )
    detect.names = copy.deepcopy(base["names"])
    convert(detect, "B2", base["initial_coefficients"])
    install_affine(detect, "S1")
    stage3.install(detect, upstream["candidate"], base, upstream["lineage"])
    if "adaptation" in upstream:
        detect.stage3_adaptation = copy.deepcopy(upstream["adaptation"])
    pose = PoseModel(
        copy.deepcopy(data["pose_architecture"]),
        ch=3,
        nc=2,
        data_kpt_shape=(2, 3),
        verbose=False,
    )
    pose.names = copy.deepcopy(data["pose_names"])
    model = assemble(detect, pose)
    model.stage6_manifest = copy.deepcopy(data)
    return model.eval()


def role(name):
    if ".attn." in name or ".dfl." in name:
        return "fixed"
    if name.startswith("graph.model.23."):
        return "pose_head" if ".pose_head." in name else "detect_head"
    if not name.startswith("graph.model."):
        raise ValueError(name)
    return "neck" if int(name.split(".")[2]) >= 11 else "backbone"


def configure(model, recipe):
    scope = recipe["scope"]
    if scope not in ("head", "pose_full", "joint_neck", "joint_back9", "joint_full"):
        raise ValueError(scope)
    model.train()
    for name, parameter in model.named_parameters():
        r = role(name)
        enabled = r == "pose_head"
        if scope == "pose_full":
            enabled = r in ("backbone", "neck", "pose_head")
        elif scope.startswith("joint"):
            enabled = r in ("neck", "detect_head", "pose_head") or (
                r == "backbone"
                and (
                    scope == "joint_full"
                    or scope == "joint_back9"
                    and int(name.split(".")[2]) >= 9
                )
            )
        parameter.requires_grad_(enabled)
    mutable = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            is_head = ".pose_head." in name or ".detect_head." in name
            active = is_head and any(p.requires_grad for p in module.parameters())
            module.train(active)
            if active:
                mutable.update(f"{name}.{k}" for k, _ in module.named_buffers())
            elif not is_head:
                for p in module.parameters():
                    p.requires_grad_(False)
    return mutable
