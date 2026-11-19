"""帶完整上游定點契約的 HOG training／deployment checkpoint。"""

import copy
import json
from pathlib import Path

import torch
from binaryqk.train import atomic_torch, state_cpu
from ultralytics.nn.tasks import DetectionModel

from yolo_attention import stage3_checkpoint as stage3

from .common import REVIEW, read, review, sha256, write_json
from .models import HOG_SPEC, attach


def build(calibration_path=None):
    data = review()
    model, upstream = stage3.load(data["parent"])
    attach(model)
    model.stage5_manifest = {
        "schema": "hog-stage5-v1",
        "parent": data["parent"],
        "parent_sha256": data["parent_sha256"],
        "parent_handoff": data["parent_handoff"],
        "parent_handoff_sha256": data["parent_handoff_sha256"],
        "upstream": upstream,
        "hog": copy.deepcopy(HOG_SPEC),
        "auxiliary_present": True,
        "reference_review_sha256": sha256(REVIEW),
        "calibration_sha256": sha256(calibration_path) if calibration_path else None,
        "data": "COCO2017 full train118287/val5000",
        "evaluator": "official COCO FP32 E2E",
        "new_inference_parameters": 0,
        "masf": None,
        "repconv": None,
    }
    return model.eval()


def manifest_for(model, calibration_sha=None):
    data = model.stage5_manifest
    if data.get("schema") != "hog-stage5-v1" or data.get("hog") != HOG_SPEC:
        raise ValueError("HOG manifest missing or changed")
    if data["auxiliary_present"] != hasattr(model, "hog_aux"):
        raise ValueError("HOG training/deployment graph mismatch")
    if hasattr(model, "hog_aux"):
        aux = model.hog_aux
        if (
            aux.projection.in_channels,
            aux.projection.out_channels,
            aux.cell_size,
            aux.bins,
            aux.energy_epsilon,
        ) != (256, 9, 8, 9, 1e-6):
            raise ValueError("HOG numerical contract changed")
    if stage3.manifest_for(model) != data["upstream"]:
        raise ValueError("inherited Attention/activation contract changed")
    return copy.deepcopy(data)


def save(path, model, manifest):
    path = Path(path)
    if (
        path.exists()
        or path.with_suffix(".json").exists()
        or manifest_for(model) != manifest
    ):
        raise ValueError("checkpoint exists or manifest changed")
    state = state_cpu(model)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in state.values()
    ):
        raise ValueError("HOG checkpoints require finite FP32 tensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch(path, {"manifest": manifest, "state_dict": state})
    write_json(
        path.with_suffix(".json"), {"sha256": sha256(path), "manifest": manifest}
    )


def load(path):
    from binaryqk.model import convert
    from binaryqk.scale_checkpoint import install as install_affine

    path = Path(path)
    meta = read(path.with_suffix(".json"))
    if sha256(path) != meta["sha256"]:
        raise ValueError("HOG checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    data = payload["manifest"]
    if (
        data.get("schema") != "hog-stage5-v1"
        or json.loads(json.dumps(data)) != meta["manifest"]
    ):
        raise ValueError("HOG manifest mismatch")
    upstream = data["upstream"]
    base = upstream["parent_manifest"]
    model = DetectionModel(
        copy.deepcopy(base["architecture"]), ch=3, nc=80, verbose=False
    )
    model.names = copy.deepcopy(base["names"])
    convert(model, "B2", base["initial_coefficients"])
    install_affine(model, "S1")
    stage3.install(model, upstream["candidate"], base, upstream["lineage"])
    if "adaptation" in upstream:
        model.stage3_adaptation = copy.deepcopy(upstream["adaptation"])
    if data["auxiliary_present"]:
        attach(model)
    model.stage5_manifest = copy.deepcopy(data)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in payload["state_dict"].values()
    ):
        raise ValueError("invalid HOG checkpoint tensors")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if manifest_for(model) != data:
        raise ValueError("reloaded HOG contract mismatch")
    return model, data
