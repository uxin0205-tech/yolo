"""自包含融合推論 checkpoint；resume 另存。"""

import json
from pathlib import Path

import torch
from binaryqk.train import atomic_torch, state_cpu

from .common import read, sha256, write_json
from .models import from_manifest, manifest_for


def save(path, model):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("checkpoint already exists")
    manifest = manifest_for(model)
    state = state_cpu(model)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in state.values()
    ):
        raise ValueError("checkpoint requires finite FP32")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch(path, {"manifest": manifest, "state_dict": state})
    write_json(
        path.with_suffix(".json"), {"sha256": sha256(path), "manifest": manifest}
    )


def load(path):
    path = Path(path)
    meta = read(path.with_suffix(".json"))
    if sha256(path) != meta["sha256"]:
        raise ValueError("checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = payload["manifest"]
    if json.loads(json.dumps(manifest)) != meta["manifest"]:
        raise ValueError("checkpoint manifest mismatch")
    model = from_manifest(manifest)
    model.load_state_dict(payload["state_dict"], strict=True)
    if manifest_for(model) != manifest:
        raise ValueError("checkpoint contract mismatch after reload")
    return model.eval()
