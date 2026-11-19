"""自帶架構及完整階段契約的 FP32 checkpoint；缺項即失敗。"""

import copy
import json
from pathlib import Path

import torch
from ultralytics.nn.tasks import DetectionModel

from .basis import BASES, BranchScore
from .model import (
    PARENT,
    PARENT_SHA,
    SITES,
    BranchAttention,
    ProbabilityValue,
    convert,
    sha256,
)

CONTRACT = {
    "schema": "binaryqk-stage1-v2",
    "parent_sha256": PARENT_SHA,
    "sites": list(SITES), "heads": 4, "key_dim": 32, "head_dim": 64,
    "scale_format": "fixed-fp32-parent-calibration",
    "bias_format": "none", "exp_backend": "native-exp",
    "normalization": "native-softmax-once", "reciprocal_backend": "native-float-division",
    "pwl_range": None, "pwl_segments": None, "newton_iterations": 0,
    "probability_format": "fp32", "quantization": None,
    "token_policy": "dynamic-H-times-W", "sign_zero": 1,
    "backward": "signed-dot-surrogate-and-clipped-STE-abs-le-1",
    "activation_policy": "inherit-native-SiLU-stage7-pending",
    "repconv": {"sites": [], "state": "not-installed"},
    "data": "COCO2017 train118287-val5000; BBAT5 unchanged",
    "evaluator": "E2E-FP32-conf0.001-iou0.7-maxdet300-COCO80-internal-and-official-bbox",
}


def activations(model):
    return {name: type(module).__name__ for name, module in model.named_modules(remove_duplicate=False)
            if name.endswith(".act")}


def manifest_for(model, calibration_sha):
    modules = [model.get_submodule(site) for site in SITES]
    if not all(type(m) is BranchAttention for m in modules):
        raise ValueError("both sites must contain BranchAttention")
    if not all(type(m.score) is BranchScore and type(m.normalizer) is torch.nn.Softmax
               and m.normalizer.dim == -1 and type(m.aggregate) is ProbabilityValue for m in modules):
        raise ValueError("actual normalization/score/PV implementation violates stage1 contract")
    variants = {m.score.variant for m in modules}
    if len(variants) != 1:
        raise ValueError("mixed branch variants")
    variant = variants.pop()
    manifest = copy.deepcopy(CONTRACT)
    manifest.update(variant=variant, bases=list(BASES[variant]),
                    scale_slots=8 * len(BASES[variant]), calibration_sha256=calibration_sha,
                    coefficients={s: m.score.coefficients.tolist() for s, m in zip(SITES, modules)},
                    activation_calls=activations(model), architecture=copy.deepcopy(model.yaml),
                    names=copy.deepcopy(model.names),
                    lineage=copy.deepcopy(getattr(model, "research_lineage", {
                        "parent": str(PARENT), "parent_sha256": PARENT_SHA, "phase": "conversion"})))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    for key, value in CONTRACT.items():
        if key not in manifest or manifest[key] != value:
            raise ValueError(f"missing or unsupported contract: {key}")
    variant = manifest.get("variant")
    if variant not in BASES or manifest.get("bases") != list(BASES[variant]):
        raise ValueError("invalid variant/bases")
    if manifest.get("scale_slots") != 8 * len(BASES[variant]):
        raise ValueError("invalid scale slots")
    if set(manifest.get("coefficients", {})) != set(SITES):
        raise ValueError("missing coefficients")
    for field in ("activation_calls", "architecture", "names", "calibration_sha256"):
        if not manifest.get(field):
            raise ValueError(f"missing {field}")
    digest = manifest["calibration_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid calibration SHA")
    lineage = manifest.get("lineage", {})
    if not lineage.get("parent") or lineage.get("phase") not in {"conversion", "screening", "recovery"}:
        raise ValueError("missing or invalid immediate parent lineage")
    digest = lineage.get("parent_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid immediate parent SHA")


def save(path, model, manifest):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace checkpoint: {path}")
    validate_manifest(manifest)
    if manifest_for(model, manifest["calibration_sha256"]) != manifest:
        raise ValueError("model does not match manifest")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in state.values()):
        raise ValueError("stage1 checkpoint requires finite FP32 tensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "manifest": manifest}, path)
    path.with_suffix(".json").write_text(json.dumps({"checkpoint_sha256": sha256(path)}, indent=2) + "\n")


def load(path):
    path = Path(path)
    expected = json.loads(path.with_suffix(".json").read_text())["checkpoint_sha256"]
    if sha256(path) != expected:
        raise ValueError("checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = payload["manifest"]
    validate_manifest(manifest)
    model = DetectionModel(copy.deepcopy(manifest["architecture"]), ch=3, nc=80, verbose=False)
    convert(model, manifest["variant"], manifest["coefficients"])
    model.names = manifest["names"]
    model.research_lineage = copy.deepcopy(manifest["lineage"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if manifest_for(model, manifest["calibration_sha256"]) != manifest:
        raise ValueError("reloaded model violates activation/scale/architecture contract")
    return model.eval(), manifest
