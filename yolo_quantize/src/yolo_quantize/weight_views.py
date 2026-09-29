"""Non-mutating Full35 master and BN-folded deployment weight views."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .weight_quantization import Full35WeightRegionCatalog

_INFERENCE_CONTRACT_KEYS = (
    "model_kind",
    "shared_layers",
    "head_inputs",
    "feature_channels",
    "strides",
    "reg_max",
    "end2end",
    "detect_nc",
    "pose_nc",
    "kpt_shape",
    "detect_names",
    "pose_names",
)


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _batch_norm_count(model: nn.Module) -> int:
    return sum(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )


def _inference_contract(model: nn.Module) -> dict[str, object]:
    contract = getattr(model, "contract", None)
    if not callable(contract):
        raise TypeError("Full35 weight view source exposes no contract()")
    payload = contract()
    if not isinstance(payload, dict):
        raise TypeError("Full35 contract() must return a mapping")
    missing = tuple(key for key in _INFERENCE_CONTRACT_KEYS if key not in payload)
    if missing:
        raise ValueError("Full35 inference contract is missing: " + ", ".join(missing))
    return {key: payload[key] for key in _INFERENCE_CONTRACT_KEYS}


def _fused_inference_contract(model: nn.Module) -> dict[str, object]:
    try:
        prediction = model.prediction
        detect_head = model.detect_head
        pose_head = model.pose_head
        branches = detect_head.one2one_cv2
    except AttributeError as error:
        raise TypeError(
            "fused Full35 model lacks one-to-one inference state"
        ) from error
    feature_channels: list[int] = []
    for branch in branches:
        first_conv = next(
            (module for module in branch.modules() if isinstance(module, nn.Conv2d)),
            None,
        )
        if first_conv is None:
            raise TypeError("fused Full35 one-to-one branch contains no Conv2d")
        feature_channels.append(first_conv.in_channels)
    return {
        "model_kind": model.model_kind,
        "shared_layers": len(model.trunk_layers),
        "head_inputs": list(prediction.f),
        "feature_channels": feature_channels,
        "strides": [float(value) for value in detect_head.stride.tolist()],
        "reg_max": int(detect_head.reg_max),
        "end2end": bool(detect_head.end2end),
        "detect_nc": int(detect_head.nc),
        "pose_nc": int(pose_head.nc),
        "kpt_shape": list(pose_head.kpt_shape),
        "detect_names": dict(model.detect_names),
        "pose_names": dict(model.pose_names),
    }


def _all_floating_state_is_fp32(model: nn.Module) -> bool:
    return all(
        not tensor.dtype.is_floating_point or tensor.dtype == torch.float32
        for tensor in model.state_dict().values()
    )


def _flatten_tensors(value: Any, prefix: str = "output") -> dict[str, torch.Tensor]:
    flattened: dict[str, torch.Tensor] = {}

    def visit(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            flattened[path] = item
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, prefix)
    return flattened


def _forward_parity(master: nn.Module, deployment: nn.Module) -> dict[str, object]:
    master_probe = copy.deepcopy(master).eval()
    deployment_probe = copy.deepcopy(deployment).eval()
    image = torch.linspace(0.0, 1.0, 3 * 64 * 64, dtype=torch.float32).reshape(
        1, 3, 64, 64
    )
    with torch.inference_mode():
        complete_reference = _flatten_tensors(master_probe(image, task="both"))
        candidate = _flatten_tensors(deployment_probe(image, task="both"))
    removed_paths = tuple(path for path in complete_reference if path not in candidate)
    reference = {
        path: tensor for path, tensor in complete_reference.items() if path in candidate
    }
    same_structure = (
        tuple(reference) == tuple(candidate)
        and all(".one2many." in path for path in removed_paths)
        and all(reference[path].shape == candidate[path].shape for path in reference)
    )
    if not same_structure:
        return {
            "same_structure": False,
            "all_finite": False,
            "normalized_rmse": math.inf,
            "maximum_absolute_error": math.inf,
            "maximum_relative_to_reference_peak": math.inf,
            "removed_training_only_tensor_paths": removed_paths,
            "passed": False,
        }
    squared_error = 0.0
    reference_energy = 0.0
    maximum_error = 0.0
    reference_peak = 0.0
    all_finite = True
    for path, reference_tensor in reference.items():
        reference_float = reference_tensor.float()
        candidate_float = candidate[path].float()
        error = candidate_float - reference_float
        squared_error += float(error.square().sum().item())
        reference_energy += float(reference_float.square().sum().item())
        maximum_error = max(
            maximum_error,
            float(error.abs().max().item()) if error.numel() else 0.0,
        )
        reference_peak = max(
            reference_peak,
            float(reference_float.abs().max().item())
            if reference_float.numel()
            else 0.0,
        )
        all_finite = all_finite and bool(torch.isfinite(candidate_float).all())
    normalized_rmse = math.sqrt(squared_error / max(reference_energy, 1e-24))
    relative_peak_error = maximum_error / max(reference_peak, 1e-24)
    return {
        "same_structure": True,
        "all_finite": all_finite,
        "normalized_rmse": normalized_rmse,
        "maximum_absolute_error": maximum_error,
        "maximum_relative_to_reference_peak": relative_peak_error,
        "removed_training_only_tensor_paths": removed_paths,
        "passed": (
            all_finite and normalized_rmse <= 1e-5 and relative_peak_error <= 1e-4
        ),
    }


@dataclass(frozen=True)
class WeightViewParityManifest:
    """Machine-readable structural parity evidence for a dual weight view."""

    schema_version: int
    source_state_sha256: str
    master_state_sha256: str
    deployment_state_sha256: str
    source_unchanged: bool
    master_matches_source: bool
    source_batch_norm_modules: int
    master_batch_norm_modules: int
    deployment_batch_norm_modules: int
    folded_batch_norm_modules: int
    deployment_path_parity: bool
    inference_contract_parity: bool
    forward_same_structure: bool
    forward_all_finite: bool
    forward_normalized_rmse: float
    forward_maximum_absolute_error: float
    forward_maximum_relative_to_reference_peak: float
    forward_removed_training_only_tensor_paths: tuple[str, ...]
    forward_parity_passed: bool
    master_catalog: dict[str, object]
    deployment_catalog: dict[str, object]
    forward_parity_evaluated: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_state_sha256": self.source_state_sha256,
            "master_state_sha256": self.master_state_sha256,
            "deployment_state_sha256": self.deployment_state_sha256,
            "source_unchanged": self.source_unchanged,
            "master_matches_source": self.master_matches_source,
            "source_batch_norm_modules": self.source_batch_norm_modules,
            "master_batch_norm_modules": self.master_batch_norm_modules,
            "deployment_batch_norm_modules": self.deployment_batch_norm_modules,
            "folded_batch_norm_modules": self.folded_batch_norm_modules,
            "deployment_path_parity": self.deployment_path_parity,
            "inference_contract_parity": self.inference_contract_parity,
            "forward_same_structure": self.forward_same_structure,
            "forward_all_finite": self.forward_all_finite,
            "forward_normalized_rmse": self.forward_normalized_rmse,
            "forward_maximum_absolute_error": self.forward_maximum_absolute_error,
            "forward_maximum_relative_to_reference_peak": (
                self.forward_maximum_relative_to_reference_peak
            ),
            "forward_removed_training_only_tensor_paths": list(
                self.forward_removed_training_only_tensor_paths
            ),
            "forward_parity_passed": self.forward_parity_passed,
            "master_catalog": self.master_catalog,
            "deployment_catalog": self.deployment_catalog,
            "forward_parity_evaluated": self.forward_parity_evaluated,
        }


@dataclass(frozen=True)
class Full35WeightViews:
    """Independent FP32 master and deployment models plus parity evidence."""

    master: nn.Module
    deployment: nn.Module
    manifest: WeightViewParityManifest


class Full35WeightViewAdapter:
    """Build independent Full35 weight views without mutating the caller model."""

    def build(self, model: nn.Module) -> Full35WeightViews:
        source_before = _state_sha256(model)
        source_contract = _inference_contract(model)
        source_bn = _batch_norm_count(model)

        master = copy.deepcopy(model).cpu().float().eval()
        master_digest = _state_sha256(master)
        master_catalog = Full35WeightRegionCatalog.inspect(master)

        deployment = copy.deepcopy(master)
        graph = getattr(deployment, "graph", None)
        fuse = getattr(graph, "fuse", None)
        if not callable(fuse):
            raise TypeError("Full35 weight view source graph exposes no fuse()")
        fuse(verbose=False)
        deployment.cpu().float().eval()

        source_after = _state_sha256(model)
        deployment_catalog = Full35WeightRegionCatalog.inspect(deployment)
        deployment_contract = _fused_inference_contract(deployment)
        master_bn = _batch_norm_count(master)
        deployment_bn = _batch_norm_count(deployment)
        master_paths = tuple(site.path for site in master_catalog.deployment_sites)
        deployment_paths = tuple(
            site.path for site in deployment_catalog.deployment_sites
        )
        source_unchanged = source_after == source_before
        master_matches_source = master_digest == source_before
        path_parity = deployment_paths == master_paths
        contract_parity = deployment_contract == source_contract

        if not source_unchanged:
            raise RuntimeError("Full35 weight view construction mutated the source")
        if not master_matches_source or not _all_floating_state_is_fp32(master):
            raise RuntimeError("Full35 unfused master is not an exact FP32 copy")
        if not _all_floating_state_is_fp32(deployment):
            raise RuntimeError("Full35 deployment view is not FP32")
        if deployment_bn:
            raise RuntimeError("Full35 deployment view retains BatchNorm modules")
        if not path_parity:
            raise RuntimeError("Full35 deployment weight paths changed during fuse")
        if not contract_parity:
            raise RuntimeError("Full35 inference contract changed during fuse")
        if deployment_catalog.training_only_sites:
            raise RuntimeError("Full35 deployment view retains training-only weights")
        forward_parity = _forward_parity(master, deployment)
        if not forward_parity["passed"]:
            raise RuntimeError(
                "Full35 BN-folded deployment view failed deterministic forward parity"
            )

        return Full35WeightViews(
            master=master,
            deployment=deployment,
            manifest=WeightViewParityManifest(
                schema_version=1,
                source_state_sha256=source_before,
                master_state_sha256=master_digest,
                deployment_state_sha256=_state_sha256(deployment),
                source_unchanged=source_unchanged,
                master_matches_source=master_matches_source,
                source_batch_norm_modules=source_bn,
                master_batch_norm_modules=master_bn,
                deployment_batch_norm_modules=deployment_bn,
                folded_batch_norm_modules=master_bn - deployment_bn,
                deployment_path_parity=path_parity,
                inference_contract_parity=contract_parity,
                forward_same_structure=bool(forward_parity["same_structure"]),
                forward_all_finite=bool(forward_parity["all_finite"]),
                forward_normalized_rmse=float(forward_parity["normalized_rmse"]),
                forward_maximum_absolute_error=float(
                    forward_parity["maximum_absolute_error"]
                ),
                forward_maximum_relative_to_reference_peak=float(
                    forward_parity["maximum_relative_to_reference_peak"]
                ),
                forward_removed_training_only_tensor_paths=tuple(
                    forward_parity["removed_training_only_tensor_paths"]
                ),
                forward_parity_passed=bool(forward_parity["passed"]),
                master_catalog=master_catalog.summary(),
                deployment_catalog=deployment_catalog.summary(),
            ),
        )
