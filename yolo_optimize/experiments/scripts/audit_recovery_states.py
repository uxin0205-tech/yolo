#!/usr/bin/env python3
"""CPU-only differential audit for the parent and native recovery states.

The audit intentionally reads inference checkpoints as state dictionaries only.
It does not import the model implementation, construct a device, run a forward
pass, or alter either source checkpoint.  Exact float differences are reported
separately from a material absolute threshold because a live-frozen EMA can
accumulate tiny float rounding drift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

# Keep this process CPU-only before importing torch.  The audit never touches
# torch.cuda and every checkpoint load explicitly maps tensors to CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = Path(
    "/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt"
)
DEFAULT_CHILD = WORKSPACE / "artifacts/direction1-20260908/native-control/inference/epoch-0001.pt"
DEFAULT_OUTPUT = WORKSPACE / "artifacts/direction1-20260908/state-audit.json"

MODEL_LAYER = re.compile(r"^graph\.model\.(\d+)(?:\.|$)")
HEAD_NAMES = ("detect", "pose")
BN_STATS = ("running_mean", "running_var", "num_batches_tracked")
MATERIAL_ABS_TOL = 1.0e-5
EPS = 1.0e-12


def _layer_index(name: str) -> int | None:
    match = MODEL_LAYER.match(name)
    return None if match is None else int(match.group(1))


def _is_detect_head(name: str) -> bool:
    return ".detect_head." in name


def _is_pose_head(name: str) -> bool:
    return ".pose_head." in name


def _head_name(name: str) -> str | None:
    if _is_detect_head(name):
        return "detect"
    if _is_pose_head(name):
        return "pose"
    return None


def _head_branch(name: str) -> str:
    return "one_to_one" if ".one2one_" in name else "one_to_many"


def _is_bn_stat(name: str, suffix: str | None = None) -> bool:
    if ".bn." not in name:
        return False
    return any(name.endswith(f".bn.{item}") for item in (BN_STATS if suffix is None else (suffix,)))


def _is_qk(name: str) -> bool:
    lowered = name.lower()
    return (
        ".attn.qkv.q." in lowered
        or ".attn.qkv.k." in lowered
        or ".binaryqk." in lowered
        or ".qk." in lowered
    )


def _is_attention(name: str) -> bool:
    lowered = name.lower()
    return ".attn." in lowered or ".attention." in lowered


def _is_masf(name: str) -> bool:
    return "masf" in name.lower()


def _is_backbone(name: str) -> bool:
    layer = _layer_index(name)
    return layer is not None and layer <= 10


def _is_neck(name: str) -> bool:
    layer = _layer_index(name)
    return layer is not None and 11 <= layer <= 22


def _is_weight(name: str) -> bool:
    return name.endswith(".weight")


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state entry is not a tensor: {name} ({type(value).__name__})")
    if value.device.type != "cpu":
        raise RuntimeError(f"non-CPU tensor encountered: {name} ({value.device})")
    # float64 makes the aggregate norms deterministic without constructing a
    # model or moving any state to an accelerator.
    return value.detach().to(dtype=torch.float64)


def _load(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    # Keep this call explicit: mmap and weights_only are part of the audit
    # contract, and map_location prevents device restoration from the file.
    payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload is not a dict: {path}")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"checkpoint has no state_dict: {path}")
    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError(f"state key is not a string: {name!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry is not a tensor: {name}")
        if value.device.type != "cpu":
            raise RuntimeError(f"state entry is not on CPU: {name} ({value.device})")
    return payload, state


def _per_tensor_stats(
    name: str,
    parent: torch.Tensor,
    child: torch.Tensor,
) -> dict[str, Any]:
    left = _tensor(parent, name)
    right = _tensor(child, name)
    if left.shape != right.shape:
        raise ValueError(f"shape changed for {name}: {tuple(left.shape)} != {tuple(right.shape)}")
    delta = right - left
    abs_delta = delta.abs()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    exact = int((delta != 0).sum().item())
    material = int((abs_delta > MATERIAL_ABS_TOL).sum().item())
    delta_sq = float((delta * delta).sum().item())
    parent_sq = float((left * left).sum().item())
    maxabs = float(abs_delta.max().item()) if delta.numel() else 0.0
    return {
        "name": name,
        "shape": list(left.shape),
        "dtype": str(parent.dtype),
        "elements": int(delta.numel()),
        "exact_changed_elements": exact,
        "material_changed_elements": material,
        "maxabs": maxabs,
        "rel_l2": math.sqrt(delta_sq) / (math.sqrt(parent_sq) + EPS),
        "delta_l2_sq": delta_sq,
        "parent_l2_sq": parent_sq,
        "finite": finite,
    }


def _aggregate(
    names: list[str], stats: Mapping[str, dict[str, Any]], *, top_n: int = 10
) -> dict[str, Any]:
    selected = [stats[name] for name in names]
    delta_sq = sum(float(item["delta_l2_sq"]) for item in selected)
    parent_sq = sum(float(item["parent_l2_sq"]) for item in selected)
    exact_elements = sum(int(item["exact_changed_elements"]) for item in selected)
    material_elements = sum(int(item["material_changed_elements"]) for item in selected)
    return {
        "tensor_count": len(selected),
        "element_count": sum(int(item["elements"]) for item in selected),
        "exact_changed_tensor_count": sum(
            int(item["exact_changed_elements"] > 0) for item in selected
        ),
        "material_changed_tensor_count": sum(
            int(item["material_changed_elements"] > 0) for item in selected
        ),
        "exact_changed_element_count": exact_elements,
        "material_changed_element_count": material_elements,
        "maxabs": max((float(item["maxabs"]) for item in selected), default=0.0),
        "rel_l2": math.sqrt(delta_sq) / (math.sqrt(parent_sq) + EPS),
        "nonfinite_tensor_count": sum(int(not item["finite"]) for item in selected),
        "material_drift": material_elements > 0,
        "top_changes": [
            {
                key: item[key]
                for key in (
                    "name",
                    "shape",
                    "dtype",
                    "exact_changed_elements",
                    "material_changed_elements",
                    "maxabs",
                    "rel_l2",
                )
            }
            for item in sorted(
                selected,
                key=lambda item: (
                    int(item["material_changed_elements"] > 0),
                    int(item["exact_changed_elements"] > 0),
                    float(item["maxabs"]),
                    float(item["rel_l2"]),
                ),
                reverse=True,
            )[:top_n]
        ],
    }


def _select(names: list[str], predicate: Callable[[str], bool]) -> list[str]:
    return [name for name in names if predicate(name)]


def _metadata_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("stage", "epoch", "global_macro_step", "training_auxiliary_removed", "full_resume_sha256"):
        if key in metadata:
            result[key] = metadata[key]
    metrics = metadata.get("metrics")
    if isinstance(metrics, Mapping):
        result["metrics"] = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    return result


def _metric_delta(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, float]:
    left = parent.get("metrics", {})
    right = child.get("metrics", {})
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {}
    return {
        str(key): float(right[key]) - float(left[key])
        for key in sorted(set(left) & set(right))
        if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float))
    }


def _head_bn_report(
    names: list[str], stats: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in HEAD_NAMES:
        head_result: dict[str, Any] = {}
        for branch in ("all", "one_to_many", "one_to_one"):
            branch_result: dict[str, Any] = {}
            for suffix in BN_STATS:
                selected = _select(
                    names,
                    lambda name, head=head, branch=branch, suffix=suffix: (
                        _head_name(name) == head
                        and (branch == "all" or _head_branch(name) == branch)
                        and name.endswith(f".bn.{suffix}")
                    ),
                )
                branch_result[suffix] = _aggregate(selected, stats)
            head_result[branch] = branch_result
        result[head] = head_result
    return result


def audit(parent_path: Path, child_path: Path) -> dict[str, Any]:
    parent_payload, parent_state = _load(parent_path)
    child_payload, child_state = _load(child_path)
    parent_names = set(parent_state)
    child_names = set(child_state)
    missing = sorted(parent_names - child_names)
    unexpected = sorted(child_names - parent_names)
    if missing or unexpected:
        raise ValueError(f"state key set changed: missing={missing[:10]}, unexpected={unexpected[:10]}")

    names = sorted(parent_names)
    stats = {
        name: _per_tensor_stats(name, parent_state[name], child_state[name])
        for name in names
    }

    frozen_selectors: dict[str, Callable[[str], bool]] = {
        # Shared BN running statistics are frozen by the stage policy even
        # though neck BN affine parameters remain trainable.
        "shared_bn_running_stats": lambda name: (
            ".detect_head." not in name
            and ".pose_head." not in name
            and _is_bn_stat(name)
        ),
        "attention": _is_attention,
        "qk": _is_qk,
        "masf": _is_masf,
        "backbone": _is_backbone,
    }
    frozen_scope: dict[str, Any] = {}
    for category, selector in frozen_selectors.items():
        selected = _select(names, selector)
        summary = _aggregate(selected, stats)
        summary["expected_policy"] = "frozen"
        summary["verdict"] = (
            "material_drift_detected" if summary["material_drift"] else "no_material_drift"
        )
        frozen_scope[category] = summary

    weight_selectors: dict[str, Callable[[str], bool]] = {
        "neck_all_weights": lambda name: _is_neck(name) and _is_weight(name),
        "neck_without_masf_weights": lambda name: (
            _is_neck(name) and not _is_masf(name) and _is_weight(name)
        ),
        "masf_weights": lambda name: _is_masf(name) and _is_weight(name),
        "detect_head_weights": lambda name: _is_detect_head(name) and _is_weight(name),
        "pose_head_weights": lambda name: _is_pose_head(name) and _is_weight(name),
    }
    weight_changes: dict[str, Any] = {}
    for category, selector in weight_selectors.items():
        selected = _select(names, selector)
        summary = _aggregate(selected, stats)
        summary["expected_policy"] = (
            "trainable" if category in {"neck_without_masf_weights", "detect_head_weights", "pose_head_weights"}
            else "mixed_trainable_and_frozen"
            if category == "neck_all_weights"
            else "frozen"
        )
        weight_changes[category] = summary

    shared_bn_affine = {
        "all": _aggregate(
            _select(
                names,
                lambda name: (
                    ".bn." in name
                    and ".detect_head." not in name
                    and ".pose_head." not in name
                    and (name.endswith(".bn.weight") or name.endswith(".bn.bias"))
                ),
            ),
            stats,
        ),
        "backbone_or_attention_or_masf": _aggregate(
            _select(
                names,
                lambda name: (
                    ".bn." in name
                    and ".detect_head." not in name
                    and ".pose_head." not in name
                    and (name.endswith(".bn.weight") or name.endswith(".bn.bias"))
                    and (_is_backbone(name) or _is_attention(name) or _is_masf(name))
                ),
            ),
            stats,
        ),
        "neck": _aggregate(
            _select(
                names,
                lambda name: (
                    _is_neck(name)
                    and ".bn." in name
                    and (name.endswith(".bn.weight") or name.endswith(".bn.bias"))
                ),
            ),
            stats,
        ),
    }

    parent_meta = _metadata_summary(parent_payload)
    child_meta = _metadata_summary(child_payload)
    return {
        "schema_version": 1,
        "audit": {
            "device": "cpu",
            "torch_threads": 2,
            "load_contract": "torch.load(weights_only=True, map_location='cpu', mmap=True)",
            "material_absolute_tolerance": MATERIAL_ABS_TOL,
            "exact_difference_note": "exact float differences are reported separately; exact_changed_count alone is not a freeze failure",
            "scope_overlap_note": "attention and qk are subsets of the backbone index scope when their layer is <=10; categories are intentionally reported independently",
        },
        "inputs": {
            "parent": str(parent_path),
            "child": str(child_path),
            "parent_checkpoint_kind": parent_payload.get("checkpoint_kind"),
            "child_checkpoint_kind": child_payload.get("checkpoint_kind"),
            "state_tensor_count": len(names),
            "state_key_set_equal": True,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
        "checkpoint_metadata": {
            "parent": parent_meta,
            "child": child_meta,
            "metric_delta_child_minus_parent": _metric_delta(parent_meta, child_meta),
        },
        "head_bn": {
            "expected_policy": "head BN remains in native train behavior; large running-stat drift is therefore not by itself a freeze violation",
            "by_head_and_branch": _head_bn_report(names, stats),
        },
        "frozen_scope": frozen_scope,
        "shared_bn_affine": {
            "expected_policy": "neck affine is trainable; frozen backbone/attention/MASF affine is separated below",
            **shared_bn_affine,
        },
        "neck_head_weight_changes": weight_changes,
        "overall": _aggregate(names, stats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--child", type=Path, default=DEFAULT_CHILD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-material-frozen-drift",
        action="store_true",
        help="return non-zero if any expected-frozen category exceeds the material threshold",
    )
    args = parser.parse_args()
    parent = args.parent.expanduser().resolve()
    child = args.child.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not parent.is_file():
        raise FileNotFoundError(parent)
    if not child.is_file():
        raise FileNotFoundError(child)
    if not output.is_relative_to(WORKSPACE):
        raise ValueError("audit output must remain inside yolo_optimize")

    torch.set_num_threads(2)
    report = audit(parent, child)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frozen = report["frozen_scope"]
    material_categories = [name for name, item in frozen.items() if item["material_drift"]]
    summary = {
        "status": "material_frozen_drift" if material_categories else "pass",
        "output": str(output),
        "state_tensors": report["inputs"]["state_tensor_count"],
        "material_frozen_categories": material_categories,
        "head_detect_bn": report["head_bn"]["by_head_and_branch"]["detect"]["all"],
        "head_pose_bn": report["head_bn"]["by_head_and_branch"]["pose"]["all"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.fail_on_material_frozen_drift and material_categories:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
