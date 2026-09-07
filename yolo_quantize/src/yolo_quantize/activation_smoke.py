"""No-training GPU smoke matrix for Full35 activation-output quantization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .experiment_plan import CoupledExperimentPlanner
from .full35_adapter import Full35ActivationAdapter, Full35ActivationPolicy
from .intake import ActivationIntake

WORKSPACE = Path("/home/uxin/yolo")
COCO_ROOT = WORKSPACE / "coco2017"
BBAT5_POSE_ROOT = WORKSPACE / "original/pose/derived/bbat5-v1/pose"
ACTIVATION_ROOT = WORKSPACE / "yolo_activation"
FULL35_ROOT = WORKSPACE / "yolo_combine/final/full35"
DEFAULT_OUTPUT = Path("artifacts/reports/activation-smoke-v2.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _coco_paths(split: str, count: int) -> tuple[Path, ...]:
    list_path = COCO_ROOT / f"{split}2017.txt"
    selected: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        relative = raw.removeprefix("./")
        candidate = COCO_ROOT / relative
        if candidate.is_file():
            selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"COCO {split} provides only {len(selected)} exemplars")
    return tuple(selected)


def _bbat5_paths(split: str, count: int) -> tuple[Path, ...]:
    directory = BBAT5_POSE_ROOT / "images" / split
    selected: list[Path] = []
    source_groups: set[str] = set()
    for candidate in sorted(directory.iterdir()):
        if not candidate.is_file():
            continue
        group = candidate.name.split(".rf.", maxsplit=1)[0]
        if group in source_groups:
            continue
        source_groups.add(group)
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"BBAT5 {split} provides only {len(selected)} source groups")
    return tuple(selected)


def _letterbox(path: Path, image_size: int) -> Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    ratio = min(image_size / width, image_size / height)
    resized = image.resize(
        (round(width * ratio), round(height * ratio)),
        Image.Resampling.BILINEAR,
    )
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    left = (image_size - resized.width) // 2
    top = (image_size - resized.height) // 2
    canvas[top : top + resized.height, left : left + resized.width] = np.asarray(
        resized
    )
    return (
        torch.from_numpy(canvas)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div_(255.0)
        .unsqueeze(0)
    )


def _flatten_tensors(value: Any, prefix: str = "output") -> dict[str, Tensor]:
    flattened: dict[str, Tensor] = {}

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


def _compare_outputs(reference: Any, candidate: Any) -> dict[str, Any]:
    before = _flatten_tensors(reference)
    after = _flatten_tensors(candidate)
    if tuple(before) != tuple(after):
        raise RuntimeError(
            f"output tensor paths changed: {tuple(before)} != {tuple(after)}"
        )
    total_elements = 0
    squared_error = 0.0
    reference_energy = 0.0
    dot_product = 0.0
    candidate_energy = 0.0
    absolute_error = 0.0
    maximum_error = 0.0
    details: list[dict[str, Any]] = []
    all_finite = True
    for path, reference_tensor in before.items():
        candidate_tensor = after[path]
        if reference_tensor.shape != candidate_tensor.shape:
            raise RuntimeError(
                f"output shape changed at {path}: "
                f"{reference_tensor.shape} != {candidate_tensor.shape}"
            )
        reference_float = reference_tensor.float()
        candidate_float = candidate_tensor.float()
        error = candidate_float - reference_float
        elements = reference_float.numel()
        tensor_error = float(error.square().sum().item())
        tensor_reference = float(reference_float.square().sum().item())
        tensor_candidate = float(candidate_float.square().sum().item())
        tensor_dot = float((reference_float * candidate_float).sum().item())
        tensor_absolute = float(error.abs().sum().item())
        tensor_maximum = float(error.abs().max().item()) if elements else 0.0
        tensor_nrmse = math.sqrt(tensor_error / max(tensor_reference, 1e-24))
        tensor_cosine = tensor_dot / math.sqrt(
            max(tensor_reference * tensor_candidate, 1e-24)
        )
        finite = bool(torch.isfinite(candidate_float).all())
        all_finite = all_finite and finite
        total_elements += elements
        squared_error += tensor_error
        reference_energy += tensor_reference
        dot_product += tensor_dot
        candidate_energy += tensor_candidate
        absolute_error += tensor_absolute
        maximum_error = max(maximum_error, tensor_maximum)
        details.append(
            {
                "path": path,
                "shape": list(reference_tensor.shape),
                "elements": elements,
                "finite": finite,
                "normalized_rmse": tensor_nrmse,
                "cosine": tensor_cosine,
                "mean_absolute_error": tensor_absolute / max(elements, 1),
                "maximum_absolute_error": tensor_maximum,
            }
        )
    return {
        "tensor_count": len(before),
        "elements": total_elements,
        "same_structure": True,
        "all_finite": all_finite,
        "normalized_rmse": math.sqrt(squared_error / max(reference_energy, 1e-24)),
        "cosine": dot_product
        / math.sqrt(max(reference_energy * candidate_energy, 1e-24)),
        "mean_absolute_error": absolute_error / max(total_elements, 1),
        "maximum_absolute_error": maximum_error,
        "tensors": details,
    }


def _topk_overlap(reference_scores: Tensor, candidate_scores: Tensor) -> dict[str, Any]:
    if reference_scores.shape != candidate_scores.shape or reference_scores.ndim != 3:
        raise RuntimeError("one2one score tensors changed shape or rank")
    batch, _, anchors = reference_scores.shape
    selected_overlap: list[float] = []
    anchor_overlap: list[float] = []
    ordered_pair_match: list[float] = []
    ordered_anchor_match: list[float] = []
    for batch_index in range(batch):
        reference_flat = reference_scores[batch_index].reshape(-1)
        candidate_flat = candidate_scores[batch_index].reshape(-1)
        count = min(300, reference_flat.numel())
        reference_indices = torch.topk(reference_flat, count).indices
        candidate_indices = torch.topk(candidate_flat, count).indices
        reference_set = set(reference_indices.cpu().tolist())
        candidate_set = set(candidate_indices.cpu().tolist())
        reference_anchors = set((reference_indices % anchors).cpu().tolist())
        candidate_anchors = set((candidate_indices % anchors).cpu().tolist())
        selected_overlap.append(len(reference_set & candidate_set) / count)
        anchor_overlap.append(
            len(reference_anchors & candidate_anchors)
            / max(1, len(reference_anchors | candidate_anchors))
        )
        ordered_pair_match.append(
            float((reference_indices == candidate_indices).float().mean().item())
        )
        ordered_anchor_match.append(
            float(
                ((reference_indices % anchors) == (candidate_indices % anchors))
                .float()
                .mean()
                .item()
            )
        )
    return {
        "topk": min(300, reference_scores[0].numel()),
        "selected_pair_overlap": sum(selected_overlap) / len(selected_overlap),
        "selected_anchor_jaccard": sum(anchor_overlap) / len(anchor_overlap),
        "ordered_pair_match": sum(ordered_pair_match) / len(ordered_pair_match),
        "ordered_anchor_match": sum(ordered_anchor_match) / len(ordered_anchor_match),
    }


def _deployment_comparison(
    reference: Any,
    candidate: Any,
    *,
    task: str,
) -> dict[str, Any]:
    before = _flatten_tensors(reference)
    after = _flatten_tensors(candidate)
    decoded_path = f"output.{task}[0]"
    raw_paths = [
        f"output.{task}[1].one2one.boxes",
        f"output.{task}[1].one2one.scores",
    ]
    if task == "pose":
        raw_paths.append(f"output.{task}[1].one2one.kpts")
    required = (decoded_path, *raw_paths)
    missing = tuple(
        path for path in required if path not in before or path not in after
    )
    if missing:
        raise RuntimeError(
            "deployment comparison tensors are missing: " + ", ".join(missing)
        )
    reference_raw = {path: before[path] for path in raw_paths}
    candidate_raw = {path: after[path] for path in raw_paths}
    scores_path = f"output.{task}[1].one2one.scores"
    return {
        "decoded": _compare_outputs(before[decoded_path], after[decoded_path]),
        "one2one_raw": _compare_outputs(reference_raw, candidate_raw),
        "topk_overlap": _topk_overlap(before[scores_path], after[scores_path]),
    }


def _parse_bits(value: str) -> tuple[int, ...]:
    try:
        bits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bits must be comma-separated integers"
        ) from error
    if not bits:
        raise argparse.ArgumentTypeError(
            "at least one activation bit width is required"
        )
    return bits


def _path_evidence(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _run_cell(
    *,
    activation: str,
    bits: int,
    calibration: Mapping[str, tuple[Tensor, ...]],
    probes: Mapping[str, Tensor],
    device_index: int,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    built = Full35ActivationAdapter().build(
        Full35ActivationPolicy(activation=activation, bits=bits)
    )
    build_seconds = time.perf_counter() - started
    model = built.model.cuda(device_index).eval()

    calibration_started = time.perf_counter()
    with torch.inference_mode():
        for task, images in calibration.items():
            for image in images:
                model(image.cuda(device_index), task=task)
    torch.cuda.synchronize(device_index)
    calibration_seconds = time.perf_counter() - calibration_started
    ranges = built.applied.observer_ranges()
    invalid_ranges = tuple(
        path
        for path, observed in ranges.items()
        if observed is None or observed[1] <= observed[0]
    )
    if invalid_ranges:
        raise RuntimeError(
            "deployment activation observers are missing or degenerate: "
            + ", ".join(invalid_ranges)
        )

    built.applied.disable_quantization()
    float_started = time.perf_counter()
    with torch.inference_mode():
        float_outputs = {
            task: model(image.cuda(device_index), task=task)
            for task, image in probes.items()
        }
    torch.cuda.synchronize(device_index)
    float_seconds = time.perf_counter() - float_started

    built.applied.freeze_observers()
    quantized_started = time.perf_counter()
    with torch.inference_mode():
        quantized_outputs = {
            task: model(image.cuda(device_index), task=task)
            for task, image in probes.items()
        }
    torch.cuda.synchronize(device_index)
    quantized_seconds = time.perf_counter() - quantized_started

    comparisons = {
        task: {
            **_compare_outputs(float_outputs[task], quantized_outputs[task]),
            "deployment": _deployment_comparison(
                float_outputs[task],
                quantized_outputs[task],
                task=task,
            ),
        }
        for task in probes
    }
    result = {
        "policy_id": built.policy.policy_id,
        "activation": activation,
        "activation_bits": bits,
        "weight_format": "fp32",
        "formal_training": False,
        "quantized_deployment_sites": built.applied.quantizer_count,
        "training_only_unquantized_sites": len(built.training_only_paths),
        "training_only_regions": list(built.training_only_regions),
        "protected_modules": built.protected_after,
        "observer": {
            "valid_sites": len(ranges),
            "invalid_sites": list(invalid_ranges),
            "minimum": min(value[0] for value in ranges.values() if value is not None),
            "maximum": max(value[1] for value in ranges.values() if value is not None),
            "ranges": {
                path: [observed[0], observed[1]]
                for path, observed in ranges.items()
                if observed is not None
            },
        },
        "tasks": comparisons,
        "all_finite": all(item["all_finite"] for item in comparisons.values()),
        "same_structure": all(item["same_structure"] for item in comparisons.values()),
        "timing_seconds": {
            "build_cpu": build_seconds,
            "calibration_gpu": calibration_seconds,
            "matched_fp_probe_gpu": float_seconds,
            "fake_quant_probe_gpu": quantized_seconds,
        },
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device_index) / 2**20,
    }
    del model, built, float_outputs, quantized_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="執行不含訓練的 Full35 activation×A3–A8 GPU smoke matrix",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bits", type=_parse_bits, default=(3, 4, 5, 6, 7, 8))
    parser.add_argument("--calibration-per-task", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.calibration_per_task < 1:
        parser.error("--calibration-per-task must be positive")
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        parser.error(f"CUDA device {args.device} is unavailable")

    intake = ActivationIntake.from_workspace(
        activation_root=ACTIVATION_ROOT,
        full35_root=FULL35_ROOT,
    ).analyze()
    plan = CoupledExperimentPlanner(activation_bits=args.bits).activation_screen(intake)
    quantized_cells = tuple(
        cell for cell in plan.cells if cell.activation_bits is not None
    )
    calibration_paths = {
        "detect": _coco_paths("train", args.calibration_per_task),
        "pose": _bbat5_paths("train", args.calibration_per_task),
    }
    probe_paths = {
        "detect": _coco_paths("val", 1)[0],
        "pose": _bbat5_paths("val", 1)[0],
    }
    image_size = 640
    calibration = {
        task: tuple(_letterbox(path, image_size) for path in paths)
        for task, paths in calibration_paths.items()
    }
    probes = {task: _letterbox(path, image_size) for task, path in probe_paths.items()}
    intake_files = {
        digest.role: digest.sha256
        for digest in intake.files
        if digest.role
        in {
            "qsilu_float_source",
            "qsilu_bittrue_source",
            "activation_manifest",
            "activation_recipe",
            "full35_release_manifest",
            "full35_joint_config",
            "full35_accepted_checkpoint",
        }
    }
    contract = {
        "activation_candidates": list(plan.candidate_activations),
        "preferred_seed": plan.preferred_seed,
        "activation_bits": list(args.bits),
        "weight_format": "fp32",
        "formal_training": False,
        "selection_claim": False,
        "comparison_contract": "decoded_one2one_raw_topk_v2",
        "image_size": image_size,
        "calibration_per_task": args.calibration_per_task,
        "calibration_split": "canonical_train_exemplars_only",
        "probe_split": "canonical_val_exemplars_only",
        "bbat5_dataset_id": "bbat5-v1",
        "bbat5_assignment_changed": False,
        "intake_finalization_ready": intake.ready,
        "intake_files": intake_files,
        "calibration_images": {
            task: _path_evidence(paths) for task, paths in calibration_paths.items()
        },
        "probe_images": {
            task: _path_evidence((path,)) for task, path in probe_paths.items()
        },
    }
    output = args.output.expanduser().resolve()
    results: dict[str, Any] = {}
    if args.resume and output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("contract") != contract:
            raise RuntimeError(
                "existing smoke report contract differs; refusing resume"
            )
        raw_results = existing.get("results", {})
        if isinstance(raw_results, dict):
            results.update(raw_results)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "full35_activation_output_gpu_smoke_v2",
        "status": "running",
        "contract": contract,
        "matrix_cells": [cell.to_dict() for cell in quantized_cells],
        "results": results,
    }
    _atomic_json(output, payload)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(args.device)
    for index, cell in enumerate(quantized_cells, start=1):
        if (
            cell.policy_id in results
            and results[cell.policy_id].get("status") == "passed"
        ):
            print(f"[{index}/{len(quantized_cells)}] skip {cell.policy_id}", flush=True)
            continue
        print(f"[{index}/{len(quantized_cells)}] run {cell.policy_id}", flush=True)
        try:
            result = _run_cell(
                activation=cell.activation,
                bits=int(cell.activation_bits),
                calibration=calibration,
                probes=probes,
                device_index=args.device,
            )
            result["status"] = (
                "passed"
                if result["all_finite"] and result["same_structure"]
                else "failed"
            )
        except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as error:
            result = {
                "policy_id": cell.policy_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            gc.collect()
            torch.cuda.empty_cache()
        results[cell.policy_id] = result
        payload["results"] = results
        _atomic_json(output, payload)
        print(
            f"[{index}/{len(quantized_cells)}] {cell.policy_id}: {result['status']}",
            flush=True,
        )

    passed = sum(item.get("status") == "passed" for item in results.values())
    failed = sum(item.get("status") == "failed" for item in results.values())
    payload["status"] = "completed" if failed == 0 else "completed_with_failures"
    payload["summary"] = {
        "planned": len(quantized_cells),
        "passed": passed,
        "failed": failed,
    }
    _atomic_json(output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if failed == 0 and passed == len(quantized_cells) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
