"""Artifact lifecycle for the 10-epoch attention-only fine-tuning plan."""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

import torch
from torch import nn
from ultralytics.nn.modules.block import Attention

from .attention.base import (
    FPAttention,
    MagnitudeSideChannelAttention,
    ParallelDualBinaryAttention,
    ResidualDualFullBasisAttention,
    ResidualDualMatchedBasisAttention,
    ScaledBinaryAttention,
    SignOnlyBinaryAttention,
)
from .config import materialize_variant_yaml
from .report import _read_results, render_experiment_report
from .training_profiles import FORMAL_EPOCHS, FORMAL_TRAINING_ARGS, PAPER_REFERENCE_EPOCHS
from .variants.definitions import VariantDefinition, get_variant, quantization_contract
from .verification.checkpoint import binary_modules, save_checkpoint, strict_reload
from .model import _source_state, build_student


PLAN_NAME = "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def _git_worktree_state() -> dict[str, str | bool]:
    """Fingerprint tracked changes without embedding a potentially large diff."""

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], capture_output=True, check=True,
        ).stdout
        return {
            "git_tracked_dirty": bool(status.strip()),
            "git_tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {"git_tracked_dirty": True, "git_tracked_diff_sha256": "unavailable"}


def _environment(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpu = result.stdout.strip() if result.returncode == 0 else ""
    except OSError:
        gpu = ""
    if not gpu and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = f"{properties.name}, {properties.total_memory / (1024 ** 2):.0f} MiB (PyTorch fallback)"
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu or "unavailable",
        "ultralytics": __import__("ultralytics").__version__,
        "git_revision": _git_revision(),
        "command": command,
    }
    environment.update(_git_worktree_state())
    return environment


def create_run_artifact(
    root: Path,
    variant: VariantDefinition,
    stage: str,
    command: list[str],
    *,
    data_manifest: Path | None = None,
) -> Path:
    """Create the complete immutable run directory before any GPU work."""

    if stage not in {"full", "validation"}:
        raise ValueError("only full attention-only training or zero-training validation stages are supported")
    run = root / "artifacts" / "runs" / variant.artifact_name / stage / str(uuid.uuid4())
    for directory in (run / "logs", run / "checkpoints"):
        directory.mkdir(parents=True)
    materialize_variant_yaml(variant, run / "model.yaml")
    resolved = {
        **asdict(variant),
        "config_hash": variant.config_hash,
        "quantization_contract": quantization_contract(variant),
        "plan_name": PLAN_NAME,
        "baseline_source": "original/weight/yolo11m.pt",
        "baseline_epochs": 600,
        "paper_reference_epochs": PAPER_REFERENCE_EPOCHS,
        "formal_epochs": FORMAL_EPOCHS,
        "run_epochs": FORMAL_EPOCHS if stage == "full" else 0,
        "trainable_scope": "attention_only" if stage == "full" else "none",
        "initialization": "full-precision source checkpoint fine-tuning",
        "teacher": "full-precision source checkpoint",
        "paper_profile": (
            "BinaryAttention QAT fine-tuning; batch=128, lr=5e-5, min_lr=5e-6, wd=0.02"
            if stage == "full" and variant.use_qat
            else "FP attention-only control; batch=128, lr=5e-5, min_lr=5e-6, wd=0.02"
            if stage == "full"
            else "zero-training source-checkpoint validation"
        ),
        "dataset_scope": "COCO2017 train2017/val2017 full",
        "engineering_gates": "not executed by plan",
    }
    _write_json(run / "resolved_config.json", resolved)
    training_args = dict(FORMAL_TRAINING_ARGS) if stage == "full" else {"epochs": 0, "batch": 128, "imgsz": 640, "seed": 0}
    training_args.update({"stage": stage, "model": str((run / "model.yaml").resolve()), "data_manifest": str(data_manifest.resolve()) if data_manifest else None})
    _write_json(run / "training_args.json", training_args)
    _write_json(run / "environment.json", _environment(command))
    _write_json(run / "architecture_manifest.json", {"status": "pending", "construction": "parser-direct-concrete-attention"})
    _write_json(run / "checkpoint_manifest.json", {"status": "pending"})
    _write_json(run / "validation_metrics.json", {"status": "pending"})
    _write_json(run / "attention_diagnostics.json", {"status": "pending"})
    if stage == "full":
        _write_json(run / "parameter_delta_diagnostics.json", {"status": "pending"})
    (run / "training_curves.csv").write_text("epoch\n")
    status = {
        "plan_name": PLAN_NAME,
        "stage": stage,
        "completed": False,
        "valid_for_research": False,
        "engineering_gates": "not executed by plan",
        "g0_g5": "not executed",
        "micro_pilot": "not executed",
        "data_manifest": str(data_manifest.resolve()) if data_manifest else "COCO2017 full train2017/val2017",
        "seed": 0,
        "batch": 128,
        "epochs": FORMAL_EPOCHS if stage == "full" else 0,
        "initialization": "full-precision source checkpoint fine-tuning" if stage == "full" else "source checkpoint validation",
        "trainable_scope": "attention_only" if stage == "full" else "none",
    }
    _write_json(run / "status.json", status)
    render_experiment_report(run, resolved, status)
    return run


def _architecture_manifest(model: nn.Module, variant: VariantDefinition) -> dict:
    modules = binary_modules(model)
    all_parameter_names = [name for name, _parameter in model.named_parameters()]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_scope = (
        "none" if not trainable_names else "all" if len(trainable_names) == len(all_parameter_names) else "attention_only"
    )
    return {
        "construction": "parser-direct-concrete-attention",
        "variant_id": variant.id,
        "variant_config_hash": variant.config_hash,
        "attention_class": sorted({type(module).__name__ for _, module in modules}) if modules else ["Attention"],
        "attention_module_paths": [name for name, _ in modules],
        "attention_module_count": len(modules),
        "theoretical_cost": variant.theoretical_cost,
        "quantization_contract": quantization_contract(variant),
        "model_class": type(model).__name__,
        "trainable_scope": trainable_scope,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _attention_diagnostics(model: nn.Module, variant: VariantDefinition) -> dict:
    modules = binary_modules(model)
    entries = []
    for name, module in modules:
        scores = getattr(module, "last_scores", None)
        probability = getattr(module, "last_probabilities", None)
        entries.append(
            {
                "path": name,
                "class": type(module).__name__,
                "binary_forward_count": int(module.binary_forward_count.item()),
                "binary_qk_count": int(module.binary_qk_count.item()),
                "softmax_count": int(module.softmax_count.item()),
                "pv_count": int(module.pv_count.item()),
                "score_shape": list(scores.shape) if isinstance(scores, torch.Tensor) else None,
                "probability_shape": list(probability.shape) if isinstance(probability, torch.Tensor) else None,
                "score_finite": bool(torch.isfinite(scores).all()) if isinstance(scores, torch.Tensor) else None,
                "probability_finite": bool(torch.isfinite(probability).all()) if isinstance(probability, torch.Tensor) else None,
            }
        )
    return {
        "variant_id": variant.id,
        "theoretical_cost": variant.theoretical_cost,
        "modules": entries,
        "fake_quantization": bool(variant.p_bits or variant.v_bits or variant.magnitude_bits),
        "hardware_speed_claim": False,
    }


def _attention_only_delta_diagnostics(
    model: nn.Module,
    variant: VariantDefinition,
    model_yaml: Path,
    source_weights: Path,
) -> dict:
    """Prove frozen tensors stayed at the FP source while Attention changed."""

    source_model = build_student(model_yaml, variant, source_weights)
    source_parameters = dict(source_model.named_parameters())
    original_source_keys = set(_source_state(source_weights))
    attention_paths = tuple(name for name, module in model.named_modules() if isinstance(module, Attention))

    def inside_attention(name: str) -> bool:
        return any(name == path or name.startswith(f"{path}.") for path in attention_paths)

    frozen_deltas = []
    changed_source_attention = []
    missing_frozen = []
    for name, parameter in model.named_parameters():
        baseline = source_parameters.get(name)
        if baseline is None or baseline.shape != parameter.shape:
            if not inside_attention(name):
                missing_frozen.append(name)
            continue
        delta = float((parameter.detach().cpu() - baseline.detach().cpu()).abs().max())
        if inside_attention(name):
            if name in original_source_keys and delta > 0:
                changed_source_attention.append(name)
        else:
            frozen_deltas.append(delta)

    source_buffers = dict(source_model.named_buffers())
    frozen_bn_deltas = []
    missing_bn_buffers = []
    non_attention_bn_paths = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d) and not inside_attention(name)
    )
    for name, buffer in model.named_buffers():
        if not any(name == path or name.startswith(f"{path}.") for path in non_attention_bn_paths):
            continue
        baseline = source_buffers.get(name)
        if baseline is None or baseline.shape != buffer.shape:
            missing_bn_buffers.append(name)
            continue
        if buffer.is_floating_point():
            delta = float((buffer.detach().cpu() - baseline.detach().cpu()).abs().max())
        else:
            delta = 0.0 if torch.equal(buffer.detach().cpu(), baseline.detach().cpu()) else float("inf")
        frozen_bn_deltas.append(delta)

    max_frozen = max(frozen_deltas, default=float("inf"))
    max_frozen_bn = max(frozen_bn_deltas, default=float("inf"))
    passed = (
        not missing_frozen
        and not missing_bn_buffers
        and max_frozen == 0.0
        and max_frozen_bn == 0.0
        and bool(changed_source_attention)
    )
    return {
        "variant_id": variant.id,
        "passed": passed,
        "frozen_parameter_tensor_count": len(frozen_deltas),
        "max_frozen_parameter_delta": max_frozen,
        "frozen_non_attention_bn_buffer_count": len(frozen_bn_deltas),
        "max_frozen_non_attention_bn_buffer_delta": max_frozen_bn,
        "changed_source_attention_tensor_count": len(changed_source_attention),
        "changed_source_attention_parameter_names": changed_source_attention,
        "missing_frozen_parameter_names": missing_frozen,
        "missing_frozen_bn_buffer_names": missing_bn_buffers,
    }


def _copy_training_curves(run: Path) -> None:
    source = run / "ultralytics" / "train" / "results.csv"
    target = run / "training_curves.csv"
    if source.exists():
        shutil.copyfile(source, target)
    elif not target.exists():
        target.write_text("epoch\n")


def _complete_metrics(run: Path) -> dict:
    metrics, rows = _read_results(run)
    for key in ("mAP50_95", "mAP50", "mAP75", "mAPs", "mAPm", "mAPl"):
        metrics.setdefault(key, None)
    _write_json(run / "validation_metrics.json", metrics or {"status": "not available"})
    return metrics


def finalize_training_artifact(
    run: Path,
    model: nn.Module,
    variant: VariantDefinition,
    *,
    source_weights: Path | None = None,
) -> Path:
    """Persist strict checkpoint, diagnostics, curves and report after training."""

    strict_path = run / "checkpoints" / "strict-last.pt"
    manifest = save_checkpoint(
        strict_path,
        model,
        variant,
        model_yaml=run / "model.yaml",
        source_checkpoint=source_weights,
    )
    strict_reload(strict_path, build_student(run / "model.yaml", variant), variant)
    _write_json(run / "checkpoint_manifest.json", asdict(manifest))
    _write_json(run / "architecture_manifest.json", _architecture_manifest(model, variant))
    _write_json(run / "attention_diagnostics.json", _attention_diagnostics(model, variant))
    if source_weights is None:
        raise ValueError("formal training artifact requires FP source weights for freeze-delta verification")
    _write_json(
        run / "parameter_delta_diagnostics.json",
        _attention_only_delta_diagnostics(model, variant, run / "model.yaml", source_weights),
    )
    _copy_training_curves(run)
    metrics = _complete_metrics(run)
    status = json.loads((run / "status.json").read_text())
    status.update(
        {
            "completed": True,
            "checkpoint_reload_verified": True,
            "valid_for_research": True,
            "pipeline_gates_passed": "not applicable; G0-G5 not executed",
            "metrics_available": bool(metrics),
            "conclusion": "completed formal artifact; interpret metrics from validation_metrics.json",
        }
    )
    _write_json(run / "status.json", status)
    render_experiment_report(run, json.loads((run / "resolved_config.json").read_text()), status)
    return strict_path


def execute_zero_training_validation(
    run: Path,
    variant: VariantDefinition,
    data: Path,
    source_weights: Path,
    device: str = "0",
) -> Path:
    """Validate E0/E1 variants once, without an optimizer or training epochs."""

    from ultralytics.models.yolo.detect.val import DetectionValidator

    model = build_student(run / "model.yaml", variant, source_weights)
    model.eval()
    args = {
        "data": str(data),
        "imgsz": 640,
        "batch": 128,
        "workers": 8,
        "device": device,
        "split": "val",
        "plots": True,
        "rect": True,
        "project": str(run / "validation"),
        "name": "val",
        "exist_ok": False,
    }
    validator = DetectionValidator(args=args, save_dir=run / "validation")
    # Ultralytics' validator may fuse Conv+BN in-place.  Save the clean
    # parser-built state before validation so strict reload always compares
    # like-for-like unfused architecture.
    checkpoint_path = run / "checkpoints" / "strict-validation.pt"
    manifest = save_checkpoint(
        checkpoint_path,
        model,
        variant,
        model_yaml=run / "model.yaml",
        source_checkpoint=source_weights,
    )
    validator(model=model)
    metrics = getattr(validator.metrics, "results_dict", {}) or {}
    metric_aliases = {
        "metrics/mAP50-95(B)": "mAP50_95",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP75(B)": "mAP75",
    }
    metrics.update({target: metrics[source] for source, target in metric_aliases.items() if source in metrics})
    for key in ("mAP50_95", "mAP50", "mAP75", "mAPs", "mAPm", "mAPl"):
        metrics.setdefault(key, None)
    _write_json(run / "validation_metrics.json", metrics)
    _write_json(run / "architecture_manifest.json", _architecture_manifest(model, variant))
    _write_json(run / "attention_diagnostics.json", _attention_diagnostics(model, variant))
    strict_reload(checkpoint_path, build_student(run / "model.yaml", variant), variant)
    _write_json(run / "checkpoint_manifest.json", asdict(manifest))
    status = json.loads((run / "status.json").read_text())
    status.update({"completed": True, "valid_for_research": True, "checkpoint_reload_verified": True,
                   "metrics_available": bool(metrics), "epochs": 0,
                   "conclusion": "zero-training validation completed"})
    _write_json(run / "status.json", status)
    render_experiment_report(run, json.loads((run / "resolved_config.json").read_text()), status)
    return run


def _module_for(variant: VariantDefinition) -> nn.Module:
    kwargs = {"use_qat": variant.use_qat, "p_bits": variant.p_bits, "v_bits": variant.v_bits}
    if variant.qk_mode == "fp":
        return FPAttention(16, 2)
    if variant.attention_type == "SignOnlyBinaryAttention":
        return SignOnlyBinaryAttention(16, 2, **kwargs)
    if variant.attention_type == "ScaledBinaryAttention":
        return ScaledBinaryAttention(16, 2, **kwargs)
    mapping = {
        "ParallelDualBinaryAttention": ParallelDualBinaryAttention,
        "ResidualDualFullBasisAttention": ResidualDualFullBasisAttention,
        "ResidualDualMatchedBasisAttention": ResidualDualMatchedBasisAttention,
        "MagnitudeSideChannelAttention": MagnitudeSideChannelAttention,
    }
    return mapping[variant.attention_type](16, num_heads=2, **kwargs, magnitude_bits=variant.magnitude_bits)


def run_local_checks(root: Path, variant_id: str) -> Path:
    """Run numerical construction checks only; no G0–G5 gate is executed."""

    variant = get_variant(variant_id)
    run = create_run_artifact(root, variant, "validation", ["python", "-m", "binary_attention.cli", "verify", "--variant", variant_id])
    model = _module_for(variant)
    model.train()
    output = model(torch.randn(2, 16, 4, 4, requires_grad=True))
    output.square().mean().backward()
    qkv = getattr(model, "qkv", None)
    qkv_weight = getattr(qkv, "weight", None) if qkv is not None else None
    if qkv_weight is None and qkv is not None:
        qkv_weight = getattr(getattr(qkv, "conv", None), "weight", None)
    if variant.qk_mode != "fp" and qkv_weight is not None and qkv_weight.grad is None:
        raise RuntimeError("binary QKV path has no gradient")
    checkpoint = run / "checkpoints" / "self-check.pt"
    manifest = save_checkpoint(checkpoint, model, variant)
    restored = _module_for(variant)
    strict_reload(checkpoint, restored, variant)
    _write_json(run / "architecture_manifest.json", _architecture_manifest(model, variant))
    _write_json(run / "checkpoint_manifest.json", asdict(manifest))
    _write_json(run / "attention_diagnostics.json", _attention_diagnostics(model, variant))
    _write_json(run / "validation_metrics.json", {"status": "numerical self-check only; no COCO metrics"})
    status = json.loads((run / "status.json").read_text())
    status.update({"completed": True, "valid_for_research": False, "self_check": "passed", "conclusion": "numerical self-check only"})
    _write_json(run / "status.json", status)
    render_experiment_report(run, json.loads((run / "resolved_config.json").read_text()), status)
    return run


# Compatibility name for clients of the earlier draft.  It performs the same
# non-research numerical self-check and does not run any engineering gate.
run_local_gates = run_local_checks
