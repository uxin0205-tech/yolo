"""Parent-compatible model construction and Float/Bit-True checkpoint materialization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model

from .model import graft_p3_masf, inspect_yolo26_graph

MINIMUM_STATE_COVERAGE = 0.95


@dataclass(frozen=True)
class TransferReport:
    compatible_tensors: int
    target_tensors: int
    source_tensors: int
    coverage: float
    missing_tensors: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_model(weights: Any) -> nn.Module:
    source = weights.get("model") if isinstance(weights, dict) else weights
    if not isinstance(source, nn.Module):
        raise TypeError("weights must contain an nn.Module")
    return source


def _transfer_report(target: nn.Module, source: nn.Module) -> TransferReport:
    source_state = source.state_dict()
    target_state = target.state_dict()
    compatible = {
        name
        for name, value in source_state.items()
        if name in target_state and value.shape == target_state[name].shape
    }
    missing = tuple(sorted(set(target_state) - compatible))
    unexpected = tuple(sorted(set(source_state) - compatible))
    return TransferReport(
        compatible_tensors=len(compatible),
        target_tensors=len(target_state),
        source_tensors=len(source_state),
        coverage=len(compatible) / len(target_state),
        missing_tensors=missing,
        unexpected_tensors=unexpected,
    )


def build_training_model(
    *,
    cfg: str | dict[str, Any] | None,
    nc: int,
    channels: int,
    weights: Any,
    masf_variant: str,
    attention_config: str | Path,
    verbose: bool,
) -> tuple[DetectionModel, TransferReport]:
    """Rebuild Float-PWL YOLO26m, graft MASF, then transfer the complete parent state."""

    source = _source_model(weights)
    model = DetectionModel(cfg, nc=nc, ch=channels, verbose=verbose)
    convert_yolo26_model(model, VariantConfig.from_yaml(attention_config))
    graft_p3_masf(model, masf_variant)
    report = _transfer_report(model, source)
    if report.coverage < MINIMUM_STATE_COVERAGE:
        raise RuntimeError(
            f"incompatible parent: {report.compatible_tensors}/{report.target_tensors} "
            f"({report.coverage:.3%}) state tensors match"
        )
    model.load(source)
    inspect_yolo26_graph(model)
    return model, report


def materialize_bittrue_checkpoint(
    float_checkpoint: str | Path,
    bittrue_config: str | Path,
    destination: str | Path,
) -> Path:
    """Convert the actual Float checkpoint to Bit-True attention and prove reloadability."""

    source = Path(float_checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    yolo = YOLO(str(source))
    convert_yolo26_model(yolo.model, VariantConfig.from_yaml(bittrue_config))
    inspect_yolo26_graph(yolo.model)
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(target))
    reloaded = YOLO(str(target)).model
    inspect_yolo26_graph(reloaded)
    return target


def prepare_variant_checkpoint(
    parent_checkpoint: str | Path,
    variant: str,
    destination: str | Path,
) -> tuple[Path, TransferReport]:
    """Graft a new identity-safe MASF branch onto the immutable Bit-True parent."""

    parent = Path(parent_checkpoint).resolve()
    yolo = YOLO(str(parent))
    source_state = yolo.model.state_dict()
    graft = graft_p3_masf(yolo.model, variant)
    target_state = yolo.model.state_dict()
    compatible = tuple(name for name, value in source_state.items() if torch.equal(value, target_state[name]))
    missing = tuple(sorted(set(target_state) - set(source_state)))
    report = TransferReport(
        compatible_tensors=len(compatible),
        target_tensors=len(target_state),
        source_tensors=len(source_state),
        coverage=len(compatible) / len(target_state),
        missing_tensors=missing,
        unexpected_tensors=(),
    )
    if len(compatible) != graft.preserved_tensors:
        raise RuntimeError("parent tensor preservation count changed during graft")
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(target))
    inspect_yolo26_graph(YOLO(str(target)).model)
    return target, report


def export_masf_state(checkpoint: str | Path, destination: str | Path) -> Path:
    """Export only MASF tensors plus checkpoint lineage."""

    source = Path(checkpoint).resolve()
    model = YOLO(str(source)).model
    graph = inspect_yolo26_graph(model)
    masf = getattr(model.model[graph.p3_index], "p3_masf", None)
    if not isinstance(masf, nn.Module):
        raise TypeError("checkpoint does not contain P3 MASF")
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "p3_index": graph.p3_index,
            "checkpoint_sha256": file_sha256(source),
            "state_dict": masf.state_dict(),
        },
        target,
    )
    return target
