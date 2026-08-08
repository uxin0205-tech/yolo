"""Build the fixed P2 template with slots installed before first forward."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.nn import tasks
from ultralytics.nn.modules import Detect

from masf_yolo.variants import VariantDefinition, get_variant

from .mfam import MFAM, PartialMFAM


P2_SLOT_INDEX = 20
P3_SLOT_INDEX = 24
P2_CHANNELS = 128
P3_CHANNELS = 256
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "yolo11m-p2-slots.yaml"


def _with_metadata(replacement: nn.Module, original: nn.Module) -> nn.Module:
    replacement.i = original.i  # type: ignore[attr-defined]
    replacement.f = original.f  # type: ignore[attr-defined]
    replacement.type = f"{replacement.__class__.__module__}.{replacement.__class__.__name__}"  # type: ignore[attr-defined]
    replacement.np = sum(parameter.numel() for parameter in replacement.parameters())  # type: ignore[attr-defined]
    return replacement


def _slot_module(variant: VariantDefinition, channels: int) -> nn.Module:
    if variant.p2_slot == "identity":
        return nn.Identity()
    if variant.p2_slot == "mfam":
        return MFAM(channels, kernels=variant.kernel_branches)
    if variant.p2_slot == "partial_mfam":
        return PartialMFAM(
            channels,
            processed_ratio=variant.processed_ratio,
            kernels=variant.kernel_branches,
        )
    raise ValueError(f"unsupported P2 slot: {variant.p2_slot}")


def _install_slots(model: tasks.DetectionModel, variant: VariantDefinition) -> None:
    model.model[P2_SLOT_INDEX] = _with_metadata(
        _slot_module(variant, P2_CHANNELS), model.model[P2_SLOT_INDEX]
    )
    model.model[P3_SLOT_INDEX] = _with_metadata(nn.Identity(), model.model[P3_SLOT_INDEX])


def _initialize_strides(model: tasks.DetectionModel, channels: int = 3) -> None:
    detect = model.model[-1]
    if not isinstance(detect, Detect):
        raise TypeError("P2 template must end in Detect")
    sample_size = 256
    detect.inplace = model.inplace
    model.model.eval()
    detect.training = True
    with torch.no_grad():
        output = model.forward(torch.zeros(1, channels, sample_size, sample_size))
    features = output["one2many"]["feats"] if model.end2end else output["feats"]
    detect.stride = torch.tensor([sample_size / feature.shape[-2] for feature in features])
    model.stride = detect.stride
    model.model.train()
    detect.bias_init()


def build_model(
    variant: str | VariantDefinition,
    source_weights: Path | None = None,
    checkpoint: Path | None = None,
) -> tasks.DetectionModel:
    """Build a variant without modifying Ultralytics files or module globals."""
    if source_weights is not None and checkpoint is not None:
        raise ValueError("source_weights and checkpoint are mutually exclusive")
    definition = get_variant(variant) if isinstance(variant, str) else variant
    model = tasks.DetectionModel.__new__(tasks.DetectionModel)
    tasks.BaseModel.__init__(model)

    previous_legacy = Detect.legacy
    try:
        tasks._initialize_yolo_model(model, str(TEMPLATE_PATH), 3, 2, False)
        parsed_legacy = Detect.legacy
        model.model[-1].legacy = parsed_legacy
    finally:
        Detect.legacy = previous_legacy

    _install_slots(model, definition)
    _initialize_strides(model)
    tasks.initialize_weights(model)
    model.names = {0: "ball", 1: "bat"}
    model.args = get_cfg()
    model.masf_variant = definition.variant_id
    model.masf_variant_hash = definition.config_hash

    if source_weights is not None:
        from .transfer import transfer_official_weights

        model.masf_transfer_report = transfer_official_weights(model, source_weights).to_dict()
    if checkpoint is not None:
        from masf_yolo.artifacts.checkpoints import load_canonical_checkpoint

        load_canonical_checkpoint(model, checkpoint, definition)
    return model
