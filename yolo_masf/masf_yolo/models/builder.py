"""Build the fixed P2 template with slots installed before first forward."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.nn import tasks
from ultralytics.nn.modules import Detect, DWConv, Conv

from masf_yolo.variants import VariantDefinition, get_variant, is_selective_variant

from .mfam import MFAM, PartialMFAM, PaperFormulaMFAM, PartialPaperFormulaMFAM
from .selective import SelectiveDetectionLoss, SelectiveP2Detect


P2_SLOT_INDEX = 20
P3_SLOT_INDEX = 24
P2_CHANNELS = 128
P3_CHANNELS = 256
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "yolo11m-p2-slots.yaml"
B1R_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "yolo11m-p2-slots-b1r.yaml"
B1R_CLS_CHANNELS = (128, 256, 256, 256)


class PerScaleDetect(Detect):
    """Four-scale Detect with a new P2 tower and B0-compatible P3-P5 towers."""

    def __init__(self, nc: int, ch: tuple[int, ...], cls_channels: tuple[int, ...]) -> None:
        if len(ch) != len(cls_channels):
            raise ValueError("one classification width is required per detection scale")
        super().__init__(nc=nc, reg_max=16, ch=ch, cls_channels=max(cls_channels))
        self.cls_channels_per_scale = tuple(cls_channels)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, width, 1)),
                nn.Sequential(DWConv(width, width, 3), Conv(width, width, 1)),
                nn.Conv2d(width, self.nc, 1),
            )
            for x, width in zip(ch, cls_channels)
        )


class MasfDetectionModel(tasks.DetectionModel):
    """DetectionModel that selects the repository-owned SP2 criterion when needed."""

    def init_criterion(self):
        if isinstance(self.model[-1], SelectiveP2Detect):
            return SelectiveDetectionLoss(self)
        return super().init_criterion()


def _with_metadata(replacement: nn.Module, original: nn.Module) -> nn.Module:
    replacement.i = original.i  # type: ignore[attr-defined]
    replacement.f = original.f  # type: ignore[attr-defined]
    replacement.type = f"{replacement.__class__.__module__}.{replacement.__class__.__name__}"  # type: ignore[attr-defined]
    replacement.np = sum(parameter.numel() for parameter in replacement.parameters())  # type: ignore[attr-defined]
    return replacement


def _slot_module(slot: str, variant: VariantDefinition, channels: int) -> nn.Module:
    if slot == "identity":
        return nn.Identity()
    if slot == "mfam":
        return MFAM(channels, kernels=variant.kernel_branches)
    if slot == "partial_mfam":
        return PartialMFAM(
            channels,
            processed_ratio=variant.processed_ratio,
            kernels=variant.kernel_branches,
        )
    raise ValueError(f"unsupported feature slot: {slot}")


def _install_slots(model: tasks.DetectionModel, variant: VariantDefinition) -> None:
    model.model[P2_SLOT_INDEX] = _with_metadata(
        _slot_module(variant.p2_slot, variant, P2_CHANNELS), model.model[P2_SLOT_INDEX]
    )
    model.model[P3_SLOT_INDEX] = _with_metadata(
        _slot_module(variant.p3_slot, variant, P3_CHANNELS), model.model[P3_SLOT_INDEX]
    )


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
    *,
    _template_path: Path = TEMPLATE_PATH,
) -> tasks.DetectionModel:
    """Build a variant without modifying Ultralytics files or module globals."""
    if source_weights is not None and checkpoint is not None:
        raise ValueError("source_weights and checkpoint are mutually exclusive")
    definition = get_variant(variant) if isinstance(variant, str) else variant
    model = MasfDetectionModel.__new__(MasfDetectionModel)
    tasks.BaseModel.__init__(model)

    previous_legacy = Detect.legacy
    try:
        tasks._initialize_yolo_model(model, str(_template_path), 3, 2, False)
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
    if is_selective_variant(definition.variant_id):
        standard = model.model[-1]
        if not isinstance(standard, Detect):
            raise TypeError("SP2 template must end in Detect before head replacement")
        selective = SelectiveP2Detect(standard, p2_channels=P2_CHANNELS)
        model.model[-1] = _with_metadata(selective, standard)
        model.stride = selective.stride
        selective.bias_init()
    if checkpoint is not None:
        from masf_yolo.artifacts.checkpoints import load_canonical_checkpoint

        load_canonical_checkpoint(model, checkpoint, definition)
    return model


def build_b1r_model(
    source_weights: Path | None = None,
    checkpoint: Path | None = None,
) -> tasks.DetectionModel:
    """Build the revised B1 graph while preserving B0 P3-P5 head shapes."""
    if source_weights is not None and checkpoint is not None:
        raise ValueError("source_weights and checkpoint are mutually exclusive")
    model = build_model("B1", _template_path=B1R_TEMPLATE_PATH)
    standard = model.model[-1]
    if not isinstance(standard, Detect):
        raise TypeError("P2 template must end in Detect")
    revised = PerScaleDetect(
        nc=standard.nc,
        ch=(P2_CHANNELS, P3_CHANNELS, 512, 512),
        cls_channels=B1R_CLS_CHANNELS,
    )
    revised.stride = standard.stride.clone()
    model.model[-1] = _with_metadata(revised, standard)
    model.stride = revised.stride
    revised.bias_init()
    model.masf_variant = "B1R"
    model.masf_variant_hash = "b1r-per-scale-detect-v1"
    if source_weights is not None:
        from .transfer import transfer_official_weights

        model.masf_transfer_report = transfer_official_weights(model, source_weights).to_dict()
    if checkpoint is not None:
        from masf_yolo.artifacts.checkpoints import load_canonical_checkpoint

        load_canonical_checkpoint(model, checkpoint, "B1")
    return model


def build_p3_model(
    variant: str | None = None,
    source_weights: Path | None = None,
) -> tasks.DetectionModel:
    """Build the original three-scale B0 graph with an optional P3 MFAM slot."""
    model = MasfDetectionModel.__new__(MasfDetectionModel)
    tasks.BaseModel.__init__(model)
    previous_legacy = Detect.legacy
    try:
        tasks._initialize_yolo_model(
            model,
            str(Path(__file__).resolve().parents[2] / "bbt5-detect-baseline" / "yolo11m_detect_2cls.yaml"),
            3,
            2,
            False,
        )
        parsed_legacy = Detect.legacy
        model.model[-1].legacy = parsed_legacy
    finally:
        Detect.legacy = previous_legacy
    tasks.initialize_weights(model)
    _initialize_strides(model)
    model.names = {0: "ball", 1: "bat"}
    model.args = get_cfg()
    model.masf_variant = "P3-Base-Original" if variant is None else f"P3-{variant}"
    model.masf_variant_hash = "p3-b0-base-v1" if variant is None else f"p3-paper-{variant}"
    if variant is not None:
        from .transfer import transfer_b0_p3_parent

        definition = get_variant("M0")
        if variant == "PaperFormula-Full":
            replacement = PaperFormulaMFAM(256, kernels=(3, 5, 7, 9))
        elif variant == "Lite-35":
            replacement = PaperFormulaMFAM(256, kernels=(3, 5))
        elif variant == "Lite-35-F7":
            replacement = PaperFormulaMFAM(256, kernels=(3, 5, 7))
        elif variant == "Partial50-35":
            replacement = PartialPaperFormulaMFAM(256, 0.5, kernels=(3, 5))
        elif variant == "Partial25-35":
            replacement = PartialPaperFormulaMFAM(256, 0.25, kernels=(3, 5))
        else:
            raise ValueError(f"unsupported P3 variant: {variant}")
        model.model[16] = _with_metadata(replacement, model.model[16])
        if source_weights is not None:
            model.masf_transfer_report = transfer_b0_p3_parent(model, source_weights).to_dict()
    elif source_weights is not None:
        from ultralytics import YOLO

        source_model = YOLO(str(source_weights), task="detect").model
        model.load_state_dict(source_model.state_dict(), strict=True)
    return model


def build_p2_retest_model(
    variant: str,
    source_weights: Path | None = None,
) -> tasks.DetectionModel:
    """Build a paper-formula MFAM variant on the revised B1R P2 slot."""
    model = build_b1r_model(source_weights=source_weights)
    if variant == "PaperFormula-Full":
        replacement = PaperFormulaMFAM(P2_CHANNELS, kernels=(3, 5, 7, 9))
    elif variant == "Lite-35":
        replacement = PaperFormulaMFAM(P2_CHANNELS, kernels=(3, 5))
    elif variant == "Lite-35-F7":
        replacement = PaperFormulaMFAM(P2_CHANNELS, kernels=(3, 5, 7))
    elif variant == "Partial50-35":
        replacement = PartialPaperFormulaMFAM(P2_CHANNELS, 0.5, kernels=(3, 5))
    elif variant == "Partial25-35":
        replacement = PartialPaperFormulaMFAM(P2_CHANNELS, 0.25, kernels=(3, 5))
    else:
        raise ValueError(f"unsupported P2 retest variant: {variant}")
    model.model[P2_SLOT_INDEX] = _with_metadata(replacement, model.model[P2_SLOT_INDEX])
    model.masf_variant = f"P2-{variant}"
    model.masf_variant_hash = f"p2-paper-{variant}"
    return model
