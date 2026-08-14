"""Model construction behind the clean experiment interface."""

from __future__ import annotations

from pathlib import Path

from ..models.builder import build_b1r_model, build_p3_model
from ..models.transfer import transfer_same_graph_compatible
from .contracts import CLEAN_EXPERIMENTS


def build_clean_model(experiment: str, official_weights: Path):
    """Build one clean model and record an auditable transfer report."""
    try:
        spec = CLEAN_EXPERIMENTS[experiment]
    except KeyError as error:
        raise ValueError(f"unsupported clean experiment: {experiment}") from error
    if spec.family == "B0":
        model = build_p3_model()
        model.masf_transfer_report = transfer_same_graph_compatible(
            model, official_weights
        ).to_dict()
    elif spec.family == "P3":
        model = build_p3_model(spec.variant, source_weights=official_weights)
    elif spec.family == "P2":
        model = build_b1r_model(source_weights=official_weights)
    else:  # pragma: no cover - locked specs make this unreachable
        raise ValueError(f"unsupported clean family: {spec.family}")
    model.masf_variant = experiment
    model.masf_variant_hash = f"clean:{experiment}"
    model.masf_initializer = "ultralytics-yolo11m-coco80-v8.3.0"
    return model
