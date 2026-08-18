from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from achitechure_1.checkpoint import build_training_model
from achitechure_1.training import make_bittrue_validation_copy

ROOT = Path(__file__).resolve().parents[1]


def test_early_stopping_validation_uses_bittrue_copy_without_mutating_float_model() -> None:
    source = YOLO(str(ROOT / "inputs/parent/best.pt")).model
    model, _ = build_training_model(
        cfg=source.yaml,
        nc=80,
        channels=3,
        weights=source,
        masf_variant="full35",
        attention_config=ROOT / "configs/attention/float-pwl-final.yaml",
        verbose=False,
    )

    bittrue = make_bittrue_validation_copy(
        model,
        ROOT / "configs/attention/bittrue-pwl-final.yaml",
    )

    source_kinds = [m.config.normalization.value for m in model.modules() if hasattr(m, "config")]
    target_kinds = [m.config.normalization.value for m in bittrue.modules() if hasattr(m, "config")]
    assert "piecewise_linear" in source_kinds
    assert "bit_true_pwl" in target_kinds
    assert bittrue.model[16].p3_masf is not model.model[16].p3_masf
