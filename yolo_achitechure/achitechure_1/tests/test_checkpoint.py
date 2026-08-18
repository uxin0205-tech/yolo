from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from achitechure_1.checkpoint import build_training_model, materialize_bittrue_checkpoint
from achitechure_1.model import inspect_yolo26_graph

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "inputs" / "parent" / "best.pt"
FLOAT_CONFIG = ROOT / "configs" / "attention" / "float-pwl-final.yaml"
BITTRUE_CONFIG = ROOT / "configs" / "attention" / "bittrue-pwl-final.yaml"


def test_training_build_converts_parent_to_float_and_grafts_masf() -> None:
    source = YOLO(str(PARENT)).model

    model, report = build_training_model(
        cfg=source.yaml,
        nc=80,
        channels=3,
        weights=source,
        masf_variant="full35",
        attention_config=FLOAT_CONFIG,
        verbose=False,
    )

    assert report.coverage >= 0.95
    assert model.model[16].p3_masf.channels == 256
    assert all(
        module.config.normalization.value == "piecewise_linear"
        for module in model.modules()
        if module.__class__.__name__ == "HardwareFriendlyAttention"
    )


def test_float_checkpoint_converts_to_reloadable_bittrue(tmp_path: Path) -> None:
    source_yolo = YOLO(str(PARENT))
    model, _ = build_training_model(
        cfg=source_yolo.model.yaml,
        nc=80,
        channels=3,
        weights=source_yolo.model,
        masf_variant="partial75",
        attention_config=FLOAT_CONFIG,
        verbose=False,
    )
    source_yolo.model = model
    float_path = tmp_path / "float.pt"
    source_yolo.save(str(float_path))

    bittrue_path = materialize_bittrue_checkpoint(float_path, BITTRUE_CONFIG, tmp_path / "bittrue.pt")
    reloaded = YOLO(str(bittrue_path)).model

    assert reloaded.model[16].p3_masf.context_channels == 64
    assert inspect_yolo26_graph(reloaded).end2end
    assert all(
        module.config.normalization.value == "bit_true_pwl"
        for module in reloaded.modules()
        if module.__class__.__name__ == "HardwareFriendlyAttention"
    )
