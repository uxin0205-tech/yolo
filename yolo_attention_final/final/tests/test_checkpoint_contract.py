from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.config import NormalizationKind
from yolo_attention.integration import YOLO26M_ATTENTION_PATHS


def test_delivered_checkpoint_is_yolo26m_bittrue_and_reloadable() -> None:
    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / "pwl-final-best.pt")).model
    assert model.yaml["yaml_file"] == "yolo26m.yaml"
    assert model.yaml["scale"] == "m"
    assert model.yaml["nc"] == 80
    sites = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, HardwareFriendlyAttention)
    }
    assert set(sites) == set(YOLO26M_ATTENTION_PATHS)
    assert all(module.config.normalization is NormalizationKind.BIT_TRUE_PWL for module in sites.values())
    assert all(module.normalize.endpoint_table.numel() == 21 for module in sites.values())
    assert all(not module.score.use_ste for module in sites.values())
