from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from achitechure_1.profiling import configure_official_loss

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "artifacts" / "prepared" / "a1-full35.pt"


def test_configure_official_loss_restores_checkpoint_training_args() -> None:
    model = YOLO(str(PREPARED)).model

    assert isinstance(model.args, dict)
    assert "box" not in model.args

    configure_official_loss(model)
    criterion = model.init_criterion()

    assert criterion.one2many.hyp.box == DEFAULT_CFG_DICT["box"]
    assert criterion.one2many.hyp.cls == DEFAULT_CFG_DICT["cls"]
    assert criterion.one2many.hyp.dfl == DEFAULT_CFG_DICT["dfl"]
