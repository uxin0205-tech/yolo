"""驗證邊界的真實 CPU materialization regression；不訓練、不讀資料集。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn

from yolo_optimize import runtime
from yolo_optimize.training import SCOPE, TrainingModel, validation_boundary
from yolo_optimize.training_safety import FixedStateEMA
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.stage_policy import apply_stage
from ultralytics.nn.modules.conv import Conv


def test_actual_parent_validation_restores_shared_activation_without_source_changes():
    """走正式 EMA Float／BitTrue + live BitTrue 路徑，鎖住 singleton 實際錯誤。"""
    previous_threads = torch.get_num_threads()
    previous_default_training = Conv.default_act.training
    torch.set_num_threads(2)
    try:
        config = runtime.load_config(runtime.WORKSPACE / "artifacts/direction1-20260908")
        source, live, _factory, _loaded = runtime.load_model(
            config, runtime.FINAL_ROOT / "weights/combined/inference/best_joint.pt", "cpu")
        apply_stage(live, SCOPE)
        ema = FixedStateEMA(TrainingModel(live), decay=.9999, tau=2000).ema.base
        models = (live, ema)
        before = [{"state": {name: value.detach().clone() for name, value in model.state_dict().items()},
                   "training": {name: module.training for name, module in model.named_modules()},
                   "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()}}
                  for model in models]
        rng = torch.get_rng_state()
        with validation_boundary(*models) as audit:
            for model, kind in ((ema, "float"), (ema, "bittrue"), (live, "bittrue")):
                materialized = build_graph_validation_models(model, source, kind=kind)
                assert materialized.detect_report.complete and materialized.pose_report.complete
                assert live.pose_head.cv2[0][0].act.training is False
                del materialized
        assert torch.equal(torch.get_rng_state(), rng)
        for model, expected in zip(models, before, strict=True):
            assert {name: module.training for name, module in model.named_modules()} == expected["training"]
            assert {name: p.requires_grad for name, p in model.named_parameters()} == expected["requires_grad"]
            assert model.state_dict().keys() == expected["state"].keys()
            assert all(torch.equal(value, expected["state"][name]) for name, value in model.state_dict().items())
        restored = audit["restored_training_modes"]
        assert len(restored) == 1
        assert restored[0]["model_index"] == 0
        assert restored[0]["module_name"] == "graph.model.23.pose_head.cv2.0.0.act"
        assert restored[0]["before"] is True and restored[0]["observed"] is False
        assert restored[0]["restored"] is True
        assert len(restored[0]["all_alias_paths"]) == 24
    finally:
        Conv.default_act.training = previous_default_training
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("kind", ["other_mode", "subclass", "parameter", "buffer", "child"])
def test_validation_boundary_rejects_non_whitelisted_mode_changes(kind):
    class SpecializedSiLU(nn.SiLU):
        pass

    if kind == "other_mode":
        changed = nn.BatchNorm2d(1)
    elif kind == "subclass":
        changed = SpecializedSiLU()
    else:
        changed = nn.SiLU()
        if kind == "parameter":
            changed.register_parameter("extra", nn.Parameter(torch.ones(1)))
        elif kind == "buffer":
            changed.register_buffer("extra", torch.ones(1))
        else:
            changed.add_module("extra", nn.Identity())
    model = nn.Sequential(changed)
    with pytest.raises(RuntimeError, match="來源 state"):
        with validation_boundary(model) as audit:
            changed.eval()
    assert audit["restored_training_modes"] == []


@pytest.mark.parametrize("kind", ["tensor", "requires_grad"])
def test_allowed_activation_restore_never_hides_source_mutation(kind):
    model = nn.Sequential(nn.SiLU(), nn.Linear(1, 1))
    with pytest.raises(RuntimeError, match="來源 state"):
        with validation_boundary(model) as audit:
            model[0].eval()
            if kind == "tensor":
                with torch.no_grad():
                    model[1].weight.add_(1)
            else:
                model[1].weight.requires_grad_(False)
    assert model[0].training is True
    assert len(audit["restored_training_modes"]) == 1


def test_activation_restore_preserves_mixed_bn_modes_rng_and_original_exception():
    model = nn.ModuleDict({"act": nn.SiLU(), "shared_bn": nn.BatchNorm2d(1).eval(),
                           "head_bn": nn.BatchNorm2d(1).train()})
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match="original validation error"):
        with validation_boundary(model) as audit:
            model["act"].eval()
            torch.rand(3)
            raise ValueError("original validation error")
    assert model["act"].training is True
    assert model["shared_bn"].training is False
    assert model["head_bn"].training is True
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(audit["restored_training_modes"]) == 1
