from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.optim import MuSGD

from achitechure_1.model import graft_p3_masf
from achitechure_1.phases import (
    PHASES,
    apply_phase_scope,
    assert_frozen_state_unchanged,
    build_phase_optimizer,
    enforce_frozen_modules_eval,
    snapshot_frozen_state,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "inputs" / "parent" / "best.pt"


def _model():
    model = YOLO(str(PARENT)).model
    graft_p3_masf(model, "full35")
    return model


def test_fixed_phase_contract_matches_the_experiment_spec() -> None:
    assert PHASES["a1"].epochs == 5
    assert PHASES["a1"].learning_rates == {"masf": 1.0e-3}
    assert PHASES["a2"].epochs == 10 and PHASES["a2"].lrf == 0.5
    assert PHASES["b"].epochs == 10
    assert PHASES["c"].epochs == 55 and PHASES["c"].patience == 5


def test_phase_scopes_freeze_attention_until_full_model_finetuning() -> None:
    model = _model()

    a1 = apply_phase_scope(model, "a1")
    assert a1.trainable_names
    assert all(".p3_masf." in name for name in a1.trainable_names)

    phase_b = apply_phase_scope(model, "b")
    assert any(name.startswith("model.23.") for name in phase_b.trainable_names)
    assert any(".p3_masf." in name for name in phase_b.trainable_names)
    assert not any(name.startswith("model.10.m.0.attn.") for name in phase_b.trainable_names)
    assert not any(name.startswith("model.22.m.0.1.attn.") for name in phase_b.trainable_names)

    phase_c = apply_phase_scope(model, "c")
    assert len(phase_c.trainable_names) == len(tuple(model.named_parameters()))


def test_musgd_groups_use_discriminative_lrs_and_no_decay_for_alpha() -> None:
    model = _model()
    apply_phase_scope(model, "c")

    optimizer = build_phase_optimizer(model, PHASES["c"])

    assert isinstance(optimizer, MuSGD)
    assert {group["role"] for group in optimizer.param_groups} == {
        "masf",
        "neck_detect",
        "backbone",
        "attention",
    }
    actual_lrs = {group["role"]: group["lr"] for group in optimizer.param_groups}
    assert actual_lrs == PHASES["c"].learning_rates
    alpha = model.model[16].p3_masf.alpha
    alpha_groups = [group for group in optimizer.param_groups if any(p is alpha for p in group["params"])]
    assert len(alpha_groups) == 1 and alpha_groups[0]["weight_decay"] == 0.0


def test_phase_a_optimizer_step_cannot_change_frozen_state_or_bn_buffers() -> None:
    model = _model().train()
    apply_phase_scope(model, "a1")
    enforce_frozen_modules_eval(model)
    before = snapshot_frozen_state(model)
    optimizer = build_phase_optimizer(model, PHASES["a1"])
    sample = torch.randn(1, 3, 64, 64)

    prediction = model(sample)["one2many"]
    loss = prediction["boxes"].float().mean() + prediction["scores"].float().mean()
    loss = loss + sum(tensor.float().mean() for tensor in prediction["feats"])
    loss.backward()
    optimizer.step()

    assert_frozen_state_unchanged(model, before)
