from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from ultralytics import YOLO

from yolo_attention.optimizer_scope import apply_layerwise_learning_rates, restrict_optimizer_to_trainable
from yolo_attention.recovery_workflow import RECOVERY_STAGES, create_pwl_recovery_state
from yolo_attention.run_config import TrainingRecipe
from yolo_attention.scopes import apply_trainable_scope, enforce_frozen_batchnorm, learning_rate_group

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt"
RATES = {
    "attention": 5e-6,
    "adjacent_block": 1e-6,
    "neck_detect": 5e-7,
    "backbone": 1e-7,
}


@pytest.fixture(scope="module")
def recovery_model() -> nn.Module:
    return YOLO(str(PARENT)).model


def _groups(summary) -> set[str]:
    return {learning_rate_group(name) for name in summary.trainable_names}


def test_recovery_scopes_are_nested_on_real_yolo26m(recovery_model: nn.Module) -> None:
    block = apply_trainable_scope(recovery_model, "block_recovery")
    assert _groups(block) == {"attention", "adjacent_block"}
    assert not any(name.startswith("model.11.") for name in block.trainable_names)

    neck = apply_trainable_scope(recovery_model, "neck_recovery")
    assert _groups(neck) == {"attention", "adjacent_block", "neck_detect"}
    assert any(name.startswith("model.23.") for name in neck.trainable_names)
    assert not any(name.startswith("model.7.") for name in neck.trainable_names)

    backbone = apply_trainable_scope(recovery_model, "backbone_last_recovery")
    assert _groups(backbone) == set(RATES)
    assert any(name.startswith("model.7.") for name in backbone.trainable_names)
    assert not any(name.startswith("model.6.") for name in backbone.trainable_names)

    full = apply_trainable_scope(recovery_model, "full_model_recovery")
    assert _groups(full) == set(RATES)
    assert any(name.startswith("model.0.") for name in full.trainable_names)
    assert not any(name.endswith("score.gamma") or ".normalize." in name for name in full.trainable_names)


def test_recovery_bn_and_optimizer_lr_groups(recovery_model: nn.Module) -> None:
    recovery_model.train()
    apply_trainable_scope(recovery_model, "backbone_last_recovery")
    enforce_frozen_batchnorm(recovery_model, "backbone_last_recovery")
    batchnorms = [
        module for module in recovery_model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batchnorms and all(not module.training for module in batchnorms)

    optimizer = torch.optim.AdamW(recovery_model.parameters(), lr=5e-6)
    restrict_optimizer_to_trainable(optimizer, recovery_model)
    counts = apply_layerwise_learning_rates(optimizer, recovery_model, RATES)
    assert set(counts) == set(RATES) and all(count > 0 for count in counts.values())
    for group in optimizer.param_groups:
        assert group["lr"] == RATES[group["layer_group"]]
    expected = {id(parameter) for parameter in recovery_model.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert actual == expected


def test_recovery_queue_and_recipes() -> None:
    state = create_pwl_recovery_state(ROOT)
    assert len(state.jobs) == 15
    assert state.job("recovery-parent-bittrue").status.value == "ready"
    assert state.job("export-recovery").parent_job_ids == ("recovery-select",)
    assert state.job("recovery-neck").model_parent_job_id == "recovery-block-gate"
    for stage in RECOVERY_STAGES:
        recipe = TrainingRecipe.from_yaml(ROOT / "configs/training" / f"recovery-{stage}.yaml")
        assert recipe.layer_lrs is not None
        assert recipe.layer_lrs["attention"] == 5e-6
        assert recipe.warmup_epochs == recipe.warmup_bias_lr == 0
    assert TrainingRecipe.from_yaml(ROOT / "configs/training/recovery-neck.yaml").layer_lrs == {
        "attention": 5e-6,
        "adjacent_block": 1e-6,
        "neck_detect": 5e-7,
    }
