"""用真實 optimizer builder／scheduler 核對四分之一 LR，不跑 GPU。"""
import copy
from dataclasses import asdict

import pytest

from test_training_contract import ToyBase
from yolo_optimize.training import SCOPE, scaled_base_scope
from yolo_combine.stage_policy import build_joint_optimizer
from yolo_combine.joint_trainer import StageWarmupCosineScheduler


def test_quarter_lr_preserves_scope_and_entire_warmup_cosine_ratio():
    base = ToyBase()
    original = asdict(SCOPE)
    scaled = scaled_base_scope(.25)
    altered = asdict(scaled)
    assert {k: v for k, v in altered.items() if k != "learning_rates"} == {
        k: v for k, v in original.items() if k != "learning_rates"}
    assert scaled.learning_rates["neck"] == 2.5e-6
    assert scaled.learning_rates["pose_head"] == 6.25e-6
    pairs = []
    for scope in (SCOPE, scaled):
        model = copy.deepcopy(base)
        optimizer, _ = build_joint_optimizer(model, scope, optimizer_name="AdamW",
            weight_decay=.00027, beta1=.948, beta2=.999)
        scheduler = StageWarmupCosineScheduler(optimizer, stage="recovery", epochs=10,
            steps_per_epoch=463, warmup_epochs=1, warmup_start_factor=.1, final_lr_factor=.5)
        pairs.append((model, optimizer, scheduler))
    for step in range(4630):
        for _, _, scheduler in pairs:
            scheduler.prepare_step()
        for left, right in zip(pairs[0][1].param_groups, pairs[1][1].param_groups, strict=True):
            assert left["param_names"] == right["param_names"]
            assert right["lr"] == pytest.approx(left["lr"] * .25, rel=1e-13)
            assert right["weight_decay"] == left["weight_decay"]
        for _, _, scheduler in pairs:
            scheduler.advance()
    assert asdict(SCOPE) == original
    assert [p.requires_grad for p in pairs[0][0].parameters()] == [
        p.requires_grad for p in pairs[1][0].parameters()]
    for bad in (0, -1, .5, True, float("nan")):
        with pytest.raises(ValueError):
            scaled_base_scope(bad)
