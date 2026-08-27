from __future__ import annotations

import copy

import pytest
from torch import nn

from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.source import BuiltTaskModels
from yolo_combine.stage_policy import (
    JOINT_STAGES,
    apply_stage,
    build_joint_optimizer,
)
from yolo_combine.xnor import XNORExecutionConfig, install_xnor_backend


@pytest.fixture(scope="module")
def graph_shared(full35_models: BuiltTaskModels):
    install_xnor_backend(XNORExecutionConfig(token_tile=32))
    model, _ = assemble_graph_shared_model(
        copy.deepcopy(full35_models.detect),
        copy.deepcopy(full35_models.pose),
    )
    return model


@pytest.mark.integration
def test_j0_j1_and_j2_apply_the_accepted_trainable_scope(graph_shared) -> None:
    j0 = apply_stage(graph_shared, JOINT_STAGES["j0"])
    trainable_j0 = set(j0.trainable_names)
    assert trainable_j0
    assert all(".pose_head." in name for name in trainable_j0)

    j1 = apply_stage(graph_shared, JOINT_STAGES["j1"])
    trainable_j1 = set(j1.trainable_names)

    assert any(".detect_head." in name for name in trainable_j1)
    assert any(".pose_head." in name for name in trainable_j1)
    assert any("graph.model.13." in name for name in trainable_j1)
    assert not any(".p3_masf." in name for name in trainable_j1)
    assert not any("graph.model.10." in name for name in trainable_j1)
    assert not any(".attn." in name for name in trainable_j1)

    j2 = apply_stage(graph_shared, JOINT_STAGES["j2"])
    trainable_j2 = set(j2.trainable_names)
    assert any("graph.model.9." in name for name in trainable_j2)
    assert not any("graph.model.8." in name for name in trainable_j2)
    assert any(".p3_masf." in name for name in trainable_j2)
    assert not any(".attn.qkv.v." in name for name in trainable_j2)
    assert not any(".attn.pe." in name for name in trainable_j2)
    assert not any(".attn.proj." in name for name in trainable_j2)
    assert not any(".attn.bias." in name for name in trainable_j2)
    assert not any(".attn.qkv.q." in name for name in trainable_j2)
    assert not any(".attn.qkv.k." in name for name in trainable_j2)
    assert not any(".attn.score.gamma" in name for name in trainable_j2)

    j3 = apply_stage(graph_shared, JOINT_STAGES["j3"])
    trainable_j3 = set(j3.trainable_names)
    assert any(".attn.qkv.v." in name for name in trainable_j3)
    assert any(".attn.pe." in name for name in trainable_j3)
    assert any(".attn.proj." in name for name in trainable_j3)
    assert any(".attn.bias." in name for name in trainable_j3)
    assert not any(".attn.qkv.q." in name for name in trainable_j3)
    assert not any(".attn.qkv.k." in name for name in trainable_j3)
    assert not any(".attn.score.gamma" in name for name in trainable_j3)


@pytest.mark.integration
def test_shared_bn_stats_stay_eval_while_head_bn_trains(graph_shared) -> None:
    graph_shared.train()
    report = apply_stage(graph_shared, JOINT_STAGES["j2"])

    shared_bn = [
        module
        for name, module in graph_shared.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and ".detect_head." not in name
        and ".pose_head." not in name
    ]
    head_bn = [
        module
        for name, module in graph_shared.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and (".detect_head." in name or ".pose_head." in name)
    ]
    assert shared_bn and head_bn
    assert all(not module.training for module in shared_bn)
    assert all(module.training for module in head_bn)
    assert report.shared_bn_frozen == len(shared_bn)
    assert report.head_bn_training == len(head_bn)


@pytest.mark.integration
def test_j0_keeps_detect_head_bn_eval_and_pose_head_bn_training(graph_shared) -> None:
    report = apply_stage(graph_shared, JOINT_STAGES["j0"])
    detect_bn = [
        module
        for name, module in graph_shared.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and ".detect_head." in name
    ]
    pose_bn = [
        module
        for name, module in graph_shared.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and ".pose_head." in name
    ]
    assert detect_bn and pose_bn
    assert all(not module.training for module in detect_bn)
    assert all(module.training for module in pose_bn)
    assert report.head_bn_training == len(pose_bn)


@pytest.mark.integration
def test_optimizer_groups_are_unique_and_keep_explicit_weight_decay(graph_shared) -> None:
    optimizer, report = build_joint_optimizer(
        graph_shared,
        JOINT_STAGES["j1"],
        optimizer_name="AdamW",
        weight_decay=0.00027,
        beta1=0.948,
    )

    grouped = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert report.duplicate_parameter_names == ()
    assert set(report.semantic_roles) == {
        "backbone",
        "neck",
        "masf",
        "attention",
        "detect_head",
        "pose_head",
    }
    assert {group["group_name"] for group in optimizer.param_groups} == set(
        report.group_names
    )
    assert {
        group["weight_decay"] for group in optimizer.param_groups
    } == {0.0, 0.00027}
    assert all(
        group["lr"] == JOINT_STAGES["j1"].learning_rates[group["role"]]
        for group in optimizer.param_groups
    )
