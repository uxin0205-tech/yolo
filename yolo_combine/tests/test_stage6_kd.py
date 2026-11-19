"""KD 梯度、macro 可加性與恢復選型不掩蓋原精度缺口。"""

import pytest
import torch
from stage6.evaluate import accepted, expected
from stage6.kd import SITES, capture, spatial_loss
from stage6.policy import rank, safety


def test_spatial_gradient_detach_and_microbatch_additivity():
    torch.manual_seed(23)
    student = [torch.randn(4, 3, h, h, requires_grad=True) for h in (8, 4, 2)]
    teacher = [torch.randn(4, 5, h, h, requires_grad=True) for h in (8, 4, 2)]
    full = spatial_loss(student, teacher).sum()
    split = sum(
        spatial_loss(
            [s[i : i + 1] for s in student], [t[i : i + 1] for t in teacher]
        ).sum()
        for i in range(4)
    )
    torch.testing.assert_close(full, split)
    fg = torch.autograd.grad(full, student, retain_graph=True)
    sg = torch.autograd.grad(split, student)
    for a, b in zip(fg, sg):
        assert a.abs().sum() > 0
        torch.testing.assert_close(a, b)
    full.backward()
    assert all(t.grad is None for t in teacher)


def test_spatial_zero_channel_replication_and_misalignment():
    z = [torch.zeros(2, 3, 4, 5) for _ in range(3)]
    assert torch.equal(spatial_loss(z, z), torch.zeros(2))
    x = [torch.randn(2, 1, 4, 5) for _ in range(3)]
    torch.testing.assert_close(
        spatial_loss(x, [a.repeat(1, 4, 1, 1) for a in x]),
        torch.zeros(2),
        atol=1e-12,
        rtol=0,
    )
    with pytest.raises(ValueError, match="coordinates"):
        spatial_loss(x, [a.transpose(2, 3) for a in x])


def test_temporary_feature_hooks_clean_up_on_error():
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(23)])
    with pytest.raises(RuntimeError, match="repeated"), capture(layers):
        layers[16](torch.ones(1))
        layers[16](torch.ones(1))
    assert all(not layers[i]._forward_hooks for i in SITES)


def test_recovery_selection_reports_deficit_without_rejecting_known_parent():
    recipe = {
        "scope": "joint_back9",
        "task": "joint",
        "selection": "balanced_recovery",
        "recovery_safety": {
            "coco_from_parent": 0.01,
            "coco_from_original": 0.05,
            "pose_from_parent": 0.03,
        },
    }
    parent = expected()
    parent["baseball_bat_AP"] -= 0.025
    parent["pose_AP50_95"] -= 0.025
    improved = {**parent, "baseball_bat_AP": parent["baseball_bat_AP"] + 0.003}
    assert not accepted(improved)
    assert rank(improved, recipe) > rank(parent, recipe)
    assert safety(improved, parent, recipe, 5) is None
    worse = {**parent, "baseball_bat_AP": parent["baseball_bat_AP"] - 0.011}
    assert safety(worse, parent, recipe, 1)
    assert rank(expected(), recipe) > rank(improved, recipe)
