"""驗證重建邊界，避免重演不相容 trunk、終點 LR cap 或錯誤 EMA。"""

import copy

import pytest
import torch
from binaryqk.train import FrozenEMA
from stage6 import checkpoint
from stage6.common import review
from stage6.evaluate import accepted, expected
from stage6.models import build, configure, manifest_for, mix, role, view
from stage6.ops import optimizer_for
from stage6.policy import Plateau, recipe_for

from yolo_attention import stage3_checkpoint


@pytest.fixture(scope="module")
def anchor():
    torch.set_num_threads(2)
    return build()


def test_initialization_keeps_entire_detect_and_pose_head(anchor):
    from stage6.models import native_pose

    detect, _ = stage3_checkpoint.load(review()["detect_parent"])
    for n, value in detect.state_dict().items():
        assert torch.equal(value, view(anchor, "detect").state_dict()[n]), n
    for n, value in native_pose().model[-1].state_dict().items():
        assert torch.equal(value, anchor.pose_head.state_dict()[n]), n


@pytest.mark.parametrize("phase", ["H0", "P", "J1", "J2", "J3"])
def test_scope_bn_and_fresh_ema_and_lr(anchor, phase):
    model = copy.deepcopy(anchor)
    recipe = recipe_for(phase)
    mutable = configure(model, recipe)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    roles = {role(n) for n in active}
    assert "fixed" not in roles
    assert roles == set(recipe["lrs"])
    for n, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            is_active = ".pose_head." in n or (
                recipe["task"] == "joint" and ".detect_head." in n
            )
            assert module.training == is_active
            if not is_active:
                assert not any(p.requires_grad for p in module.parameters())
    assert all("_head." in n for n in mutable)
    if phase == "J2":
        assert all(int(n.split(".")[2]) >= 9 for n in active if role(n) == "backbone")
    ema = FrozenEMA(model, mutable_buffers=mutable)
    assert ema.updates == 0
    optimizer = optimizer_for(model, recipe)
    for group in optimizer.param_groups:
        assert group["lr"] == recipe["lrs"][group["group_name"].rsplit("-", 1)[0]]


def test_reject_unadapted_mix_and_mix_real_nonzero_weights(anchor):
    branch = copy.deepcopy(anchor)
    with pytest.raises(ValueError, match="completed compatible"):
        mix(anchor, branch, 0.1, parent={})
    branch.stage6_manifest["training_lineage"] = [{"recipe": recipe_for("P")}]
    names = dict(branch.named_parameters())
    shared = next(n for n in names if role(n) == "neck" and names[n].ndim == 4)
    head = next(n for n in names if role(n) == "pose_head" and names[n].ndim == 4)
    with torch.no_grad():
        names[shared].add_(0.02)
        names[head].add_(0.03)
    original = anchor.state_dict()[shared].clone()
    for rho in (0.0, 0.1, 0.25):
        result = mix(anchor, branch, rho, parent={"test": True})
        torch.testing.assert_close(
            result.state_dict()[shared],
            original + rho * (names[shared] - original),
            rtol=0,
            atol=0,
        )
        assert torch.equal(result.state_dict()[head], names[head])
        assert torch.equal(anchor.state_dict()[shared], original)
    with torch.no_grad():
        next(p for n, p in branch.named_parameters() if ".attn." in n).add_(0.01)
    with pytest.raises(ValueError, match="Attention"):
        mix(anchor, branch, 0.1, parent={})


def test_checkpoint_roundtrip_and_required_contract(anchor, tmp_path):
    path = tmp_path / "selected.pt"
    checkpoint.save(path, anchor)
    reloaded = checkpoint.load(path)
    assert manifest_for(reloaded) == manifest_for(anchor)
    assert all(
        torch.equal(v, reloaded.state_dict()[k]) for k, v in anchor.state_dict().items()
    )
    bad = copy.deepcopy(reloaded)
    bad.stage6_manifest["upstream"].pop("candidate")
    with pytest.raises((KeyError, ValueError)):
        manifest_for(bad)


def test_final_gate_all_classes_and_plateau_resume():
    base = expected()
    assert accepted(base)
    for key in base:
        bad = dict(base)
        bad[key] -= 0.1
        assert not accepted(bad), key
    policy = Plateau(recipe_for("J2"))
    policy.observe(0.7)
    for _ in range(8):
        assert not policy.observe(0.7)
    assert policy.multiplier == 0.5
    reload = Plateau(recipe_for("J2"))
    reload.load_state_dict(policy.state_dict())
    for _ in range(8):
        assert not reload.observe(0.7)
    assert reload.observe(0.7)
    assert recipe_for("J2")["lrs"]["pose_head"] == 2e-5
    assert recipe_for("J1", 1)["lrs"]["pose_head"] == 1e-5


@pytest.mark.parametrize("phase", ["H0", "H1", "M0"])
def test_head_adaptation_does_not_use_full_pose_regression_gate(phase):
    from stage6.policy import safety

    parent = expected()
    current = dict(parent)
    current["ball_box_AP50_95"] -= 0.03047312159085155
    assert safety(current, parent, recipe_for(phase), epoch=3) is None
    # A changed frozen Detect branch must still stop immediately.
    current["coco_AP"] -= 0.000001
    assert safety(current, parent, recipe_for(phase), epoch=3) is not None


def test_full_pose_and_joint_keep_their_regression_gates():
    from stage6.policy import safety

    parent = expected()
    current = dict(parent)
    current["ball_box_AP50_95"] -= 0.03047312159085155
    assert safety(current, parent, recipe_for("P"), epoch=3) is not None
    assert safety(current, parent, recipe_for("J1"), epoch=3) is None
    current["ball_box_AP50_95"] = parent["ball_box_AP50_95"] - 0.081
    assert safety(current, parent, recipe_for("J1"), epoch=3) is not None
    # Final acceptance never uses the looser training thresholds.
    assert not accepted(current)


@pytest.mark.parametrize("progress", [{}, {"time_unix": 100.0}, {"time_unix": 10001.0}])
def test_restart_monitor_ignores_previous_worker_timestamp(progress):
    from stage6.runner import last_activity_time

    started, now = 10000.0, 10002.0
    assert now - last_activity_time(progress, started) <= 2.0
    assert 12000.0 - last_activity_time(progress, started) > 1800.0


def test_monitor_uses_current_worker_heartbeat():
    from stage6.runner import last_activity_time

    assert last_activity_time({"time_unix": 11900.0}, 10000.0) == 11900.0


def test_late_pose_head_recovery_preserves_coco_and_checks_bbat():
    from stage6.policy import safety

    recipe = {
        **recipe_for("H1"),
        "selection": "pose_recovery",
        "head_recovery_max_pose_drop": 0.02,
    }
    parent = expected()
    parent["baseball_bat_AP"] -= 0.017
    improved = {**parent, "ball_box_AP50_95": parent["ball_box_AP50_95"] + 0.001}
    assert safety(improved, parent, recipe, 1) is None
    changed = {**improved, "baseball_bat_AP": parent["baseball_bat_AP"] + 0.000001}
    assert safety(changed, parent, recipe, 1) == "固定共享層與 Detect 的完整 AP 改變"
    worse = {**parent, "ball_pose_AP50_95": parent["ball_pose_AP50_95"] - 0.021}
    assert safety(worse, parent, recipe, 1) == "Pose head 恢復較起點退化超過安全線"


def test_pose_recovery_failed_smoke_blocks_training(tmp_path):
    from stage6.recovery import run_recovery

    calls = []

    def job(name, action, **kwargs):
        calls.append(action)
        return {"status": "needs_review"}

    protocol = {
        "kind": "stage6-pose-head-recovery-v1",
        "parent": {},
        "recipe": recipe_for("H1"),
    }
    with pytest.raises(ValueError, match="smoke requires review"):
        run_recovery(job, tmp_path, protocol)
    assert calls == ["smoke-head-recovery"]
