import copy

import pytest
import torch
from binaryqk.model import SITES, bn_buffers, frozen_hashes, train_scope
from binaryqk.protocol import learning_rate, recipe_for, should_stop
from binaryqk.train import FrozenEMA, restore_resume, save_resume


def test_frozen_ema_and_optimizer_resume(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 1))
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("2."))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    ema = FrozenEMA(model)
    initial = copy.deepcopy(model.state_dict())
    frozen = frozen_hashes(model)
    for _ in range(4):
        optimizer.zero_grad()
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        ema.update(model)
    assert frozen_hashes(model) == frozen
    for name, value in ema.ema.state_dict().items():
        if not name.startswith("2."):
            assert torch.equal(value, initial[name])
    assert not torch.equal(ema.ema[2].weight, initial["2.weight"])
    state = copy.deepcopy(model.state_dict())
    ema_state = copy.deepcopy(ema.ema.state_dict())
    recipe = recipe_for("screening")
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    save_resume(tmp_path / "resume.pt", model, ema, optimizer, {"contract": 1}, 3, {"score": 1}, [],
                recipe=recipe, scaler=scaler)
    for p in model.parameters():
        with torch.no_grad():
            p.zero_()
    restore_resume(tmp_path / "resume.pt", model, ema, optimizer, {"contract": 1}, recipe=recipe, scaler=scaler)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
        assert torch.equal(ema.ema.state_dict()[name], ema_state[name])
    assert ema.updates == 4
    with pytest.raises(ValueError, match="manifest"):
        restore_resume(tmp_path / "resume.pt", model, ema, optimizer, {"contract": 2}, recipe=recipe, scaler=scaler)
    with pytest.raises(ValueError, match="recipe"):
        restore_resume(tmp_path / "resume.pt", model, ema, optimizer, {"contract": 1},
                       recipe=recipe_for("recovery"), scaler=scaler)
    with (tmp_path / "resume.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_resume(tmp_path / "resume.pt", model, ema, optimizer, {"contract": 1}, recipe=recipe, scaler=scaler)


def test_ema_tracks_bn_stats_without_changing_frozen_affine():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 1)).train()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("2."))
    frozen = frozen_hashes(model, bn_buffers(model))
    ema = FrozenEMA(model)
    mean = ema.ema[1].running_mean.clone()
    for _ in range(3):
        model(torch.ones(8, 3) * 4)
        ema.update(model)
    assert not torch.equal(ema.ema[1].running_mean, mean)
    assert ema.ema[1].num_batches_tracked.item() == 3
    assert torch.equal(ema.ema[1].weight, model[1].weight)
    assert frozen_hashes(model, bn_buffers(model)) == frozen


def test_recovery_horizon_warmup_and_patience_are_distinct():
    screening, recovery = recipe_for("screening"), recipe_for("recovery")
    assert screening["epochs"] == 10 and recovery["epochs"] == 40
    assert learning_rate(recovery, 0, 0) == 0
    assert 0 < learning_rate(recovery, 2, .5) < recovery["lr0"]
    assert learning_rate(recovery, 10, 1) > learning_rate(screening, 9, 1)
    assert not should_stop(screening, 9, 0)
    assert not should_stop(recovery, 9, 5)
    assert should_stop(recovery, 10, 5)


def test_attention_scope_and_full_recovery_keep_bn_adaptation():
    model = torch.nn.Module()
    model.add_module("backbone", torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)))
    for site in SITES:
        current = model
        pieces = site.split(".")
        for piece in pieces[:-1]:
            if not hasattr(current, piece):
                current.add_module(piece, torch.nn.Module())
            current = getattr(current, piece)
        current.add_module(pieces[-1], torch.nn.Linear(3, 3))
    train_scope(model, "attention")
    assert not model.backbone[0].weight.requires_grad
    assert model.backbone[1].training
    assert all(model.get_submodule(s).weight.requires_grad for s in SITES)
    train_scope(model, "full")
    assert all(p.requires_grad for p in model.parameters())
