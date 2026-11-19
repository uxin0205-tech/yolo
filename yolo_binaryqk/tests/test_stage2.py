import copy
import json

import pytest
import torch
from binaryqk import checkpoint, scale_checkpoint
from binaryqk.model import ROOT, SITES, frozen_hashes, sha256
from binaryqk.scale_study import (
    ScaleStudy,
    configure,
    optimizer,
    recipe,
    verify_reference,
)
from binaryqk.train import FrozenEMA, restore_resume, save_resume


def build(candidate):
    calibration = json.loads((ROOT / 'yolo_binaryqk/artifacts/stage1/calibration.json').read_text())
    return ScaleStudy.build(candidate, 'scale_bias', None, calibration)


@pytest.mark.parametrize('candidate', ['S1', 'S2'])
def test_actual_b2_parent_roundtrip_and_deployment_constants(candidate, tmp_path):
    model = build(candidate).eval()
    review = verify_reference()
    parent, _ = checkpoint.load(review['parent'])
    for name, value in parent.state_dict().items():
        if '.score.' not in name:
            assert torch.equal(value, model.state_dict()[name]), name
    for site in SITES:
        affine = model.get_submodule(site).score.affine
        assert torch.count_nonzero(affine.table_x) == torch.count_nonzero(affine.table_y) == 0
    initial = scale_checkpoint.manifest_for(model, sha256(ROOT / 'yolo_binaryqk/artifacts/stage1/calibration.json'))
    with torch.no_grad():
        for site in SITES:
            affine = model.get_submodule(site).score.affine
            affine.coefficients.add_(.01)
            affine.table_x[:, 31] = .0014
            affine.table_y[:, 32] = -.0026
    model.eval()
    scale_checkpoint.save(tmp_path / 'model.pt', model, initial)
    restored, manifest = scale_checkpoint.load(tmp_path / 'model.pt')
    assert manifest['constants'] != initial['constants']
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        assert torch.equal(model(x)[0], restored(x)[0])
    bad = copy.deepcopy(manifest)
    bad['normalization'] = 'float-fallback'
    with pytest.raises(ValueError):
        scale_checkpoint.validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad['constants'][SITES[0]]['bias_x_codes'][0][0] = 32768
    with pytest.raises(ValueError, match='bias_x_codes'):
        scale_checkpoint.validate_manifest(bad)
    with pytest.raises(ValueError):
        checkpoint.load(tmp_path / 'model.pt')


def test_scope_fixed_bn_ema_and_resume(tmp_path):
    model = build('S1')
    cfg = recipe()
    configure(model, cfg['scope'])
    assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert not model.model[0].conv.weight.requires_grad
    assert all(model.get_parameter(n).requires_grad for n in ScaleStudy.gradient_names(model))
    fixed = frozen_hashes(model)
    ema = FrozenEMA(model, set())
    opt = optimizer(model, cfg)
    assert {g['role'] for g in opt.param_groups} == {'scale', 'bias', 'head', 'attention'}
    assert all(g['weight_decay'] == 0 for g in opt.param_groups if g['role'] in {'scale', 'bias'})
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    opt.step()
    ema.update(model)
    assert frozen_hashes(model) == fixed
    for name in fixed:
        assert torch.equal(ema.ema.state_dict()[name], model.state_dict()[name]), name
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    manifest = {'phase': 'scale_bias', 'source': 'test'}
    save_resume(tmp_path / 'resume.pt', model, ema, opt, manifest, 2, {}, [], recipe=cfg, scaler=scaler)
    expected = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.get_parameter(ScaleStudy.gradient_names(model)[2]).zero_()
    assert restore_resume(tmp_path / 'resume.pt', model, ema, opt, manifest, recipe=cfg, scaler=scaler)[0] == 2
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())


def test_scheduler_matches_optimize_source(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'yolo_combine/src'))
    from yolo_combine.joint_trainer import StageWarmupCosineScheduler
    cfg = recipe()
    p = torch.nn.Parameter(torch.zeros(1))
    source_opt = torch.optim.AdamW([{'params': [p], 'lr': 2e-4, 'group_name': 'scale/no_decay'}])
    own_opt = torch.optim.AdamW([{'params': [p], 'lr': 2e-4, 'group_name': 'scale/no_decay', 'role': 'scale'}])
    scheduler = StageWarmupCosineScheduler(source_opt, stage='scale_bias', epochs=5, steps_per_epoch=463,
                                         warmup_epochs=1, warmup_start_factor=.1, final_lr_factor=.5)
    for step in [0, 1, 462, 463, 926, 2314]:
        scheduler.current_step = step
        expected = scheduler.prepare_step()
        actual = ScaleStudy.set_lr(own_opt, cfg, step // 463, 0, step % 463, False)
        assert actual == expected


def test_missing_shape_and_mixed_candidates_fail_closed():
    score = scale_checkpoint.AffineBranchScore('S1', torch.ones(4, 2) * .1)
    with pytest.raises(ValueError, match='explicit'):
        score(torch.ones(1, 4, 32, 6), torch.ones(1, 4, 32, 6))
    model = build('S1')
    model.get_submodule(SITES[1]).score.candidate = 'S2'
    with pytest.raises(ValueError):
        scale_checkpoint.manifest_for(model, 'a' * 64)


def test_stage2_plot_uses_small_json_only(tmp_path):
    from binaryqk.scale_report import plot
    (tmp_path / 'comparison.json').write_text(json.dumps({
        'parent_AP50_95': .51,
        'curves': [{'candidate': 'S1', 'epochs': [0, 1], 'official_AP50_95': [.50, .51]},
                   {'candidate': 'S2', 'epochs': [0, 1], 'official_AP50_95': [.49, .50]}]}))
    plot(tmp_path)
    assert (tmp_path / 'curves.svg').stat().st_size > 1000
