import copy

import pytest
import torch

from yolo_attention.config import Stage3KDConfig
from yolo_attention.stage3_kd_loss import capture, protected_mask, spatial_loss
from yolo_attention.stage3_kd_training import eligible, metric_vector


def batch():
    return {'cls': torch.tensor([[34.], [32.]]), 'batch_idx': torch.tensor([0, 1]),
            'bboxes': torch.tensor([[.5, .5, .01, .01], [.5, .5, 1., 1.]])}


def test_small_bat_box_protected_but_ball_and_other_images_retained():
    mask = protected_mask(batch(), (2, 1, 8, 8), Stage3KDConfig())
    assert not mask[0, 3:5, 3:5].any()
    assert mask[0, 0, 0] and mask[1].all()


def test_masked_teacher_difference_does_not_create_kd_gradient():
    s = torch.ones(2, 3, 8, 8, requires_grad=True)
    t = s.detach().clone().requires_grad_()
    with torch.no_grad():
        t[0, :, 3:5, 3:5] = 100
    loss = spatial_loss([s]*3, [t]*3, batch(), Stage3KDConfig()).sum()
    assert loss == 0
    loss.backward()
    assert s.grad.abs().sum() == 0 and t.grad is None


def test_kd_microbatch_sum_and_gradient_equal_whole_batch():
    s = torch.rand(2, 4, 8, 8, requires_grad=True)
    t = torch.rand(2, 9, 8, 8, requires_grad=True)
    b = batch()
    whole = spatial_loss([s]*3, [t]*3, b, Stage3KDConfig()).sum()
    parts = []
    for i in range(2):
        keep = b['batch_idx'] == i
        small = {k: v[keep].clone() for k, v in b.items()}
        small['batch_idx'].zero_()
        parts.append(spatial_loss([s[i:i+1]]*3, [t[i:i+1]]*3, small, Stage3KDConfig()).sum())
    assert torch.allclose(whole, sum(parts))
    assert torch.allclose(torch.autograd.grad(whole, s, retain_graph=True)[0],
                          torch.autograd.grad(sum(parts), s)[0])
    assert t.grad is None


def test_capture_removes_hooks_on_error():
    layers = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(23)])
    with pytest.raises(RuntimeError), capture(layers):
        raise RuntimeError('probe')
    assert all(not m._forward_hooks for m in layers)


def test_zero_features_and_fully_masked_images_are_finite():
    b = {'cls': torch.tensor([34]), 'batch_idx': torch.tensor([0]),
         'bboxes': torch.tensor([[.5, .5, 1., 1.]])}
    s = torch.zeros(1, 4, 2, 2, requires_grad=True)
    loss = spatial_loss([s]*3, [torch.ones_like(s)]*3, b, Stage3KDConfig()).sum()
    loss.backward()
    assert loss == 0 and torch.isfinite(s.grad).all()


def test_selection_rejects_bat_regression_even_with_overall_gain():
    base = {'official': {'AP50': .6, 'AP50_95': .5, 'classes': {
        k: {'AP50': .6, 'AP50_95': .5} for k in ('person', 'sports_ball', 'baseball_bat')}}}
    better = copy.deepcopy(base)
    better['official']['AP50_95'] += .002
    assert len(metric_vector(base)) == 8 and eligible(better, base)
    better['official']['classes']['baseball_bat']['AP50_95'] -= .002
    assert not eligible(better, base)
    assert not eligible(base, base)


def test_kd_checkpoint_keeps_integer_forward_and_provenance(tmp_path):
    from yolo_attention.stage3_checkpoint import build, load, save
    from yolo_attention.stage3_kd_training import attach_provenance

    model = build('D1')
    attach_provenance(model, {'mu': .1}, 'a'*64, 'b'*64)
    save(tmp_path/'kd.pt', model)
    restored, manifest = load(tmp_path/'kd.pt')
    assert manifest['lineage']['phase'] == 'kd_recovery'
    assert manifest['adaptation']['calibration_sha256'] == 'a'*64
    assert manifest['normalization']['complete_fixed_normalization']
    assert not any('teacher' in n for n in restored.state_dict())
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        assert torch.equal(model(x)[0], restored(x)[0])
    del restored.stage3_adaptation
    with pytest.raises(ValueError):
        save(tmp_path/'missing.pt', restored)


def test_shared_macro_engine_applies_combined_loss_once_per_microbatch():
    from binaryqk.train import macro_step
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    calls = []

    def loss_fn(student, item):
        calls.append(True)
        native = student.weight.square().sum()
        kd = 3*student.weight.sum()
        return native+kd, None

    result = macro_step(model, [{'img': torch.ones(1, 1)}]*2, optimizer,
                        torch.amp.GradScaler('cpu', enabled=False), list(model.parameters()),
                        {'reference_batch': 2, 'overflow_attempts': 1, 'amp': False, 'gradient_clip': 100},
                        ['weight'], loss_fn)
    assert len(calls) == 2 and result['loss_macro_sum'] == 20
    assert torch.allclose(model.weight, torch.tensor([[.6]]))
