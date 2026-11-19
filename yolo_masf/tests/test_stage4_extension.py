import copy

import pytest
import torch
from binaryqk.data import config
from binaryqk.train import FrozenEMA, _same, restore_resume, save_resume
from masf_yolo.stage4 import checkpoint
from masf_yolo.stage4.extension_training import ContinuationStudy


@pytest.mark.parametrize('task', ['detect', 'pose'])
@pytest.mark.parametrize('arm', ['L1', 'L2'])
def test_bootstrap_preserves_live_ema_moments_and_loss_clock(task, arm, tmp_path):
    torch.set_num_threads(2)
    policy = ContinuationStudy(task, arm)
    model = policy.build(arm, 'masf-extension', None, None)
    recipe = policy.recipe('masf-extension')
    model.args = config(16)
    model.args.epochs = recipe['epochs']
    model.args.task = task
    policy.configure(model, recipe['scope'])
    ema = FrozenEMA(model, set())
    optimizer = policy.optimizer(model, recipe)
    scaler = torch.amp.GradScaler('cpu', enabled=True)
    policy.initialize_state(model, ema, optimizer, scaler, recipe)
    payload = policy.payload()
    assert _same(model.state_dict(), payload['live_state'])
    assert _same(ema.ema.state_dict(), payload['ema_state'])
    assert _same(optimizer.state_dict()['state'], payload['optimizer']['state'])
    assert scaler.state_dict() == payload['scaler']
    assert ema.updates == payload['ema_updates']
    for group in optimizer.param_groups:
        assert group['lr'] == recipe['group_lrs'][group['role']]
    policy.initialize_criterion(model, 0)
    assert model.criterion.updates == 5
    for _ in range(16):
        assert model.criterion.o2m == pytest.approx(.1)
        assert model.criterion.o2o == pytest.approx(.9)
        model.criterion.update()
    manifest = checkpoint.manifest_for(model)
    path = tmp_path / 'resume.pt'
    save_resume(path, model, ema, optimizer, manifest, 3, {'epoch': 2}, [], recipe=recipe, scaler=scaler)
    restored = copy.deepcopy(optimizer.state_dict())
    optimizer.param_groups[0]['lr'] = 99
    age, _, _ = restore_resume(path, model, ema, optimizer, manifest, recipe=recipe, scaler=scaler)
    assert age == 3 and _same(restored, optimizer.state_dict())
    policy.initialize_criterion(model, age)
    assert model.criterion.updates == 8 and model.criterion.o2m == pytest.approx(.1)
    checkpoint.save(tmp_path / 'export.pt', model, manifest)
    loaded, _ = checkpoint.load(tmp_path / 'export.pt')
    assert _same(loaded.state_dict(), model.state_dict())


def test_patience_has_minimum_budget_and_each_method_starts_at_recorded_lr():
    for arm in ('L1', 'L2'):
        policy = ContinuationStudy('detect', arm)
        recipe = policy.recipe('masf-extension')
        assert not policy.should_stop(recipe, 4, 0, [])
        assert policy.should_stop(recipe, 5, 0, [])
        assert not policy.should_stop(recipe, 7, 6, [])
        assert policy.should_stop(recipe, 11, 6, [])
        p = torch.nn.Parameter(torch.tensor(1.))
        opt = torch.optim.AdamW([{'params': [p], 'role': 'head', 'group_name': 'head'}])
        policy.set_lr(opt, recipe, 0, 0., 0, False)
        assert opt.param_groups[0]['lr'] == recipe['group_lrs']['head']
        policy.set_lr(opt, recipe, 14, 1., 462, False)
        assert opt.param_groups[0]['lr'] == pytest.approx(recipe['group_lrs']['head'] * .2)


def test_extension_queue_has_valid_ids_and_complete_dependencies(tmp_path, monkeypatch):
    import json

    from masf_yolo.stage4 import extension_workflow
    from yolo_attention.queue_store import QueueStore
    monkeypatch.setattr(extension_workflow, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(extension_workflow, 'review', dict)
    monkeypatch.setattr(extension_workflow, 'verify_source', lambda *args, **kwargs: None)
    (tmp_path / 'extension-cpu.json').write_text(json.dumps({'passed': True}))
    output = tmp_path / 'run'
    extension_workflow.initialize(output)
    state = QueueStore(output).load()
    state.validate()
    assert len(state.jobs) == 15
    assert state.jobs[0].id == 'detect-diagnostics'
    assert state.jobs[-1].id == 'extension-selection'
    assert all(j.id == j.id.lower() for j in state.jobs)
