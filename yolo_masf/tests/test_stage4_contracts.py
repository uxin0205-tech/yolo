import copy
import tempfile
from pathlib import Path

import pytest
import torch
from binaryqk.data_safety import ReadOnlyDatasetError, install_readonly_data_guard
from masf_yolo.stage4 import checkpoint
from masf_yolo.stage4.common import PROJECT, ROOT
from masf_yolo.stage4.training import MASFStudy


def test_runtime_cache_allowed_but_source_and_symlink_escape_rejected(tmp_path):
    workspace = PROJECT / 'artifacts/datasets'
    workspace.mkdir(parents=True, exist_ok=True)
    install_readonly_data_guard(cache_workspace=workspace)
    # Import after installation so the actual guard is exercised.
    from ultralytics.data import utils
    with tempfile.TemporaryDirectory(dir=workspace) as temporary:
        folder = Path(temporary)
        utils.save_dataset_cache_file('', folder / 'labels.cache', {}, '1')
        assert (folder / 'labels.cache').is_file()
        (folder / 'escape').symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ReadOnlyDatasetError):
            utils.save_dataset_cache_file('', folder / 'escape/labels.cache', {}, '1')
        assert not (tmp_path / 'labels.cache').exists()
    with pytest.raises(ReadOnlyDatasetError):
        utils.save_dataset_cache_file('', ROOT / 'original/forbidden.cache', {}, '1')
    with pytest.raises(ReadOnlyDatasetError):
        install_readonly_data_guard(cache_workspace=ROOT / 'original')


def test_manifest_rejects_bridge_lie_and_float_softmax_fallback():
    torch.set_num_threads(2)
    model = checkpoint.build('detect')
    original = copy.deepcopy(model.stage4_manifest)
    model.stage4_manifest['masf']['bridge_beta'] = .25
    with pytest.raises(ValueError, match='manifest'):
        checkpoint.manifest_for(model)
    model.stage4_manifest = original
    model.get_submodule('model.10.m.0.attn').normalizer = torch.nn.Softmax(dim=-1)
    with pytest.raises(ValueError, match='fallback'):
        checkpoint.manifest_for(model)


def test_policy_freezes_parent_and_uses_distinct_task_clocks():
    model = checkpoint.build('pose')
    policy = MASFStudy('pose', ROOT / 'yolo_pose_training/artifacts/preserved/stagewise_lr_p3_640/evaluation.json')
    recipe = policy.recipe('masf')
    policy.configure(model, recipe['scope'])
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith('model.23.'))
    assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    optimizer = policy.optimizer(model, recipe)
    policy.set_lr(optimizer, recipe, 4, 1., 46, False)
    assert recipe['steps_per_epoch'] == 47
    for group in optimizer.param_groups:
        assert group['lr'] == pytest.approx(recipe['group_lrs'][group['role']] * .5)
        if group['role'] == 'alpha':
            assert group['weight_decay'] == 0
    assert 'model.23.p3_masf.context.project.conv.weight' in policy.update_names(model)
    assert 'model.23.p3_masf.context.project.conv.weight' not in policy.gradient_names(model)
