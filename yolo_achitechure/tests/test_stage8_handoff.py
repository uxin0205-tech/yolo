"""實際 Stage7 parent 到 folded 可攜包的跨程序整合驗證。"""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for name in ('yolo_achitechure', 'yolo_activation', 'yolo_activation/src', 'yolo_combine',
             'yolo_combine/src', 'yolo_attention/src', 'yolo_binaryqk', 'yolo_masf',
             'yolo_achitechure/achitechure_1/src'):
    sys.path.insert(0, str(ROOT / name))

from stage8.common import HERE, sha256
from stage8.model import folded, load, manifest_for, save
from stage8.packaging import bundle


@pytest.mark.parametrize("candidate", ["rep17", "rep17_20_silu", "acb17_20"])
def test_real_folded_handoff_preserves_outputs_and_rejects_fallback(tmp_path, candidate):
    torch.set_num_threads(2)
    initial = json.loads((HERE / 'cpu-contracts.json').read_text())['results'][candidate]
    model = load(initial['initial_checkpoint'])
    with torch.no_grad():
        model.graph.model[17].conv2.bn.weight.fill_(.025)
    deploy = folded(model, operators="hardware")
    checkpoint = tmp_path / 'deploy.pt'
    save(checkpoint, deploy)
    expected_sha = sha256(checkpoint)
    result = bundle(tmp_path / 'handoff', {candidate: {'checkpoint': str(checkpoint),
                    'sha256': expected_sha, 'metrics': {}, 'balanced_score': 0.}}, candidate, sizes=(160,))
    assert result['validation']['models'][candidate]['bitwise_equal']
    assert result['validation']['models'][candidate]['silu_rejected']
    assert checkpoint.is_symlink() and sha256(checkpoint) == expected_sha
    reloaded = load(checkpoint)
    assert manifest_for(reloaded)['mode'] == 'folded'
    reloaded.graph.model[17].act = torch.nn.SiLU()
    with pytest.raises(ValueError):
        manifest_for(reloaded)


def test_silu_multibranch_can_adapt_hardware_without_losing_alias_or_weights(tmp_path):
    from stage8.common import recipe
    from stage8.model import configure
    from stage8.operators import set_operators
    torch.set_num_threads(2)
    initial = json.loads((HERE / 'cpu-contracts.json').read_text())['results']['rep17_20_silu']
    model = load(initial['initial_checkpoint'])
    before = {n: p.detach().clone() for n,p in model.named_parameters()}
    model.stage8_manifest['activation_policy'] = set_operators(model, 'hardware')
    model.stage8_manifest['operators'] = 'hardware'
    path = tmp_path / 'adapt.pt'
    save(path, model)
    reloaded = load(path)
    cfg = recipe('rep17_20_silu')
    cfg['operators'] = 'hardware'
    configure(reloaded, cfg)
    assert manifest_for(reloaded)['operators'] == 'hardware'
    assert all(torch.equal(p, before[n]) for n,p in reloaded.named_parameters())
    assert not any(m.training for m in reloaded.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert not any(isinstance(m, torch.nn.SiLU) for m in reloaded.modules())
