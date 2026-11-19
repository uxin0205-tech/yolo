import json

import pytest
import torch

from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.stage3_checkpoint import build, load, save


@pytest.mark.parametrize('candidate', ['N2', 'N3', 'D1', 'D2', 'D3'])
def test_actual_s1_parent_is_preserved_and_reload_has_same_path(candidate, tmp_path):
    from binaryqk.model import SITES
    from binaryqk.scale_checkpoint import load as load_parent

    from yolo_attention.stage3_checkpoint import review
    source, _ = load_parent(review()['parent'])
    model = build(candidate)
    assert all(type(model.get_submodule(site)) is HardwareFriendlyAttention for site in SITES)
    source_params = dict(source.named_parameters())
    assert all(torch.equal(value, source_params[name]) for name, value in model.named_parameters())
    path = tmp_path/'model.pt'
    save(path, model)
    restored, manifest = load(path)
    assert restored.names == source.names
    assert manifest['candidate'] == candidate
    assert manifest['lineage']['parent_sha256'] == review()['parent_sha256']
    x = torch.rand(1, 3, 64, 96)
    with torch.inference_mode():
        assert torch.equal(model(x)[0], restored(x)[0])
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert metadata['sha256']


def test_missing_contract_or_float_fallback_rejected(tmp_path):
    from binaryqk.model import SITES
    model = build('D1')
    model.get_submodule(SITES[0]).normalizer = torch.nn.Softmax(-1)
    with pytest.raises(ValueError):
        save(tmp_path/'bad.pt', model)
