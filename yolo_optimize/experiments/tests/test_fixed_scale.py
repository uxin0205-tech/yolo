"""使用封存 BinaryScore 驗證等價、校準與必要錯誤不被略過。"""
import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, '/home/uxin/yolo/yolo_combine/final/full35/source_bundle/code')
from yolo_attention.binary_basis import BinaryScore
from yolo_optimize.fixed_scale import install_fixed_early_return


@pytest.mark.parametrize('mode', ['dynamic', 'power_of_two'])
def test_score_and_calibration_equivalence(mode):
    torch.manual_seed(8)
    original = BinaryScore(num_heads=4, basis='hadamard', scale_mode=mode, use_ste=False).eval()
    if mode != 'dynamic':
        original.set_fixed_coefficients(torch.rand(4, 2) + .1)
    candidate = copy.deepcopy(original)
    install_fixed_early_return(candidate)
    q, k = torch.randn(2, 4, 8, 9), torch.randn(2, 4, 8, 9)
    expected = original(q, k)
    if mode != 'dynamic':
        with patch.object(candidate, '_dynamic_coefficient', side_effect=AssertionError('冗餘計算')):
            assert torch.equal(expected, candidate(q, k))
        for model in (original, candidate):
            model.begin_calibration()
        assert torch.equal(original(q, k), candidate(q, k))
        assert torch.equal(original.finish_calibration(), candidate.finish_calibration())
    else:
        assert torch.equal(expected, candidate(q, k))
    for key, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])


def test_missing_calibration_and_duplicate_install_rejected():
    score = BinaryScore(num_heads=4, basis='hadamard', scale_mode='power_of_two').eval()
    install_fixed_early_return(score)
    with pytest.raises(ValueError):
        install_fixed_early_return(score)
    with pytest.raises(RuntimeError, match='set_fixed_coefficients'):
        score(torch.zeros(1, 4, 8, 2), torch.zeros(1, 4, 8, 2))
