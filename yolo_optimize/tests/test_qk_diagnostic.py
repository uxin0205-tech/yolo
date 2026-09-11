"""Score 介入的公式與 eval-only 保護，不涉及資料集。"""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from yolo_optimize.qk_diagnostic import fp_dot,ScoreDiagnosticSource,SITES
import torch
from torch import nn
import pytest

class FakeScore(nn.Module):
    basis='hadamard'
    fixed_coefficients_ready=False
    def forward(self,q,k): return torch.zeros(q.shape[0],q.shape[1],q.shape[-1],q.shape[-1])

def test_formula_and_selected_site_only():
    torch.manual_seed(3)
    q,k=torch.randn(1,2,8,5),torch.randn(1,2,8,5)
    torch.testing.assert_close(fp_dot(q,k),torch.einsum('bhdi,bhdj->bhij',q,k)/8**.5)
    scores={path:FakeScore().eval() for path in SITES.values()}
    model=SimpleNamespace(named_modules=lambda:iter(scores.items()))
    source=SimpleNamespace(build_task_models=lambda kind:SimpleNamespace(detect=model,pose=model))
    proxy=ScoreDiagnosticSource(source,10)
    proxy.build_task_models('bittrue')
    torch.testing.assert_close(scores[SITES[10]](q,k),fp_dot(q,k))
    assert not scores[SITES[22]](q,k).any()
    scores[SITES[10]].train()
    with pytest.raises(RuntimeError): scores[SITES[10]](q,k)

def test_invalid_shape_and_site():
    with pytest.raises(ValueError): fp_dot(torch.zeros(2,3),torch.zeros(2,3))
    with pytest.raises(ValueError): ScoreDiagnosticSource(None,17)
