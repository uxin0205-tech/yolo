"""驗證非預設 BN 與非零 RepConv 分支的 eval 格式轉換。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import pytest
from yolo_optimize.repconv import from_conv,to_eval_conv,Conv

def test_bn_metadata_zero_branch_and_nonzero_eval_roundtrip():
    torch.manual_seed(7)
    old=Conv(8,8,3,2).eval()
    old.bn.eps=.001
    old.bn.momentum=.03
    rep=from_conv(old)
    x=torch.randn(2,8,17,17)
    assert torch.equal(old(x),rep(x))
    assert rep.conv1.bn.eps==.001 and rep.conv1.bn.momentum==.03
    with torch.no_grad():
        rep.conv2.bn.weight.fill_(.3)
        rep.conv2.bn.bias.fill_(.1)
    before={k:v.clone() for k,v in rep.state_dict().items()}
    folded=to_eval_conv(rep,old)
    assert set(folded.state_dict())==set(old.state_dict())
    torch.testing.assert_close(rep(x),folded(x),atol=1e-5,rtol=1e-4)
    restored=Conv(8,8,3,2).eval()
    restored.bn.eps=old.bn.eps
    restored.load_state_dict(folded.state_dict(),strict=True)
    assert torch.equal(restored(x),folded(x))
    assert all(torch.equal(before[k],v) for k,v in rep.state_dict().items())
    assert all(not p.requires_grad for p in folded.parameters())

def test_reject_training_conversion_and_wrong_seam():
    old=Conv(8,8,3,2)
    with pytest.raises(ValueError): to_eval_conv(from_conv(old),old)
    with pytest.raises(ValueError): from_conv(Conv(8,8,1,1))
