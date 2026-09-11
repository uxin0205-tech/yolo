"""KD 前置 CPU 稽核：區分 raw binary 梯度與既有 training-only surrogate。"""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'activation/bridge_v1'))
from verify_selected import SelectedSource, initialize, SELECTED, SELECTED_SHA
import copy
import json
import torch
from qk_challenger import TrainableBinaryScore


def main():
    initialize();torch.set_num_threads(4);torch.manual_seed(1)
    pair=SelectedSource().build_task_models()
    model=pair.detect.requires_grad_(False)
    results=[]
    for name,m in model.named_modules():
        if type(m).__name__!='HardwareFriendlyAttention':continue
        native=copy.deepcopy(m.score).train()
        candidate=copy.deepcopy(native)
        candidate.__class__=TrainableBinaryScore
        candidate.use_ste=True
        q=torch.randn(1,m.num_heads,m.key_dim,25,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        a=native(q,k);b=candidate(q,k)
        assert torch.equal(a,b)
        assert not a.requires_grad and b.requires_grad
        dq,dk=torch.autograd.grad(b.square().mean(),(q,k))
        assert all(torch.isfinite(t).all() and t.abs().sum()>0 for t in (dq,dk))
        assert all(torch.equal(v,candidate.state_dict()[n]) for n,v in native.state_dict().items())
        with torch.no_grad():assert torch.equal(native.eval()(q,k),candidate.eval()(q,k))
        results.append({'site':name,'raw_binary_score_has_qk_gradient':False,
            'surrogate_forward_exact':True,'surrogate_q_gradient_norm':float(dq.norm()),
            'surrogate_k_gradient_norm':float(dk.norm()),'eval_exact':True,'state_unchanged':True})
    assert len(results)==2
    HERE.joinpath('artifacts').mkdir(exist_ok=True)
    with (HERE/'artifacts/score-gradient-preflight-v1.json').open('x') as f:
        json.dump({'status':'cpu_probe_passed_not_kd_ready','student':str(SELECTED),'sha256':SELECTED_SHA,
            'results':results,'limits':'尚未驗證全模型 KD-only 更新、AMP、GT mapping、teacher優勢；未啟動KD訓練。'},f,indent=2)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
