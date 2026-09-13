"""CPU 實作稽核：固定尺度、gamma、bias、點積維度；不是精度實驗。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from evaluate_switches import SwitchSource, graph, initialize, SELECTED, SELECTED_SHA, save
from yolo_attention.binary_basis import fast_hadamard_transform, deterministic_sign
from qk_challenger import TrainableBinaryScore


def main():
    initialize();torch.set_num_threads(4);torch.manual_seed(50900913)
    model=graph(SwitchSource())
    rows=[]
    for name,m in model.named_modules():
        if type(m).__name__!='HardwareFriendlyAttention':continue
        q=torch.randn(1,m.num_heads,m.key_dim,25);k=torch.randn_like(q)
        score=copy.deepcopy(m.score).eval()
        with torch.no_grad():
            before=score(q,k)
            score.gamma.mul_(7)
            after=score(q,k)
            assert torch.equal(before,after)
            q0,k0=deterministic_sign(q),deterministic_sign(k)
            q1=deterministic_sign(fast_hadamard_transform(q,dim=-2))
            k1=deterministic_sign(fast_hadamard_transform(k,dim=-2))
            reference=score._fixed(0)*(q0.transpose(-2,-1)@k0)+score._fixed(1)*(q1.transpose(-2,-1)@k1)
            torch.testing.assert_close(before,reference,rtol=0,atol=0)
        train_score=copy.deepcopy(m.score)
        train_score.__class__=TrainableBinaryScore
        train_score.use_ste=True;train_score.train()
        q.requires_grad_();k.requires_grad_()
        train_score(q,k).square().mean().backward()
        assert q.grad is not None and k.grad is not None
        assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
        assert train_score.gamma.grad is None
        params={n:{'shape':list(p.shape),'count':p.numel(),'rms':float(p.detach().square().mean().sqrt()),
                   'max_abs':float(p.detach().abs().max())} for n,p in m.bias.named_parameters()}
        # 合成 row 的共同常數在減 row max 後消失；位置差異則不會。
        a=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
        assert torch.equal(a-a.amax(-1,keepdim=True),(a+4)-(a+4).amax(-1,keepdim=True))
        scale=m.key_dim**-.5
        rows.append({'site':name,'heads':m.num_heads,'key_dim':m.key_dim,'head_dim':m.head_dim,
            'native_scale':scale,'native_scale_exact_power_of_two':scale.hex(),
            'fixed_coefficients':m.score.fixed_coefficients.tolist(),
            'fixed_is_buffer': 'fixed_coefficients' in dict(m.score.named_buffers()),
            'fixed_requires_grad':m.score.fixed_coefficients.requires_grad,
            'gamma_change_leaves_output_exact_equal':True,'gamma_grad_is_none_fixed_mode':True,
            'xnor_matches_signed_dot_exact':True,'surrogate_qk_gradient_finite':True,
            'q_gradient_nonzero':int(torch.count_nonzero(q.grad)),
            'k_gradient_nonzero':int(torch.count_nonzero(k.grad)),
            'bias_kind':str(m.bias.kind),'bias_parameters':params,
            'row_constant_bias_cancels':True})
    assert len(rows)==2
    save(HERE/'artifacts/binary-audit-v1.json',{'status':'passed','device':'cpu','synthetic_only':True,
        'accuracy_or_scale_optimality_test':False,'checkpoint':str(SELECTED),'sha256':SELECTED_SHA,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sites':rows})
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
