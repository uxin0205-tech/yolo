"""CPU 驗證精確前向、明確 surrogate 梯度、序列化與整圖推論一致性。"""
import copy
import io
from common import ROOT,SOURCE,registry,write_json
import torch
from ultralytics import YOLO
from qk_challenger import install, remove, ExactDotSurrogate


def tensors(value):
    if isinstance(value,torch.Tensor):return [value]
    if isinstance(value,dict):return [t for v in value.values() for t in tensors(v)]
    if isinstance(value,(tuple,list)):return [t for v in value for t in tensors(v)]
    return []


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    base=YOLO(str(SOURCE/registry()['float'])).model.float()
    candidate=copy.deepcopy(base)
    install(candidate)
    proofs=[]
    source_sites=[m for m in base.modules() if m.__class__.__name__=='HardwareFriendlyAttention']
    target_sites=[m for m in candidate.modules() if m.__class__.__name__=='HardwareFriendlyAttention']
    for source,target in zip(source_sites,target_sites):
        q=torch.randn(2,source.num_heads,source.key_dim,16,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        source.train();target.train()
        expected=source.score(q,k);actual=target.score(q,k)
        assert not expected.requires_grad and actual.requires_grad
        assert torch.equal(expected,actual)
        dq,dk=torch.autograd.grad(actual.square().mean(),(q,k))
        assert torch.isfinite(dq).all() and torch.isfinite(dk).all()
        assert dq.abs().sum()>0 and dk.abs().sum()>0
        proofs.append({'forward_exact':True,'q_grad_norm':float(dq.norm()),'k_grad_norm':float(dk.norm())})
    q=torch.sign(torch.randn(1,2,8,5)).requires_grad_()
    k=torch.sign(torch.randn_like(q)).requires_grad_()
    w=torch.randn(1,2,5,5)
    actual=ExactDotSurrogate.apply(q,k)
    dq,dk=torch.autograd.grad((actual*w).sum(),(q,k))
    assert torch.equal(dq,k.detach()@w.transpose(-2,-1))
    assert torch.equal(dk,q.detach()@w)
    image=torch.rand(1,3,160,160)
    with torch.inference_mode():
        expected=tensors(base.eval()(image));actual=tensors(candidate.eval()(image))
    assert len(expected)==len(actual) and all(torch.equal(a,b) for a,b in zip(expected,actual))
    stream=io.BytesIO();torch.save(candidate,stream);stream.seek(0)
    reloaded=torch.load(stream,map_location='cpu',weights_only=False)
    remove(reloaded)
    with torch.inference_mode():actual=tensors(reloaded.eval()(image))
    assert len(expected)==len(actual) and all(torch.equal(a,b) for a,b in zip(expected,actual))
    write_json(ROOT/'artifacts/qk-challenger-cpu.json',{'status':'passed','sites':proofs,
        'analytic_surrogate_backward_exact':True,'full_graph_160_eval_exact':True,
        'serialization_and_remove_exact':True,'accuracy_improvement_proven':False,
        'note':'surrogate gradient 非離散 sign 真導數；GPU AMP、實際 batch loss、optimizer resume 仍待驗證'})
    print('QK_CHALLENGER_CPU_PASSED',proofs)


if __name__=='__main__':main()
