"""CPU 小型可微模型核對 chain rule；不是 YOLO 訓練或精度實驗。"""
import json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent

class GS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,beta):ctx.beta=beta;return x
    @staticmethod
    def backward(ctx,g):return g*ctx.beta,None

def check(beta,alpha_value):
    dtype=torch.float64
    x=torch.tensor([.5,-1.],dtype=dtype)
    w=torch.tensor([[.2,.1],[-.3,.4]],dtype=dtype,requires_grad=True)
    alpha=torch.tensor(alpha_value,dtype=dtype,requires_grad=True)
    um=torch.tensor([1.,2.],dtype=dtype,requires_grad=True)
    uo=torch.tensor([-1.,.5],dtype=dtype,requires_grad=True)
    fm=w@x;ym=x+alpha*fm;yo=GS.apply(x+alpha*fm,beta)
    em=um@ym-.1;eo=uo@yo+.2
    loss=.1*.5*em.square()+.9*.5*eo.square()
    loss.backward()
    gm=um.detach()*em.detach();go=uo.detach()*eo.detach()
    effective=.1*gm+.9*beta*go
    torch.testing.assert_close(w.grad,alpha.detach()*torch.outer(effective,x),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(alpha.grad,torch.dot(effective,fm.detach()),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(uo.grad,.9*eo.detach()*yo.detach(),atol=1e-12,rtol=1e-12)
    assert x.grad is None
    return {'beta':beta,'alpha':alpha_value,'alpha_grad':alpha.grad.item(),
        'context_grad_norm':w.grad.norm().item(),'pose_one2one_head_grad':uo.grad.tolist()}

def main():
    old=check(.012076444778011642,.02);new=check(1.,.02);zero=check(1.,0.)
    assert old['pose_one2one_head_grad']==new['pose_one2one_head_grad']
    assert zero['context_grad_norm']==0 and zero['alpha_grad']!=0
    second=check(1.,-1e-3*zero['alpha_grad']);assert second['context_grad_norm']>0
    result={'status':'passed','gpu_used':False,'actual_yolo_training':False,
        'tests':['context chain rule','alpha chain rule','head gradient not scaled by beta',
        'alpha zero context task-gradient zero','alpha can unlock subsequent context task-gradient'],
        'cases':[old,new,zero,second],
        'limitation':'解析 toy loss，不是 BBAT 真實 loss／AMP／optimizer 更新驗證'}
    with (HERE/'derivation-check.json').open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print('PASS：5 項 CPU 梯度推導核對；未進行 YOLO 訓練。')

if __name__=='__main__':main()
