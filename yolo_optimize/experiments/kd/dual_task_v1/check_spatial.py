"""有意義的 CPU 契約：梯度、跨 channel、微批次等價與空圖。"""
import torch
from spatial_kd import spatial_loss

def main():
    torch.manual_seed(1)
    s=[torch.randn(4,3,h,h,requires_grad=True) for h in (8,4,2)]
    t=[torch.randn(4,7,h,h,requires_grad=True) for h in (8,4,2)]
    full=spatial_loss(s,t).mean()
    chunk=sum(spatial_loss([x[a:a+2] for x in s],[x[a:a+2] for x in t]).sum() for a in (0,2))/4
    torch.testing.assert_close(full,chunk)
    ga=torch.autograd.grad(full,s,retain_graph=True)
    gb=torch.autograd.grad(chunk,s)
    for a,b in zip(ga,gb):
        torch.testing.assert_close(a,b);assert torch.isfinite(a).all() and a.abs().sum()>0
    assert all(x.grad is None for x in t)
    same=spatial_loss(s,[x.detach().repeat(1,2,1,1) for x in s])
    assert same.max()<1e-10
    zeros=[torch.zeros_like(x,requires_grad=True) for x in s]
    z=spatial_loss(zeros,[x.detach() for x in zeros]).sum()
    assert float(z.detach())==0 and all(torch.isfinite(x).all() for x in torch.autograd.grad(z,zeros))
    try:spatial_loss(s,[t[0][:,:,:-1,:],t[1],t[2]])
    except ValueError:pass
    else:raise AssertionError('spatial misalignment not rejected')
    print('PASS: live gradients, teacher detach, channel invariance, microbatch equivalence, zero maps and shape guard')

if __name__=='__main__':main()
