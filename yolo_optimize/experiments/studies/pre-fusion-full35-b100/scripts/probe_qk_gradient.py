"""實測 B100 score 的 Q/K 梯度；不是 AP 或因果根因證明。"""
from common import ROOT, SOURCE, registry, write_json
import torch
from ultralytics import YOLO


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model=YOLO(str(SOURCE/registry()['float'])).model.float()
    report={}
    for name,module in model.named_modules():
        if module.__class__.__name__!='HardwareFriendlyAttention':continue
        score=module.score.train()
        q=torch.randn(2,module.num_heads,module.key_dim,16,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        value=score(q,k)
        gradients=torch.autograd.grad(value.sum(),(q,k),allow_unused=True) if value.requires_grad else (None,None)
        report[name]={'use_ste':score.use_ste,'scale_mode':str(score.scale_mode),
            'score_requires_grad':value.requires_grad,
            'q_gradient_nonzero':gradients[0] is not None and bool(gradients[0].abs().sum()>0),
            'k_gradient_nonzero':gradients[1] is not None and bool(gradients[1].abs().sum()>0),
            'fixed_coefficients':score.fixed_coefficients.detach().tolist()}
    write_json(ROOT/'artifacts/qk-gradient-probe.json',{'status':'complete',
        'scope':'B100 Float score isolated CPU, train mode, seed0, N16',
        'sites':report,'training_performed':False,
        'limitation':'確認 score 的梯度路徑；尚未證明修復梯度可提升 AP，亦非整模型所有路徑梯度稽核。'})
    print(report)


if __name__=='__main__':main()
