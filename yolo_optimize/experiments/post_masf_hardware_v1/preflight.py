"""CPU 核對 Q/K task-score 梯度、尺度與 bias 定點生效、Rep17/20 等價。"""
import copy
import json
import torch
from models import *
from yolo_combine.graph_materialize import build_graph_validation_models

def close(a,c):
    aa,cc=b.tensors(a),b.tensors(c);assert len(aa)==len(cc)
    for x,y in zip(aa,cc):torch.testing.assert_close(x,y,atol=2e-4,rtol=1e-5)

def main():
    b.initialize();torch.set_num_threads(4);torch.manual_seed(1)
    base,_=build('native_qk_reference');base.eval()
    x=torch.rand(1,3,160,160)
    with torch.no_grad():expected=base(x,task='both')
    results={}
    for arm in ARMS:
        model,source=build(arm);model.eval()
        if arm.startswith('rep'):
            with torch.no_grad():close(expected,model(x,task='both'));close(model(x,task='both'),folded(model)(x,task='both'))
            index=int(arm[3:]);rep=model.graph.model[index];assert rep.bn is None
            c=rep.conv1.conv
            results[arm]={'zero_branch_initial_equivalence':True,'fold_equivalence':True,'identity_branch':False,
                'channels':[c.in_channels,c.out_channels],'stride':list(c.stride),
                'extra_training_parameters':sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters()),
                'deployment_extra_conv_mac':0}
        else:
            details={}
            for name in SITES:
                m=model.graph.get_submodule(name)
                m.train()
                q=torch.randn(2,m.num_heads,m.key_dim,9,requires_grad=True);k=torch.randn_like(q,requires_grad=True)
                score=m.score(q,k);objective=(score*torch.randn_like(score)).mean();objective.backward()
                assert q.grad is not None and k.grad is not None and q.grad.abs().max()>0 and k.grad.abs().max()>0
                assert m.score.coefficients.grad is not None and m.score.coefficients.grad.abs().max()>0
                active=score.detach();m.eval()
                with torch.no_grad():torch.testing.assert_close(active,m.score(q,k),atol=0,rtol=0)
                m.bias.train();logits=torch.randn(2,m.num_heads,9,9,requires_grad=True)
                normalized=m.normalize(m.bias(logits,height=3,width=3))
                (normalized*torch.randn_like(normalized)).sum().backward()
                assert m.bias.table_x.grad.abs().max()>0 and m.bias.table_y.grad.abs().max()>0
                details[name]={'q_grad_max':float(q.grad.abs().max()),'k_grad_max':float(k.grad.abs().max()),
                    'scale_grad_max':float(m.score.coefficients.grad.abs().max()),
                    'bias_x_grad_max':float(m.bias.table_x.grad.abs().max()),'bias_y_grad_max':float(m.bias.table_y.grad.abs().max()),
                    'scales':m.score.quantized().tolist(),'q_ste_coverage':float((q.detach().abs()<=1).float().mean()),
                    'k_ste_coverage':float((k.detach().abs()<=1).float().mean()),
                    'offset_entries':m.bias.table_x.numel()+m.bias.table_y.numel()}
            results[arm]={'score_train_eval_exact':True,'sites':details,'dynamic_image_scales':False,'real_task_loss_checked':False}
        for kind in ('float','bittrue'):
            material=build_graph_validation_models(folded(model),source,kind=kind)
            for m in (material.detect,material.pose):b.verify_pwl(m)
            assert not any(isinstance(m,RepConv) for m in material.detect.modules())
    assert not torch.cuda.is_initialized()
    b.save(HERE/'artifacts/preflight-v1.json',{'status':'passed','gpu_used':False,'results':results,
        'parent_sha256':PARENT_SHA,'pwl':[-10,0,20],'limitation':'合成 score CPU 梯度，不是 task loss 精度；真實雙任務梯度須 GPU smoke。'})
    print('PASS 三候選 CPU：定點 scale/bias 梯度與原生 QK 參考、Rep17/20 初始／fold 等價。')

if __name__=='__main__':main()
