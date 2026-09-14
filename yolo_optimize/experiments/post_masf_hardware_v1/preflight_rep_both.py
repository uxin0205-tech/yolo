"""只驗證新雙層候選，不重跑三個已通過候選或 GPU job。"""
import json
import torch
import run_rep_both as both
from models import *
from yolo_combine.graph_materialize import build_graph_validation_models

def close(a,c):
    aa,cc=b.tensors(a),b.tensors(c);assert len(aa)==len(cc)
    for x,y in zip(aa,cc):torch.testing.assert_close(x,y,atol=2e-4,rtol=1e-5)

def main():
    b.initialize();torch.set_num_threads(4);torch.manual_seed(1)
    base,_=build('native_qk_reference');model,source=both.build(both.ARM)
    optimizer=both.optimizer_for(model,both.ARM)
    active={n for n,p in model.named_parameters() if p.requires_grad}
    grouped=[n for g in optimizer.param_groups for n in g['param_names']]
    assert set(grouped)==active and len(grouped)==len(set(grouped))
    for index in both.LAYERS:
        assert isinstance(model.graph.model[index],RepConv)
        assert model.graph.model[index].bn is None
        assert all(p.requires_grad for p in model.graph.model[index].parameters())
        assert any(g['role']=='rep' and any(n.startswith(f'graph.model.{index}.') for n in g['param_names']) for g in optimizer.param_groups)
    assert not any(p.requires_grad for n,p in model.named_parameters() if '.attn.' in n or '.p3_masf.' in n)
    x=torch.rand(1,3,160,160)
    base.eval();model.eval()
    with torch.no_grad():close(base(x,task='both'),model(x,task='both'))
    extra=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())
    assert extra==66048+263168
    # 模擬兩條新支路都非零，避免只驗到初始化零分支的平凡 fold。
    with torch.no_grad():
        for i in both.LAYERS:model.graph.model[i].conv2.bn.weight.fill_(.1)
        expected=model(x,task='both');deploy=folded(model);close(expected,deploy(x,task='both'))
    assert not any(isinstance(deploy.graph.model[i],RepConv) for i in both.LAYERS)
    for kind in ('float','bittrue'):
        pair=build_graph_validation_models(deploy,source,kind=kind)
        for m in (pair.detect,pair.pose):b.verify_pwl(m)
    assert not torch.cuda.is_initialized()
    b.save(HERE/'artifacts/rep-both-preflight-v1.json',{'status':'passed','gpu_used':False,
        'arm':both.ARM,'layers':[17,20],'parent_sha256':PARENT_SHA,
        'initial_output_matches_B5':True,'nonzero_branches_fold_equivalent':True,'float_bittrue_materialization':True,
        'both_layers_trainable_at_rep_lr':True,'attention_and_both_masf_frozen':True,
        'extra_training_parameters':extra,'extra_training_conv_MAC':209715200,'extra_deploy_conv_MAC':0,
        'limitation':'CPU合成輸入與非零branch fold；實際task-loss梯度須先GPU smoke'})
    print('PASS Rep17+20：共同起點等價、兩支路非零fold、參數群與PWL重建。')

if __name__=='__main__':main()
