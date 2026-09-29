"""CPU 核對 BinaryQK E5 lineage 與三個 RepConv 分支。"""
import json
import torch
from models import *
from run_arm import optimizer_for,constants
from yolo_combine.graph_materialize import build_graph_validation_models

def close(a,c):
    aa,cc=b.tensors(a),b.tensors(c);assert len(aa)==len(cc)
    for x,y in zip(aa,cc):torch.testing.assert_close(x,y,atol=2e-4,rtol=1e-5)

def main():
    b.initialize();torch.set_num_threads(4);torch.manual_seed(1)
    base,_=build('parent_reference');stored=torch.load(PARENT,map_location='cpu',weights_only=True)['state_dict']
    assert all(torch.equal(v,stored[n]) for n,v in base.state_dict().items())
    parent_codes=json.loads((PARENT.parent/'constants.json').read_text())
    assert constants(base)==parent_codes
    x=torch.rand(1,3,160,160)
    with torch.no_grad():expected=base(x,task='both')
    results={}
    for arm in ARMS:
        model,source=build(arm);optimizer=optimizer_for(model,arm)
        active={n for n,p in model.named_parameters() if p.requires_grad}
        grouped=[n for g in optimizer.param_groups for n in g['param_names']]
        assert set(grouped)==active and len(grouped)==len(set(grouped))
        assert not any(p.requires_grad for n,p in model.named_parameters() if '.attn.' in n or '.p3_masf.' in n)
        assert constants(model)==parent_codes
        for site in SITES:assert isinstance(model.graph.get_submodule(site),HardwareFriendlyAttention)
        for i in LAYERS[arm]:
            assert model.graph.model[i].bn is None
            assert any(g['role']=='rep' and any(n.startswith(f'graph.model.{i}.') for n in g['param_names']) for g in optimizer.param_groups)
        model.eval()
        with torch.no_grad():
            close(expected,model(x,task='both'))
            for i in LAYERS[arm]:model.graph.model[i].conv2.bn.weight.fill_(.1)
            deploy=folded(model);close(model(x,task='both'),deploy(x,task='both'))
        for kind in ('float','bittrue'):
            pair=build_graph_validation_models(deploy,source,kind=kind)
            for m in (pair.detect,pair.pose):
                b.verify_pwl(m)
                for site in SITES:
                    a=m.get_submodule(site);trained=base.graph.get_submodule(site)
                    assert isinstance(a,HardwareFriendlyAttention)
                    assert torch.equal(a.score.coefficients,trained.score.coefficients)
                    assert torch.equal(a.score.inference_coefficients,trained.score.quantized())
                    assert torch.equal(a.bias.table_x,trained.bias.table_x)
                    assert torch.equal(a.bias.table_y,trained.bias.table_y)
        results[arm]={'initial_equivalence':True,'nonzero_fold_equivalence':True,'binary_templates_and_constants_preserved':True,'optimizer_groups':True,'both_masf_and_attention_frozen':True,'extra_training_params':sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())}
    assert not torch.cuda.is_initialized()
    b.save(HERE/'artifacts/preflight-v1.json',{'status':'passed','gpu_used':False,'parent_sha256':PARENT_SHA,'results':results,'pwl':[-10,0,20]})
    print('PASS：三組起點／BinaryQK 常數／MASF／fold／Float 與 BitTrue 模板一致')
if __name__=='__main__':main()
