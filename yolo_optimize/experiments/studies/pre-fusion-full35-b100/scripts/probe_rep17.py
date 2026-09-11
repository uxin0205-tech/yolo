"""實際融合前模型的 RepConv 初始整圖等價與折疊後完整 COCO 基準。"""
import json
from common import ROOT,setup,prepare_coco,write_json
from train_rep17 import prepare_model,PARENT_SHA
from rep17 import folded
from continue_a0 import validate
from verify_qk_challenger import tensors
import torch


def main():
    output=ROOT/'artifacts/rep17-preflight-v1';output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);setup();data=prepare_coco()
    control=prepare_model('control').eval();rep=prepare_model('rep').eval();deploy=folded(rep)
    image=torch.rand(1,3,160,160,generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        expected=tensors(control(image));initial=tensors(rep(image));after=tensors(deploy(image))
    assert len(expected)==len(initial)==len(after)
    assert all(torch.equal(x,y) for x,y in zip(expected,initial))
    for x,y in zip(initial,after):torch.testing.assert_close(x,y,atol=1e-3,rtol=1e-4)
    assert set(control.state_dict())==set(deploy.state_dict())
    reference=json.loads((ROOT/'artifacts/a0-scope-late-v1/summary.json').read_text())['epochs'][-1]['ema']
    metrics=validate(deploy,'bittrue',data,output/'initial-folded')
    delta={k:metrics[k]-reference[k] for k in reference}
    assert all(abs(x)<=1e-4 for x in delta.values()),delta
    write_json(output/'summary.json',{'status':'passed','parent_sha256':PARENT_SHA,
        'initial_cpu160_exact':True,'folded_cpu160_atol':1e-3,'folded_cpu160_rtol':1e-4,
        'folded_max_abs':max(float((x-y).abs().max()) for x,y in zip(initial,after)),
        'initial_folded_metrics':metrics,'initial_folded_delta':delta,'initial_folded_ap_tolerance':1e-4,
        'inference_state_keys_match':True,'control_layer17_params':sum(p.numel() for p in control.model[17].parameters()),
        'rep_layer17_params':sum(p.numel() for p in rep.model[17].parameters()),'full_coco_images':5000})
    print('JOB_DONE REP17_PREFLIGHT',delta,flush=True)


if __name__=='__main__':main()
