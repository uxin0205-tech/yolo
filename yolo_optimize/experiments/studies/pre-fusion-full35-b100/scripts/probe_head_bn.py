"""原生 E1 回退診斷：僅恢復 Detect BN running statistics，保留已學參數。"""
import copy
import json
from common import ROOT,setup,prepare_coco,write_json,sha256
from train_hog import PARENT,inference_graph
from continue_a0 import validate
import torch


def main():
    output=ROOT/'artifacts/native-e1-head-bn-probe-v1';output.mkdir(parents=True,exist_ok=False)
    setup();data=prepare_coco();torch.set_num_threads(8)
    source=ROOT/'artifacts/prefusion-hog-control-v1/epoch-01-resume.pt'
    payload=torch.load(source,map_location='cpu',weights_only=False)
    model=inference_graph(payload['ema']).float();del payload
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)['ema'].float()
    before={n:t.clone() for n,t in model.state_dict().items()};reference=parent.state_dict();state=model.state_dict()
    changed=[]
    for name in state:
        if name.startswith('model.23.') and any(name.endswith(s) for s in ('running_mean','running_var','num_batches_tracked')):
            state[name].copy_(reference[name]);changed.append(name)
    assert changed
    assert all(torch.equal(t,state[n]) for n,t in before.items() if n not in changed)
    original=json.loads((source.parent/'summary.json').read_text())['epochs'][0]['ema']
    restored=validate(model,'bittrue',data,output/'restored-bn')
    write_json(output/'summary.json',{'status':'passed','source':str(source),'source_sha256':sha256(source),
        'restored_from':str(PARENT),'restored_buffer_names':changed,'learned_parameters_unchanged':True,
        'original':original,'restored':restored,'delta':{k:restored[k]-original[k] for k in original},
        'optimizer_steps':0,'diagnostic_only':True,'accuracy_winner':False})
    print('JOB_DONE HEAD_BN_PROBE',restored,flush=True)


if __name__=='__main__':main()
