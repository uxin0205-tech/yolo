"""從訓練快照匯出無 surrogate 的推論模型，避免 YOLO.save 隱式 FP16 rounding。"""
import argparse
import copy
from pathlib import Path
from common import ROOT,SOURCE,write_json,sha256
import torch
from ultralytics import YOLO
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from qk_challenger import remove


def export(snapshot,destination,state='ema'):
    snapshot=Path(snapshot).resolve();destination=Path(destination).resolve()
    if not snapshot.is_relative_to(ROOT/'artifacts') or not destination.is_relative_to(ROOT/'artifacts'):
        raise ValueError('來源與產物必須屬於此研究 artifacts')
    if destination.exists():raise FileExistsError(destination)
    payload=torch.load(snapshot,map_location='cpu',weights_only=False)
    model=copy.deepcopy(payload[state]).float().eval()
    if hasattr(model,'hog_aux'):del model.hog_aux
    if model.model[17].__class__.__name__=='RepConv':
        from rep17 import folded
        model=folded(model)
    remove(model)
    if hasattr(model,'criterion'):del model.criterion
    convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/'configs/attention/bittrue-pwl-final.yaml'))
    model.eval()
    destination.parent.mkdir(parents=True,exist_ok=True)
    train_args=model.args if isinstance(model.args,dict) else vars(model.args)
    torch.save({'model':model,'ema':None,'optimizer':None,'epoch':-1,'train_args':train_args,
                'study_export':{'source':str(snapshot),'state':state,'precision':'FP32 weights; Bit-True PWL'}},destination)
    reloaded=YOLO(str(destination)).model.float().eval()
    generator=torch.Generator().manual_seed(0)
    image=torch.rand(1,3,160,160,generator=generator)
    with torch.inference_mode():
        from verify_qk_challenger import tensors
        expected=tensors(model(image));actual=tensors(reloaded(image))
    if len(actual)!=len(expected) or not all(torch.equal(a,b) for a,b in zip(actual,expected)):
        raise AssertionError('匯出重載推論不一致')
    write_json(destination.with_suffix('.json'),{'status':'passed','snapshot':str(snapshot),
        'snapshot_sha256':sha256(snapshot),'inference_sha256':sha256(destination),'state':state,
        'weights_precision':'FP32','cpu160_reload_exact':True,'full_coco_revalidation_done':False,
        'deployment_latency_measured':False,'accuracy_winner':False})
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--state',choices=['ema','model'],default='ema')
    a=p.parse_args();torch.set_num_threads(4);print('EXPORT_PASSED',export(a.snapshot,a.output,a.state))
