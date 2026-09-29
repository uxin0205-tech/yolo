"""單點 layer17 原 Conv／RepConv 配對；其餘參數與全部 BN 統計固定。"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
from common import ROOT,SOURCE,setup,prepare_coco,write_json,sha256
import torch
from ultralytics.utils.torch_utils import ModelEMA
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from qk_challenger import install as install_score,remove
from continue_a0 import ContinuationHarness,check_fixed,validate
from rep17 import install,folded

PARENT=ROOT/'artifacts/a0-scope-late-v1/epoch-08-resume.pt'
PARENT_SHA='27c09c6685a003e9ff106898fd898199e184ee195cce643dd668079abadb9342'


def prepare_model(variant):
    if sha256(PARENT)!=PARENT_SHA:raise ValueError('探索 parent 已變更')
    model=copy.deepcopy(torch.load(PARENT,map_location='cpu',weights_only=False)['ema']).float()
    remove(model)
    if hasattr(model,'criterion'):del model.criterion
    convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/'configs/attention/float-pwl-final.yaml'))
    install_score(model)
    if variant=='rep':install(model)
    return model


class Harness(ContinuationHarness):
    variant='control'
    def get_model(self,cfg=None,weights=None,verbose=True):return prepare_model(self.variant)
    def build_optimizer(self,model,*args,**kwargs):
        groups={}
        for name,p in model.named_parameters():
            active=name.startswith('model.17.');p.requires_grad_(active)
            if active:groups.setdefault(p.ndim>=2 and not name.endswith('.bias'),[]).append(p)
        return torch.optim.AdamW([{'params':ps,'lr':2e-6,'initial_lr':2e-6,
            'weight_decay':0.00027 if decay else 0.0} for decay,ps in groups.items()],
            betas=(0.948,0.999),eps=1e-8)


def training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()


@torch.no_grad()
def update_ema(ema,model,trainable):
    # 與 ModelEMA 同一 decay／浮點更新式，但不對固定 state 重複算術，守住 P3 分支。
    ema.updates+=1;decay=ema.decay(ema.updates)
    source=model.state_dict();destination=ema.ema.state_dict()
    for name in trainable:destination[name].mul_(decay).add_(source[name].detach(),alpha=1-decay)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--variant',choices=['control','rep'],required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if not a.name.replace('-','').isalnum():raise ValueError('name 格式不符')
    output=ROOT/'artifacts'/a.name;output.mkdir(parents=True,exist_ok=False)
    setup();data=prepare_coco();torch.set_num_threads(8);Harness.variant=a.variant
    trainer=Harness(overrides={'model':str(SOURCE/'weights/bittrue/a0.pt'),'data':str(data),
        'epochs':10,'batch':32,'nbs':128,'imgsz':640,'device':'0','workers':4,'amp':False,
        'optimizer':'AdamW','project':str(output),'name':'setup','exist_ok':False,'seed':20260920,
        'deterministic':True,'mosaic':0.0,'mixup':0.0,'cutmix':0.0,'copy_paste':0.0,'fliplr':0.5,
        'cache':False,'fraction':1.0,'plots':False,'save_json':False,'warmup_epochs':1.0,
        'save':False,'patience':4,'cos_lr':True,'close_mosaic':0})
    trainer._setup_train();model=trainer.model;model.criterion=model.init_criterion();training_mode(model)
    assert len(trainer.train_loader.dataset)==118287
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in active}
    initial={n:p.detach().clone() for n,p in active.items()}
    optimizer=trainer.optimizer;scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    ema=ModelEMA(model);ema.updates=7400
    base=json.loads((ROOT/'artifacts/a0-scope-late-v1/summary.json').read_text())['epochs'][-1]['ema']
    tracked=('coco/box/map50_95','coco/person/box/map50_95')
    report={'status':'running','variant':a.variant,'parent':str(PARENT),'parent_sha256':PARENT_SHA,
        'parent_state':'EMA E8 exploratory parent, not promoted winner','epochs':[],
        'scope':'only layer17 parameters; all other parameters and all BN statistics fixed',
        'lr':2e-6,'optimizer':'AdamW','fresh_optimizer':True,'warmup_epochs':1,'criterion_horizon':10,
        'epochs_budget':5,'physical_batch':32,'logical_batch':128,'ema_initial_updates':7400,
        'ema_policy':'native decay and arithmetic on trainable parameters only; fixed state exact',
        'trainable_parameters':sum(p.numel() for p in active.values()),'decision_metrics':list(tracked),
        'script_sha256':sha256(Path(__file__)),'deployment_validation':'folded Conv format'}
    trace=hashlib.sha256();total=0;parameters=list(active.values())
    with (output/'progress.jsonl').open('x') as log:
        for epoch in range(1 if a.smoke else 5):
            random.seed(epoch);np.random.seed(epoch);torch.manual_seed(epoch);torch.cuda.manual_seed_all(epoch)
            training_mode(model);pending=[];seen=macro=0;started=time.time()
            for index,batch in enumerate(trainer.train_loader):
                if total==0:
                    trace.update(json.dumps(batch['im_file']).encode())
                    for key in ('img','cls','bboxes','batch_idx'):trace.update(batch[key].contiguous().numpy().tobytes())
                pending.append(trainer.preprocess_batch(batch))
                if len(pending)<4 and index+1<len(trainer.train_loader):continue
                steps=math.ceil(len(trainer.train_loader)/4)
                cosine=.5+.5*(1+math.cos(math.pi*(epoch+macro/steps)/10))/2
                warm=min(1.0,.1+.9*(total+1)/steps)
                for group in optimizer.param_groups:group['lr']=group['initial_lr']*cosine*warm
                for attempt in range(16):
                    optimizer.zero_grad(set_to_none=True);losses=[]
                    for item in pending:
                        with torch.autocast('cuda',dtype=torch.float16):loss,_=model(item);loss=loss.sum()
                        if not torch.isfinite(loss):raise FloatingPointError('loss 非有限')
                        scaler.scale(loss).backward();losses.append(float(loss.detach()))
                    scaler.unscale_(optimizer)
                    if all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):break
                    scaler.update(new_scale=scaler.get_scale()/2)
                else:raise FloatingPointError('AMP／gradient 未恢復')
                if total==0:
                    report['first_macro_trace_sha256']=trace.hexdigest();report['first_macro_native_loss_sum']=sum(losses)
                    report['gradient_norms']={n:float(p.grad.float().norm()) for n,p in active.items()}
                    assert sum(report['gradient_norms'].values())>0
                    if a.variant=='rep':assert float(model.model[17].conv2.bn.weight.grad.abs().sum())>0
                torch.nn.utils.clip_grad_norm_(parameters,10);scaler.step(optimizer);scaler.update()
                update_ema(ema,model,active)
                if total==0:
                    delta=sum(float((p.detach()-initial[n]).float().square().sum()) for n,p in active.items())**.5
                    norm=sum(float(p.float().square().sum()) for p in initial.values())**.5
                    report['first_update_ratio']=delta/max(norm,1e-12)
                    assert 0<report['first_update_ratio']<.001
                total+=1;macro+=1;seen+=sum(b['img'].shape[0] for b in pending);pending=[]
                log.write(json.dumps({'epoch':epoch+1,'macro':macro,'loss_sum':sum(losses),'amp_retries':attempt,
                    'optimizer_steps':total})+'\n');log.flush()
                if a.smoke:break
            check_fixed(model,fixed);check_fixed(ema.ema,fixed)
            payload={'model':copy.deepcopy(model).cpu(),'ema':copy.deepcopy(ema.ema).cpu(),
                'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'ema_updates':ema.updates,
                'epoch':epoch+1,'optimizer_steps':total,'variant':a.variant,'parent_sha256':PARENT_SHA,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            torch.save(payload,output/f'epoch-{epoch+1:02d}-resume.pt');del payload
            if a.smoke:
                report.update(status='passed',smoke_images=seen,optimizer_steps=total,ema_updates=ema.updates)
                write_json(output/'summary.json',report);print('JOB_DONE REP17_SMOKE',a.variant,flush=True);return
            if seen!=118287:raise ValueError('完整訓練數量不符')
            metrics=validate(folded(ema.ema),'bittrue',data,output/f'epoch-{epoch+1:02d}-ema')
            live=validate(folded(model),'bittrue',data,output/f'epoch-{epoch+1:02d}-live')
            report['epochs'].append({'epoch':epoch+1,'images':seen,'macros':macro,'ema':metrics,'live':live,
                'elapsed_seconds':time.time()-started,'delta_parent':{k:metrics[k]-base[k] for k in base}})
            if any(metrics[k]-base[k]<-.005 for k in tracked):report['status']='paused_for_analysis';break
            write_json(output/'summary.json',report)
            if hasattr(model.criterion,'update'):model.criterion.update()
        if report['status']=='running':report['status']='complete'
        write_json(output/'summary.json',report)
    print('JOB_DONE REP17',report['status'],flush=True)


if __name__=='__main__':main()
