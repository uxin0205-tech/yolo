"""分段恢復第一階段：未加 MASF 的 A0，原生 loss 的成對 QK 梯度實驗。

使用原生 Detect loader／criterion，不呼叫舊融合 runner；每 macro 固定 4×32。
"""
import argparse
import copy
import hashlib
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from common import ROOT,SOURCE,write_json,sha256,setup,prepare_coco
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import ModelEMA
from yolo_attention.config import VariantConfig,NormalizationKind
from yolo_attention.integration import convert_yolo26_model
from qk_challenger import install,remove,TrainableBinaryScore
from validate import InternalValidator

PARENT=SOURCE/'weights/bittrue/a0.pt'
PARENT_SHA='c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123'


def role(name):
    if '.attn.qkv.q.' in name or '.attn.qkv.k.' in name:return 'qk'
    if name.startswith('model.23.'):return 'head'
    return 'fixed'


def mode(model):
    model.train()
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and not name.startswith('model.23.'):
            module.eval()


class Harness(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=copy.deepcopy(weights).float()
        config=VariantConfig.from_yaml(SOURCE/'configs/attention/float-pwl-final.yaml')
        convert_yolo26_model(model,config)
        if self.challenger:install(model)
        return model

    def build_optimizer(self,model,*args,**kwargs):
        groups={}
        for name,p in model.named_parameters():
            r=role(name);p.requires_grad_(r!='fixed')
            if r=='fixed':continue
            decay=p.ndim>=2 and not name.endswith('.bias')
            groups.setdefault((r,decay),[]).append(p)
        return torch.optim.AdamW([{'params':ps,'lr':5e-7 if r=='qk' else 2.5e-5,
            'initial_lr':5e-7 if r=='qk' else 2.5e-5,'role':r,
            'weight_decay':0.00027 if decay else 0.0} for (r,decay),ps in groups.items()],
            betas=(0.948,0.999),eps=1e-8)


def snapshot_fixed(model):
    parameters=dict(model.named_parameters())
    return {name:v.detach().cpu().clone() for name,v in model.state_dict().items()
            if (name in parameters and not parameters[name].requires_grad)
            or (name not in parameters and not name.startswith('model.23.'))}


def check_fixed(model,fixed):
    state=model.state_dict()
    changed=[k for k,v in fixed.items() if not torch.equal(v,state[k].detach().cpu())]
    if changed:raise AssertionError(f'固定 state 改變：{changed[:5]}')


def validate(model,backend,data,output):
    graph=copy.deepcopy(model).float()
    remove(graph)
    if hasattr(graph,'criterion'):del graph.criterion
    if backend=='bittrue':
        config=VariantConfig.from_yaml(SOURCE/'configs/attention/bittrue-pwl-final.yaml')
        convert_yolo26_model(graph,config)
    wrapper=YOLO(str(PARENT));wrapper.model=graph.eval()
    results=wrapper.val(validator=InternalValidator,data=str(data),imgsz=640,batch=32,
        device='0',workers=4,split='val',save_json=False,plots=False,half=False,
        project=str(output),name=backend,exist_ok=False)
    maps=results.box.maps
    if len(InternalValidator.last_instance.dataloader.dataset)!=5000:raise ValueError('val 數量不符')
    metrics={'coco/box/map50_95':float(results.box.map),'coco/person/box/map50_95':float(maps[0]),
        'coco/ball/box/map50_95':float(maps[32]),'coco/bat/box/map50_95':float(maps[34])}
    if not all(math.isfinite(x) for x in metrics.values()):raise ValueError('AP 非有限')
    InternalValidator.last_instance=None
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['control','qk'],required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.name.replace('-','').isalnum():raise ValueError('name 格式不符')
    output=ROOT/'artifacts'/args.name;output.mkdir(parents=True,exist_ok=False)
    if sha256(PARENT)!=PARENT_SHA:raise ValueError('A0 雜湊不符')
    setup();data=prepare_coco()
    torch.set_num_threads(8)
    # 不呼叫 upstream check_amp（可能下載外部模型）；下方實際 loss smoke 直接檢查 AMP。
    Harness.challenger=args.arm=='qk'
    trainer=Harness(overrides={'model':str(PARENT),'data':str(data),'epochs':10,'batch':32,
        'nbs':128,'imgsz':640,'device':'0','workers':4,'amp':False,'optimizer':'AdamW',
        'project':str(output),'name':'setup','exist_ok':False,'seed':0,'deterministic':True,
        'mosaic':0.0,'mixup':0.0,'cutmix':0.0,'copy_paste':0.0,'fliplr':0.5,
        'cache':False,'fraction':1.0,'plots':False,'save_json':False,'warmup_epochs':1.0,
        'save':False,'patience':4,'cos_lr':True,'close_mosaic':0})
    trainer._setup_train()
    model=trainer.model
    model.criterion=model.init_criterion()
    mode(model)
    if len(trainer.train_loader.dataset)!=118287:raise ValueError('train 數量不符')
    fixed=snapshot_fixed(model)
    optimizer=trainer.optimizer
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    ema=ModelEMA(model)
    parent=json.loads((ROOT/'artifacts/baseline-a0/summary.json').read_text())['metrics']
    report={'status':'running','arm':args.arm,'parent':str(PARENT),'parent_sha256':PARENT_SHA,
        'fresh_optimizer':True,'fresh_ema_updates':0,'criterion_horizon':10,'warmup_epochs':1,
        'physical_batch':32,'logical_batch':128,'train_images':118287,'epochs':[],
        'qk_lr':5e-7,'head_lr':2.5e-5,'fixed_tensors':len(fixed),
        'training_scope':'Q/K projection + Detect head; all non-head BN statistics fixed',
        'loss_reduction':'native batch-sum accumulated over four microbatches; partial final macro retained',
        'challenger_contract':'exact XNOR forward; clipped signed-dot surrogate backward only in qk arm'}
    total_steps=0;best=parent['coco/box/map50_95'];best_person=parent['coco/person/box/map50_95'];stale=0
    first_macro_trace=hashlib.sha256()
    with (output/'progress.jsonl').open('x') as events:
        def emit(payload):
            events.write(json.dumps({'time_unix':time.time(),**payload})+'\n');events.flush()
        for epoch in range(1 if args.smoke else 10):
            random.seed(epoch);np.random.seed(epoch);torch.manual_seed(epoch);torch.cuda.manual_seed_all(epoch)
            mode(model);pending=[];seen=0;macro=0
            started=time.time()
            for index,batch in enumerate(trainer.train_loader):
                if total_steps==0:
                    first_macro_trace.update(json.dumps(batch['im_file']).encode())
                    for key in ('img','cls','bboxes','batch_idx'):
                        first_macro_trace.update(batch[key].contiguous().numpy().tobytes())
                pending.append(trainer.preprocess_batch(batch))
                if len(pending)<4 and index+1<len(trainer.train_loader):continue
                # 925 macros/epoch；尾批保留，warmup 精確佔第一 epoch。
                steps_per_epoch=math.ceil(len(trainer.train_loader)/4)
                position=(epoch+macro/steps_per_epoch)/10
                cosine=0.5+0.5*(1+math.cos(math.pi*position))/2
                warm=min(1.0,0.1+0.9*(total_steps+1)/steps_per_epoch)
                for group in optimizer.param_groups:group['lr']=group['initial_lr']*cosine*warm
                buffers={n:v.detach().clone() for n,v in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
                for attempt in range(16):
                    optimizer.zero_grad(set_to_none=True)
                    if attempt:
                        current=dict(model.named_buffers())
                        for n,v in buffers.items():current[n].copy_(v)
                    losses=[]
                    for item in pending:
                        with torch.autocast('cuda',dtype=torch.float16):
                            loss,parts=model(item);loss=loss.sum()
                        if not torch.isfinite(loss):raise FloatingPointError('loss 非有限')
                        scaler.scale(loss).backward();losses.append(float(loss.detach()))
                    scaler.unscale_(optimizer)
                    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    if finite:break
                    scaler.update(new_scale=scaler.get_scale()/2)
                else:raise FloatingPointError('16 次 AMP retry 未恢復')
                qk_grad=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters()
                    if role(n)=='qk' and p.grad is not None)**0.5
                if args.arm=='qk' and not qk_grad>0:raise AssertionError('challenger QK gradient 未接通')
                if args.arm=='control' and qk_grad!=0:raise AssertionError('control 梯度契約改變')
                if total_steps==0:
                    report['qk_projection_gradient_norms']={n:None if p.grad is None else float(p.grad.float().norm())
                        for n,p in model.named_parameters() if role(n)=='qk'}
                    report['first_macro_trace_sha256']=first_macro_trace.hexdigest()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                scaler.step(optimizer);scaler.update();ema.update(model)
                total_steps+=1;macro+=1;images=sum(x['img'].shape[0] for x in pending);seen+=images
                emit({'kind':'macro','epoch':epoch+1,'macro':macro,'images':images,'qk_grad_norm':qk_grad,
                    'loss_sum':sum(losses),'amp_retries':attempt,'optimizer_steps':total_steps})
                pending=[]
                if args.smoke:break
            check_fixed(model,fixed)
            checkpoint={'model':copy.deepcopy(model).cpu(),'ema':copy.deepcopy(ema.ema).cpu(),
                'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'ema_updates':ema.updates,
                'epoch':epoch+1,'optimizer_steps':total_steps,'arm':args.arm,'parent_sha256':PARENT_SHA,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            torch.save(checkpoint,output/f'epoch-{epoch+1:02d}-resume.pt')
            del checkpoint
            if args.smoke:
                report['qk_optimizer_state_count']=sum(p in optimizer.state for n,p in model.named_parameters() if role(n)=='qk')
                report.update(status='passed',smoke_images=seen,optimizer_steps=total_steps,qk_grad_norm=qk_grad)
                write_json(output/'summary.json',report);print('JOB_DONE smoke',report['qk_grad_norm'],flush=True);return
            if seen!=118287:raise ValueError(f'epoch 數量不符 {seen}')
            metrics=validate(ema.ema,'bittrue',data,output/f'epoch-{epoch+1:02d}-ema')
            live=validate(model,'bittrue',data,output/f'epoch-{epoch+1:02d}-live')
            entry={'epoch':epoch+1,'images':seen,'macros':macro,'elapsed_seconds':time.time()-started,
                'ema':metrics,'live':live,'delta_parent':{k:metrics[k]-v for k,v in parent.items()}}
            report['epochs'].append(entry);emit({'kind':'epoch_complete',**entry})
            if any(v<-.005 for v in entry['delta_parent'].values()):
                report['status']='paused_for_analysis';write_json(output/'summary.json',report);break
            score=metrics['coco/box/map50_95']
            person=metrics['coco/person/box/map50_95']
            protected=all(v>=-.001 for v in entry['delta_parent'].values())
            if protected and (score>best+0.0001 or person>best_person+0.0001):
                best=max(best,score);best_person=max(best_person,person);stale=0
            else:stale+=1
            write_json(output/'summary.json',report)
            if stale>=4:report['status']='patience_stop';break
            if hasattr(model.criterion,'update'):model.criterion.update()
        if report['status']=='running':report['status']='complete'
        write_json(output/'summary.json',report)
    print('JOB_DONE',report['status'],flush=True)


if __name__=='__main__':main()
