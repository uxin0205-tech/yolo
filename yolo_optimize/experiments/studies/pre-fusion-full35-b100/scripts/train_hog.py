"""融合前 W-CTRL5／W-HOG10：同起點原生 Detect loss 與 raw P3 training-only HOG。"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from common import ROOT,SOURCE,setup,prepare_coco,sha256,write_json
import torch
from ultralytics.utils.torch_utils import ModelEMA
from yolo_optimize.hog import HOGAuxiliary
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from qk_challenger import install,remove
from continue_a0 import ContinuationHarness,mode,check_fixed,validate

PARENT=ROOT/'artifacts/a0-scope-late-v1/epoch-08-resume.pt'
CALIBRATION=ROOT/'artifacts/prefusion-hog-calibration-v1/summary.json'


def role(name):
    if name.startswith('hog_aux.'):return 'hog'
    if name.startswith('model.23.'):return 'head'
    if any(name.startswith(f'model.{i}.') for i in (13,16,17,19,20,22)) and '.attn.' not in name:return 'neck'
    return 'fixed'


class Harness(ContinuationHarness):
    def get_model(self,cfg=None,weights=None,verbose=True):
        payload=torch.load(PARENT,map_location='cpu',weights_only=False)
        model=copy.deepcopy(payload['ema']).float()
        remove(model)
        if hasattr(model,'criterion'):del model.criterion
        convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/'configs/attention/float-pwl-final.yaml'))
        install(model)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260909)
            model.hog_aux=HOGAuxiliary(256)
        return model

    def build_optimizer(self,model,*args,**kwargs):
        groups={};rates={'head':2.5e-5,'neck':1e-5,'hog':3e-4}
        for name,p in model.named_parameters():
            r=role(name);p.requires_grad_(r!='fixed')
            if r=='fixed':continue
            decay=p.ndim>=2 and not name.endswith('.bias')
            groups.setdefault((r,decay),[]).append(p)
        return torch.optim.AdamW([{'params':ps,'lr':rates[r],'initial_lr':rates[r],'role':r,
            'weight_decay':0.00027 if decay else 0.0} for (r,decay),ps in groups.items()],
            betas=(0.948,0.999),eps=1e-8)


def fixed_state(model):
    parameters=dict(model.named_parameters())
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()
        if (n in parameters and not parameters[n].requires_grad)
        or (n not in parameters and not n.startswith('model.23.'))}


def parts(model,batch,mu,force=False):
    # 區域 hook 只活過本次 forward，finally 移除；不在 model 保存 feature／target。
    captured=[]
    handle=model.model[16].register_forward_hook(lambda _m,_i,out:captured.append(out))
    try:native,_=model(batch);native=native.sum()
    finally:handle.remove()
    if len(captured)!=1 or not captured[0].requires_grad:raise AssertionError('P3 seam 不可微或重複擷取')
    raw=captured.pop()
    auxiliary=model.hog_aux(raw,batch['img'],batch['bboxes'],batch['batch_idx']) if mu>0 or force else None
    return native if auxiliary is None else native+mu*auxiliary.loss_sum,native,auxiliary,raw


def multiplier(epoch,macro,macros):
    if epoch==1:return (macro+1)/macros
    return 1.0 if 2<=epoch<=7 else 0.0


def inference_graph(model):
    graph=copy.deepcopy(model)
    del graph.hog_aux
    if hasattr(graph,'criterion'):del graph.criterion
    return graph


def calibrate(model,trainer,output):
    squared_native=squared_hog=dot=0.0;count=valid=images_with_valid=0;ratios=[]
    parameters=[model.model[16].cv2.conv.weight,model.model[23].cv2[0][0].conv.weight]
    # 真實 callsite 的 mu=0 loss／代表性 Neck、head gradient 等價，不更新 optimizer。
    for index,batch in enumerate(trainer.train_loader):
        if index==8:break
        batch=trainer.preprocess_batch(batch)
        if index==0:
            buffers={n:b.detach().clone() for n,b in model.named_buffers()}
            with torch.autocast('cuda',dtype=torch.float16):native,_=model(batch);native=native.sum()
            gn0=torch.autograd.grad(native*1024,parameters)
            current=dict(model.named_buffers())
            for n,b in buffers.items():current[n].copy_(b)
            with torch.autocast('cuda',dtype=torch.float16):zero,_,aux,_=parts(model,batch,0.0)
            gz=torch.autograd.grad(zero*1024,parameters)
            assert aux is None and torch.equal(native.detach(),zero.detach())
            assert all(torch.equal(a,b) for a,b in zip(gn0,gz))
            assert not model.model[16]._forward_hooks
            for n,b in buffers.items():current[n].copy_(b)
            del native,zero,gn0,gz,buffers
        with torch.autocast('cuda',dtype=torch.float16):_,native,aux,raw=parts(model,batch,0.0,force=True)
        gn=torch.autograd.grad(native*1024,raw,retain_graph=True)[0].float()/1024
        gh=torch.autograd.grad(aux.loss_sum*1024,raw)[0].float()/1024
        sn=float(gn.double().square().sum());sh=float(gh.double().square().sum())
        if not (math.isfinite(sn+sh) and sn>0 and sh>0):raise AssertionError('HOG 校準梯度異常')
        squared_native+=sn;squared_hog+=sh;dot+=float((gn.double()*gh.double()).sum())
        ratios.append(math.sqrt(sh/sn));count+=batch['img'].shape[0]
        valid+=int(aux.targets.valid_mask.sum());images_with_valid+=int(aux.targets.valid_mask.flatten(1).any(1).sum())
    mu=0.05*math.sqrt(squared_native/squared_hog)
    scaled=[mu*x for x in ratios]
    if not .02<=float(np.median(scaled))<=.10:raise ValueError('HOG gradient ratio 校準未通過')
    report={'status':'passed','mu':mu,'target_ratio':0.05,'median_ratio':float(np.median(scaled)),
        'per_microbatch_ratios':scaled,'gradient_cosine':dot/math.sqrt(squared_native*squared_hog),
        'images':count,'valid_cells':valid,'images_with_valid_cells':images_with_valid,'optimizer_steps':0,
        'mu_zero_loss_and_gradient_exact':True,'gradient_space':'raw P3, batch-summed native and HOG',
        'parent_sha256':sha256(PARENT),'validation_used':False,'aux_parameters':2313}
    if count!=256:raise AssertionError('校準必須使用固定 256 張 train prefix')
    write_json(output/'summary.json',report)
    print('JOB_DONE HOG_CALIBRATION',json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--arm',choices=['control','hog','calibrate'],required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not a.name.replace('-','').isalnum():raise ValueError('name 格式不符')
    output=ROOT/'artifacts'/a.name;output.mkdir(parents=True,exist_ok=False)
    setup();data=prepare_coco();torch.set_num_threads(8)
    trainer=Harness(overrides={'model':str(SOURCE/'weights/bittrue/a0.pt'),'data':str(data),
        'epochs':10,'batch':32,'nbs':128,'imgsz':640,'device':'0','workers':4,'amp':False,
        'optimizer':'AdamW','project':str(output),'name':'setup','exist_ok':False,'seed':20260919,
        'deterministic':True,'mosaic':0.0,'mixup':0.0,'cutmix':0.0,'copy_paste':0.0,'fliplr':0.5,
        'cache':False,'fraction':1.0,'plots':False,'save_json':False,'warmup_epochs':1.0,
        'save':False,'patience':4,'cos_lr':True,'close_mosaic':0})
    trainer._setup_train();model=trainer.model;model.criterion=model.init_criterion();mode(model)
    assert len(trainer.train_loader.dataset)==118287
    fixed=fixed_state(model);aux_initial={n:v.detach().clone() for n,v in model.hog_aux.state_dict().items()}
    if a.arm=='calibrate':calibrate(model,trainer,output);return
    calibration=json.loads(CALIBRATION.read_text())
    if calibration['status']!='passed' or calibration['parent_sha256']!=sha256(PARENT):raise ValueError('校準 lineage 不符')
    optimizer=trainer.optimizer;scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    ema=ModelEMA(model);ema.updates=7400
    base=json.loads((ROOT/'artifacts/late-e8-inference-verification/summary.json').read_text())['metrics']
    tracked=tuple(base);total=0;stale=0;best=dict(base)
    report={'status':'running','arm':a.arm,'parent':str(PARENT),'parent_sha256':sha256(PARENT),
        'parent_state':'EMA E8, exploratory parent; not promoted winner','epochs':[],
        'fresh_optimizer':True,'ema_initial_updates':7400,'warmup_epochs':1,'criterion_horizon':10,
        'physical_batch':32,'logical_batch':128,'mu':calibration['mu'],'patience':4,
        'training_scope':'Neck except attention parameters + Detect head + separate training-only HOG head',
        'non_head_bn_statistics':'fixed','decision_metrics':list(tracked),'script_sha256':sha256(Path(__file__)),
        'native_head_lr':2.5e-5,'neck_lr':1e-5,'hog_lr':3e-4,'loss_reduction':'batch_sum',
        'hog_mask':'all80 GT cell-center hard union; nonzero gradient energy validity; uniform valid-cell mean',
        'smoke_forces_hog_when_requested':a.smoke and a.arm=='hog'}
    trace=hashlib.sha256();params=[v for v in model.parameters() if v.requires_grad]
    with (output/'progress.jsonl').open('x') as log:
        for epoch in range(1 if a.smoke else 5 if a.arm=='control' else 10):
            random.seed(epoch);np.random.seed(epoch);torch.manual_seed(epoch);torch.cuda.manual_seed_all(epoch)
            mode(model);pending=[];seen=macro=0;started=time.time()
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
                mu=calibration['mu']*(1.0 if a.smoke else multiplier(epoch,macro,steps)) if a.arm=='hog' else 0.0
                buffers={n:b.detach().clone() for n,b in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
                for attempt in range(16):
                    optimizer.zero_grad(set_to_none=True)
                    if attempt:
                        current=dict(model.named_buffers())
                        for n,b in buffers.items():current[n].copy_(b)
                    losses=[];aux_loss=0.0
                    for item in pending:
                        with torch.autocast('cuda',dtype=torch.float16):loss,_,aux,_=parts(model,item,mu)
                        if not torch.isfinite(loss):raise FloatingPointError('loss 非有限')
                        scaler.scale(loss).backward();losses.append(float(loss.detach()))
                        if aux is not None:aux_loss+=float(aux.loss_sum.detach())
                    scaler.unscale_(optimizer)
                    if all(v.grad is None or torch.isfinite(v.grad).all() for v in params):break
                    scaler.update(new_scale=scaler.get_scale()/2)
                else:raise FloatingPointError('AMP replay 未恢復')
                if total==0:
                    report['first_macro_trace_sha256']=trace.hexdigest()
                    report['p3_producer_grad_norm']=float(model.model[16].cv2.conv.weight.grad.float().norm())
                    report['hog_grad_norm']=None if model.hog_aux.projection.weight.grad is None else float(model.hog_aux.projection.weight.grad.float().norm())
                    assert report['p3_producer_grad_norm']>0
                    if mu>0:assert report['hog_grad_norm'] is not None and report['hog_grad_norm']>0
                    else:assert report['hog_grad_norm'] is None
                torch.nn.utils.clip_grad_norm_(params,10);scaler.step(optimizer);scaler.update();ema.update(model)
                total+=1;macro+=1;seen+=sum(b['img'].shape[0] for b in pending);pending=[]
                log.write(json.dumps({'epoch':epoch+1,'macro':macro,'mu':mu,'loss_sum':sum(losses),
                    'auxiliary_loss_sum':aux_loss,'amp_retries':attempt,'optimizer_steps':total})+'\n');log.flush()
                if a.smoke:break
            check_fixed(model,fixed)
            if a.arm=='control':assert all(torch.equal(v,model.hog_aux.state_dict()[n]) for n,v in aux_initial.items())
            payload={'model':copy.deepcopy(model).cpu(),'ema':copy.deepcopy(ema.ema).cpu(),
                'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'ema_updates':ema.updates,
                'epoch':epoch+1,'optimizer_steps':total,'arm':a.arm,'parent_sha256':sha256(PARENT),
                'calibration':calibration,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            torch.save(payload,output/f'epoch-{epoch+1:02d}-resume.pt');del payload
            if a.smoke:
                report.update(status='passed',optimizer_steps=total,smoke_images=seen,ema_updates=ema.updates)
                write_json(output/'summary.json',report);print('JOB_DONE HOG_SMOKE',a.arm,flush=True);return
            if seen!=118287:raise ValueError('完整訓練數量不符')
            metrics=validate(inference_graph(ema.ema),'bittrue',data,output/f'epoch-{epoch+1:02d}-ema')
            live=validate(inference_graph(model),'bittrue',data,output/f'epoch-{epoch+1:02d}-live')
            entry={'epoch':epoch+1,'images':seen,'macros':macro,'ema':metrics,'live':live,
                'elapsed_seconds':time.time()-started,'delta_parent':{k:metrics[k]-base[k] for k in tracked}}
            report['epochs'].append(entry)
            if any(entry['delta_parent'][k]<-.005 for k in tracked):report['status']='paused_for_analysis';break
            if any(metrics[k]>best[k]+.0001 for k in tracked):
                best={k:max(best[k],metrics[k]) for k in tracked};stale=0
            else:stale+=1
            write_json(output/'summary.json',report)
            if a.arm=='hog' and stale>=4:report['status']='patience_stop';break
            if hasattr(model.criterion,'update'):model.criterion.update()
        if report['status']=='running':report['status']='complete'
        write_json(output/'summary.json',report)
    print('JOB_DONE',report['status'],flush=True)


if __name__=='__main__':main()
