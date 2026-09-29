"""新學生 train-only 16 macro 配對；MuSGD 角色 LR 校準與安全確認。"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import MappingProxyType
from experiment import configure, ProbeSession, runtime
from teachers import HERE, SelectedSource, SELECTED_SHA
import torch
from yolo_combine.factory import FusionModelFactory
from yolo_combine.stage_policy import build_joint_optimizer
from yolo_combine.hardware_contract import HardwareContractGuard
from yolo_combine.joint_loss import MacroStepEngine, NativeTaskLossRouter
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
from yolo_combine.contracts import Task

OUT = HERE/'artifacts/optimizer-probe-v1'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm', choices=('adamw','mu-initial','mu-confirm'), required=True)
    arm=parser.parse_args().arm
    config,stage=configure()
    lrs=dict(stage.learning_rates)
    if arm!='adamw':
        lrs={k:v*100 for k,v in lrs.items()}
    if arm=='mu-confirm':
        a=json.loads((OUT/'adamw.json').read_text())
        b=json.loads((OUT/'mu-initial.json').read_text())
        assert a['trace_sha256']==b['trace_sha256']
        invalid={role:{'adamw':a['stats'][role]['median'],'mu_initial':b['stats'][role]['median']}
                 for role in lrs if a['stats'][role]['median']<=0 or b['stats'][role]['median']<=0}
        if invalid:
            with (OUT/'mu-confirm.json').open('x') as f:
                json.dump({'status':'recipe_rejected','recipe_accepted':False,'reason':'zero update median prevents finite role LR calibration',
                    'invalid_roles':invalid,'confirmation_updates':0,'trace_sha256':a['trace_sha256'],
                    'existing_successful_probes_preserved':True,'fallback':'AdamW'},f,indent=2)
            print('JOB_DONE: MuSGD recipe rejected; no confirmation GPU updates; use AdamW',flush=True)
            return
        for role in lrs:
            numerator=a['stats'][role]['median']; denominator=b['stats'][role]['median']
            assert numerator>0 and denominator>0
            lrs[role] *= numerator/denominator
    stage=replace(stage, learning_rates=MappingProxyType(lrs))
    runtime.formal.seed_everything(config.seed)
    session=ProbeSession(config,device='0',run_name='optimizer-'+arm+'-v1')
    session.run_dir.mkdir(parents=True,exist_ok=False)
    model=FusionModelFactory(SelectedSource(),detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint,
        checkpoint_kind='float').model.to('cuda:0')
    dl,pl,_=session._loaders(model)
    assert len(dl.loader.dataset)==118287 and len(pl.loader.dataset)==5964
    opt,grouping=build_joint_optimizer(model,stage,optimizer_name='AdamW' if arm=='adamw' else 'MuSGD',
        weight_decay=config.weight_decay,beta1=config.beta1,beta2=config.beta2)
    if arm!='adamw':
        assert all(p.ndim in (2,4) for g in opt.param_groups if g['use_muon'] for p in g['params'])
    fixed={n:t.detach().cpu().clone() for n,t in model.named_parameters() if not t.requires_grad}
    guard=HardwareContractGuard.capture(model)
    active=[(g['group_name'],g['role'],n,p) for g in opt.param_groups
            for n,p in zip(g['param_names'],g['params']) if p.requires_grad]
    eligible={}
    def capture(opt,args,kwargs):
        for _,_,n,p in active:
            eligible[n]=p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.ne(0).any())
    hook=opt.register_step_pre_hook(capture)
    losses=runtime.impl._AutocastTaskLossRouter(NativeTaskLossRouter(model,epochs=5,imgsz=640),
        device=torch.device('cuda:0'),enabled=True)
    ema=runtime.ActiveEMA(model,decay=.9999,tau=2000)
    engine=MacroStepEngine(model=model,losses=losses,optimizer=opt,
        reference_batch_size=config.reference_batch_size,
        task_weights={Task.DETECT:config.detect_weight,Task.POSE:config.pose_weight},
        scaler=torch.amp.GradScaler('cuda',init_scale=1024),ema=ema,max_grad_norm=10,max_amp_retries=16,
        preprocess=lambda t,b:dl.preprocess(b) if t is Task.DETECT else pl.preprocess(b))
    count=session.detect_microbatches_per_macro
    schedule=StageWarmupCosineScheduler(opt,stage='j3',epochs=5,steps_per_epoch=math.ceil(len(dl.loader)/count),
        warmup_epochs=1,warmup_start_factor=.1,final_lr_factor=.5)
    di,pi=iter(dl.loader),iter(pl.loader)
    trace=hashlib.sha256();samples={};reports=[]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(16):
        db=tuple(next(di) for _ in range(count));pb=(next(pi),)
        for batch in (*db,*pb):
            trace.update(json.dumps([Path(x).name for x in batch['im_file']]).encode())
            for k in ('img','bboxes','cls','keypoints','batch_idx'):
                if k in batch:
                    v=batch[k].detach().cpu().contiguous()
                    trace.update(k.encode());trace.update(str(tuple(v.shape)).encode());trace.update(v.numpy().tobytes())
        before={n:p.detach().clone() for _,_,n,p in active}
        eligible.clear();schedule.prepare_step()
        r=engine.run(detect_batches=db,pose_batches=pb);schedule.advance()
        reports.append(asdict(r))
        for group,role,n,p in active:
            scale=float(before[n].square().mean().sqrt())
            if not eligible.get(n) or scale<=1e-12:continue
            value=float((p.detach()-before[n]).square().mean().sqrt())/scale
            assert math.isfinite(value)
            samples.setdefault(group,[]).append(value);samples.setdefault(role,[]).append(value)
    hook.remove()
    for candidate in (model,ema.ema):
        guard.assert_unchanged(candidate)
        params=dict(candidate.named_parameters())
        assert all(torch.equal(v,params[n].detach().cpu()) for n,v in fixed.items())
    assert ema.updates==16
    result={'status':'probe_completed','arm':arm,'student_sha256':SELECTED_SHA,
        'role_lrs':lrs,'stats':{k:{'median':statistics.median(v),'count':len(v)} for k,v in samples.items()},
        'trace_sha256':trace.hexdigest(),'updates_discarded':True,'validation_used':False,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'grouping':asdict(grouping),'reports':reports}
    OUT.mkdir(exist_ok=True)
    if arm=='mu-confirm':
        a=json.loads((OUT/'adamw.json').read_text());assert result['trace_sha256']==a['trace_sha256']
        ratios={g:s['median']/a['stats'][g]['median'] for g,s in result['stats'].items()
                if '.' in g and a['stats'][g]['median']>0}
        result['group_update_ratios']=ratios
        result['recipe_accepted']=bool(ratios) and all(.5<=r<=2 for r in ratios.values())
    with (OUT/(arm+'.json')).open('x') as f:json.dump(result,f,indent=2)
    print('JOB_DONE: '+arm+' train-only optimizer probe completed',flush=True)

if __name__=='__main__':main()
