"""三個已改方法候選各 5 epoch；原生 QK 重用已完成 B5，不加訓 control。"""
import argparse
import copy
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch import nn
from models import *
from yolo_combine.joint_loss import MacroStepEngine,NativeTaskLossRouter
from yolo_combine.joint_data import TaskLoaderSettings,build_task_loader
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
from yolo_combine._formal_training_impl import _AutocastTaskLossRouter,seed_everything,reseed_loader_for_epoch
from yolo_combine.resume import TrainingProgress,save_training_snapshot,load_training_snapshot,save_inference_weights
from yolo_combine.contracts import Task
from yolo_combine.data import prepare_bbt5_view
from common import prepare_coco
sys.path.insert(0,str(HERE.parents[1]))
from tools.restructure_layout import edit
CFG=json.loads((HERE/'config.json').read_text())
BASE=json.loads((b.RUN/'analysis/summary.json').read_text())['e5']

class Guard:
    def __init__(self,model):
        active={n for n,p in model.named_parameters() if p.requires_grad}
        self.fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in active}
    def check(self,model):
        state=model.state_dict()
        assert all(torch.equal(state[n].detach().cpu(),v) for n,v in self.fixed.items()),'固定 state 改變'
        assert not any(m.training for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))

def finite(metrics,*args,**kwargs):
    assert set(metrics)==set(BASE) and all(math.isfinite(v) for v in metrics.values())

# 僅修改本子程序的檢查政策；新 Attention／Rep 會影響 COCO，不能用 B-only 的逐位不變閘。
b.metric_guard=finite

def optimizer_for(model,arm):
    groups={};modules=dict(model.named_modules())
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        role='heads'
        if '.attn.' in n:role='scale' if n.endswith('.score.coefficients') else 'bias' if '.attn.bias.' in n else 'attention'
        if arm.startswith('rep') and n.startswith(f'graph.model.{int(arm[3:])}.'):role='rep'
        no_decay=n.endswith('.bias') or '.attn.bias.' in n or role=='scale' or isinstance(modules[n.rsplit('.',1)[0]],nn.modules.batchnorm._BatchNorm)
        key=role+('/no_decay' if no_decay else '/decay')
        group=groups.setdefault(key,dict(params=[],param_names=[],group_name=key,role=role,lr=CFG['lrs'][role],weight_decay=0 if no_decay else CFG['weight_decay']))
        group['params'].append(p);group['param_names'].append(n)
    return torch.optim.AdamW(list(groups.values()),betas=tuple(CFG['betas']))

def constants(model):
    result={}
    for name in SITES:
        m=model.graph.get_submodule(name)
        if not hasattr(m,'score'):continue
        result[name]={'scale_codes':(m.score.quantized()*1024).int().tolist(),
            'bias_x_codes':(m.bias.table_x.detach().clamp(-32,32-1/1024)*1024).round().int().tolist(),
            'bias_y_codes':(m.bias.table_y.detach().clamp(-32,32-1/1024)*1024).round().int().tolist(),
            'denominator':1024,'bias_signed_bits':16}
    return result

def check_safety(metrics,e0):
    assert all(metrics[k]>=v-.05 for k,v in e0.items() if k.endswith('/map50_95')),'AP 大幅退化，已存檔，停止查因'

def validate_copy(model,source,out,epoch,kind='bittrue'):
    attempt=1
    while (out/f'attempt-{attempt}').exists():attempt+=1
    return b.validate(folded(model),source,out/f'attempt-{attempt}',epoch,kind)

def epoch_finish(out,epoch,model,source,guard,e0):
    dest=out/f'epochs/e{epoch+1}';dest.mkdir(parents=True,exist_ok=True)
    snapshot=out/f'checkpoints/epoch-{epoch+1:02d}.pt'
    data=torch.load(snapshot,map_location='cpu',weights_only=True)
    assert data['progress']['next_epoch']==epoch+1 and data['resolved_config']==CFG
    candidate=copy.deepcopy(model).cpu().eval();candidate.load_state_dict(data['ema_state'],strict=True);guard.check(candidate)
    deploy=folded(candidate)
    if not (dest/'inference.pt').exists():
        save_inference_weights(dest/'inference.pt',model=deploy,use_ema=False,metadata={'epoch':epoch,'arm':source.variant,
            'parent_sha256':PARENT_SHA,'full_resume_sha256':b.sha256(snapshot),'rep_folded_for_inference':source.variant.startswith('rep')})
    exported=torch.load(dest/'inference.pt',map_location='cpu',weights_only=True)['state_dict']
    assert set(exported)==set(deploy.state_dict()) and all(torch.equal(v,exported[n]) for n,v in deploy.state_dict().items())
    if not (dest/'constants.json').exists():b.save(dest/'constants.json',constants(candidate))
    if not (dest/'metrics.json').exists():b.save(dest/'metrics.json',validate_copy(candidate,source,dest/'validation',epoch))
    metrics=json.loads((dest/'metrics.json').read_text());finite(metrics);check_safety(metrics,e0)
    return metrics

def report(out,arm,metrics,verification):
    keys=[k for k in BASE if k.endswith('/map50_95')];end=metrics['5']
    best=max(range(1,6),key=lambda i:metrics[str(i)]['coco/box/map50_95'])
    gate=all(end[k]>=BASE[k]-.001 for k in keys) and (end['coco/box/map50_95']>=BASE['coco/box/map50_95']+.001 or end['bbat/pose/map50_95']>=BASE['bbat/pose/map50_95']+.002)
    lines=[f'# {arm}：5 epoch 與獨立驗證結果','',
        '原生 QK＋PWL 參考是已完成 MASF B5，沒有重跑未改架構的加訓對照。所有差值不能分離結構與適應訓練的收益。','',
        '| AP50–95 (%) | 原生 QK B5 | 候選 E5 | 差值 (pp) |','| --- | ---: | ---: | ---: |']
    for k in keys:lines.append(f'| {k} | {100*BASE[k]:.4f} | {100*end[k]:.4f} | {100*(end[k]-BASE[k]):+.4f} |')
    lines+=['',f'工程参考門檻：{"通過，僅列候選" if gate else "未通過，不建議取代"}。未自動升版；不增加 epoch。',
        '',f'COCO AP 最佳回合是 E{best}，僅作補充；主要比較仍固定 E5。',
        '', 'Float 與 BitTrue、逐回合完整指標、scale/bias 整數碼及完整續訓 checkpoint 均保存在本目錄。',
        '', '硬體限制：此處未量測實板 latency／energy。Rep17/20 以已 fold 的單 Conv 作推論；scale/bias 固定常數，無逐圖片估計尺度；PWL[-10,0]20段，最後正規化仍是軟體除法参考。']
    edit(out/'RESULTS.md','\n'.join(lines))
    b.save(out/'summary.json',{'status':'completed','arm':arm,'metrics_by_epoch':metrics,'independent_validation':verification,
        'best_coco_epoch':best,'gate_passed':gate,'no_auto_promote':True,'parent_sha256':PARENT_SHA,
        'report':str(out/'RESULTS.md'),'inference_sha256':b.sha256(out/'epochs/e5/inference.pt')})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=ARMS);parser.add_argument('phase',choices=['smoke','train']);args=parser.parse_args()
    b.initialize();torch.set_num_threads(4);seed_everything(CFG['seed'])
    assert json.loads((b.RUN/'final-audit.json').read_text())['status']=='completed'
    assert json.loads((HERE/'artifacts/preflight-v1.json').read_text())['status']=='passed'
    from activation_lab.training.full35 import _fp32_bbox_iou
    import ultralytics.utils.loss as loss_module
    loss_module.bbox_iou=_fp32_bbox_iou(loss_module.bbox_iou)
    out=HERE/'artifacts'/args.arm/args.phase;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():assert json.loads((out/'summary.json').read_text())['status']=='completed';return
    model,source=build(args.arm);guard=Guard(model);device=torch.device('cuda:0');b.move(model,device)
    dl=build_task_loader(model,data_yaml=prepare_coco(),settings=TaskLoaderSettings.for_detect(batch_size=16,workers=4,seed=1),device=device)
    view=prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',HERE/'artifacts/datasets/bbat5-v1-runtime')
    pl=build_task_loader(model,data_yaml=view.yaml,settings=TaskLoaderSettings.for_pose(batch_size=16,workers=4,seed=1),device=device)
    assert len(dl.dataset)==118287 and len(pl.dataset)==5964 and math.ceil(len(dl.loader)/16)==463
    optimizer=optimizer_for(model,args.arm);ema=b.PoseEMA(model)
    criteria=_AutocastTaskLossRouter(NativeTaskLossRouter(model,epochs=5,imgsz=640),device=device,enabled=True)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    scheduler=StageWarmupCosineScheduler(optimizer,stage=args.arm,epochs=5,steps_per_epoch=463,warmup_epochs=1,final_lr_factor=.5,warmup_start_factor=.1)
    bundle=dict(optimizer=optimizer,ema=ema,criteria=criteria,scaler=scaler,scheduler=scheduler)
    engine=MacroStepEngine(model=model,losses=criteria,optimizer=optimizer,reference_batch_size=64,
        task_weights={Task.DETECT:1,Task.POSE:.25},scaler=scaler,ema=ema,max_amp_retries=16,max_grad_norm=10,
        preprocess=lambda task,batch:dl.preprocess(batch) if task is Task.DETECT else pl.preprocess(batch))
    if not (out/'resolved-config.json').exists():b.save(out/'resolved-config.json',CFG)
    if args.phase=='train':
        assert json.loads((out.parent/'smoke/summary.json').read_text())['status']=='completed'
        if not (out/'e0.json').exists():
            if args.arm.startswith('rep'):b.save(out/'e0.json',BASE) # CPU 初始等價通過，重用同權重已完成驗證。
            else:b.save(out/'e0.json',validate_copy(model,source,out/'e0-validation',0))
        e0=json.loads((out/'e0.json').read_text());start=0
        snapshots=sorted((out/'checkpoints').glob('epoch-*.pt'))
        if snapshots:
            restored=load_training_snapshot(snapshots[-1],model=model,**bundle)
            assert restored.resolved_config==CFG;start=restored.progress.next_epoch
            guard.check(model);guard.check(ema.ema)
            for i in range(start):epoch_finish(out,i,model,source,guard,e0)
    else:start=0
    observed=[];coverage=[];coverage_hooks=[]
    if args.phase=='smoke' and args.arm=='scale_bias':
        for site in SITES:
            def capture(module,inputs,site=site):
                q,k=inputs
                coverage.append({'site':site,'q_ste_fraction':float((q.detach().abs()<=1).float().mean()),
                    'k_ste_fraction':float((k.detach().abs()<=1).float().mean()),
                    'q_positive_fraction':float((q.detach()>=0).float().mean()),
                    'k_positive_fraction':float((k.detach()>=0).float().mean())})
            coverage_hooks.append(model.graph.get_submodule(site).score.register_forward_pre_hook(capture))
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and ('.attn.' in n or n.startswith(('graph.model.17.','graph.model.20.')))}
    if args.phase=='smoke':
        def hook(opt,a,k):
            grads={n:float(p.grad.abs().max()) for n,p in model.named_parameters() if n in before and p.grad is not None}
            assert all(math.isfinite(v) for v in grads.values())
            if args.arm=='scale_bias':
                for site in SITES:
                    for suffix in ('.qkv.q.conv.weight','.qkv.k.conv.weight','.score.coefficients','.bias.table_x','.bias.table_y'):
                        assert grads['graph.'+site+suffix]>0,'task gradient 未到達 '+site+suffix
            else:assert any(v>0 for n,v in grads.items() if '.conv2.bn.weight' in n)
            observed.append(grads)
        h=optimizer.register_step_pre_hook(hook)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,1 if args.phase=='smoke' else 5):
        configure(model,args.arm)
        reseed_loader_for_epoch(dl.loader,seed=1,epoch=epoch,offset=0);reseed_loader_for_epoch(pl.loader,seed=1,epoch=epoch,offset=1)
        pi=iter(pl.loader);groups=b.batches_of(dl.loader,16);reports=[];begin=time.monotonic()
        if args.phase=='smoke':groups=itertools.islice(groups,2)
        for group in groups:
            try:pose=next(pi)
            except StopIteration:pi=iter(pl.loader);pose=next(pi)
            scheduler.prepare_step();reports.append(asdict(engine.run(detect_batches=group,pose_batches=(pose,))))
            scheduler.advance();(out/'heartbeat').touch()
        guard.check(model);guard.check(ema.ema)
        if args.phase=='smoke':
            h.remove()
            for handle in coverage_hooks:handle.remove()
            assert len(observed)==2
            changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in model.named_parameters() if n in before}
            assert any(v>0 for v in changes.values())
            if args.arm=='scale_bias':
                for site in SITES:
                    for suffix in ('.qkv.q.conv.weight','.qkv.k.conv.weight','.score.coefficients'):
                        assert changes['graph.'+site+suffix]>0
            b.save(out/'summary.json',{'status':'completed','arm':args.arm,'gradients':observed,'parameter_updates':changes,
                'constants_after':constants(ema.ema),'ste_coverage':coverage,'guard_passed':True,'reports':reports,'formal_epoch':False,
                'parent_sha256':PARENT_SHA,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            print('JOB_DONE smoke '+args.arm,flush=True);return
        assert len(reports)==463 and sum(r['detect_images'] for r in reports)==118287
        engine.advance_epoch((Task.DETECT,Task.POSE))
        checkpoint=out/f'checkpoints/epoch-{epoch+1:02d}.pt';assert not checkpoint.exists()
        save_training_snapshot(checkpoint,model=model,**bundle,progress=TrainingProgress(args.arm,epoch+1,scheduler.current_step,epoch+1),
            resolved_config=CFG,provenance=source.provenance(),loader_state={'reports':reports,'train_seconds':time.monotonic()-begin,'epoch_end':True},best_state={})
        epoch_finish(out,epoch,model,source,guard,e0)
    metrics={str(i):json.loads((out/f'epochs/e{i}/metrics.json').read_text()) for i in range(1,6)}
    verification={}
    # 用 E5 匯出部署圖重新載入，Rep 原始多分支快照保持不動。
    if args.arm.startswith('rep'):model,_=build('native_qk_reference')
    model.cpu().eval();model.load_state_dict(torch.load(out/'epochs/e5/inference.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    for kind in ('bittrue','float'):
        path=out/f'final-{kind}.json'
        if path.exists():verification[kind]=json.loads(path.read_text());continue
        verification[kind]=validate_copy(model,source,out/f'final-{kind}',4,kind)
        if kind=='bittrue':assert all(abs(v-metrics['5'][k])<=1e-8 for k,v in verification[kind].items())
        b.save(path,verification[kind])
    report(out,args.arm,metrics,verification)
    print('JOB_DONE '+args.arm+' 5 epoch 與完整分析',flush=True)

if __name__=='__main__':main()
