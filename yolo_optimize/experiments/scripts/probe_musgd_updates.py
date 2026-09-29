#!/usr/bin/env python3
"""原計畫 O 前置：同 PSEL／heads scope 的16 macro train-only更新校準。"""
import gc
import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from yolo_optimize import runtime
from yolo_optimize.training import _build,_raw_macros,_reseed,recovery_scope
from yolo_combine.stage_policy import build_joint_optimizer
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
import torch


def run_probe(config,data,root,label,lrs=None):
    state=_build(config,runtime.FINAL_ROOT/'weights/combined/inference/best_joint.pt',
        runtime.FINAL_ROOT/'weights/combined/full-resume/best_joint.pt',data,32,'heads','cuda:0',ema_age_mode='parent')
    if lrs is not None:
        scope=recovery_scope('heads')
        scope=replace(scope,learning_rates={**scope.learning_rates,**lrs})
        optimizer,_=build_joint_optimizer(state.model.base,scope,optimizer_name='MuSGD',
            weight_decay=.00027,beta1=.948,beta2=.999)
        for group in optimizer.param_groups:
            group['param_names']=tuple('base.'+name for name in group['param_names'])
        state.optimizer=state.engine.optimizer=optimizer
        state.scheduler=StageWarmupCosineScheduler(optimizer,stage='recovery',epochs=10,
            steps_per_epoch=state.macros,warmup_epochs=1,warmup_start_factor=.1,final_lr_factor=.5)
    state.training_mode()
    _reseed(state,0)
    fixed={key:state.model.state_dict()[key].clone() for key in state.ema.fixed_state_names}
    active=[(group['group_name'],group['role'],name,p) for group in state.optimizer.param_groups
        for name,p in zip(group['param_names'],group['params']) if p.requires_grad]
    samples={};zero_initial={};trace=hashlib.sha256();losses=[]
    eligible={}; excluded={}
    def capture_gradient(optimizer,args,kwargs):
        for group,role,name,p in active:
            eligible[name] = p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.ne(0).any())
    hook=state.optimizer.register_step_pre_hook(capture_gradient)
    iterator=iter(_raw_macros(state,32))
    for step in range(16):
        macro=next(iterator)
        for batch in (*macro.detect_batches,*macro.pose_batches):
            trace.update(json.dumps([str(x) for x in batch['im_file']]).encode())
            for key in ('img','bboxes','cls','keypoints'):
                if key in batch:
                    value=batch[key].detach().cpu().contiguous()
                    trace.update(key.encode());trace.update(str(tuple(value.shape)).encode())
                    trace.update(value.numpy().tobytes())
        before={name:p.detach().clone() for _,_,name,p in active}
        eligible.clear()
        state.scheduler.prepare_step()
        report=state.engine.run(detect_batches=macro.detect_batches,pose_batches=macro.pose_batches,
            record_gradient_statistics=False)
        state.scheduler.advance()
        losses.append(float(report.joint_mean_loss))
        for group,role,name,p in active:
            if not eligible.get(name,False):
                excluded[group]=excluded.get(group,0)+1
                continue
            delta=(p.detach().float()-before[name].float()).square().mean().sqrt()
            scale=before[name].float().square().mean().sqrt()
            if not bool(torch.isfinite(delta)) or not bool(torch.isfinite(scale)):
                raise FloatingPointError('更新非有限')
            if float(scale)<=1e-12:
                zero_initial.setdefault(group,[]).append(float(delta));continue
            value=float(delta/scale)
            samples.setdefault(group,[]).append(value)
            samples.setdefault(role,[]).append(value)
    state.guard.assert_unchanged(state.model.base)
    assert all(torch.equal(state.model.state_dict()[key],value) and
        torch.equal(state.ema.ema.state_dict()[key],value) for key,value in fixed.items())
    assert state.ema.updates==26597+16
    stats={key:{'median':statistics.median(values),'count':len(values),
        'zero_fraction':sum(x==0 for x in values)/len(values)} for key,values in samples.items()}
    result={'label':label,'updates':16,'trace_sha256':trace.hexdigest(),
        'parent_state_sha256':state.metadata['parent_initial_state_sha256'],
        'optimizer':type(state.optimizer).__name__,'role_lrs':lrs or {'detect_head':2.5e-5,'pose_head':2.5e-5},
        'statistics':stats,'zero_initial_absolute_updates':zero_initial,'losses':losses,
        'excluded_no_nonzero_finite_gradient':excluded,
        'gradient_eligibility':'actual_unscaled_clipped_gradient_before_successful_optimizer_step',
        'fixed_states_exact':True,'updates_discarded':True,'validation_used':False}
    runtime.write_json(root/f'{label}.json',result)
    hook.remove()
    del iterator,state,active,before,fixed
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    root=parser.parse_args().output.resolve()
    if not root.is_relative_to(ROOT/'artifacts/direction1-20260908'):
        raise ValueError('只允許指定第一輪 artifacts 內新輸出')
    root.mkdir(exist_ok=False)
    config=runtime.load_config(root.parent)
    data=runtime.prepare_data(config,root.parent/'datasets')
    a=run_probe(config,data,root,'adamw')
    m=run_probe(config,data,root,'musgd_initial',{'detect_head':.0025,'pose_head':.0025})
    assert a['trace_sha256']==m['trace_sha256'] and a['parent_state_sha256']==m['parent_state_sha256']
    lrs={}
    for role in ('detect_head','pose_head'):
        av,mv=a['statistics'][role]['median'],m['statistics'][role]['median']
        if min(av,mv)<=0:raise ValueError('role 更新量為零，禁止放大 LR')
        lrs[role]=.0025*av/mv
    c=run_probe(config,data,root,'musgd_confirm',lrs)
    assert a['trace_sha256']==c['trace_sha256'] and a['parent_state_sha256']==c['parent_state_sha256']
    ratios={key:c['statistics'][key]['median']/value['median']
        for key,value in a['statistics'].items() if '.' in key and value['median']>0}
    passed=all(.5<=ratio<=2 for ratio in ratios.values())
    runtime.write_json(root/'summary.json',{'status':'passed' if passed else 'recipe_rejected',
        'calibrated_role_lrs':lrs,'group_update_ratios':ratios,'same_training_trace':True,
        'training_scope':'heads_only','handoff':'original_PSEL_ema_fresh_optimizers_not_failed_E5',
        'macro_budget':48,'validation_used':False,'weights_saved':False,
        'long_training_authorized_by_this_result':False,
        'interpretation':'僅更新量校準前置；未做正式20ep比較、resume契約或AP驗收。'})
    print('JOB_DONE: train-only calibration '+('passed' if passed else 'recipe_rejected'),flush=True)


if __name__=='__main__':main()
