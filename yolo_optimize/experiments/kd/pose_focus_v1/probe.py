"""真實 BBAT train 前置：head KD 梯度校準與 Detect 不變。"""
import argparse
from dataclasses import asdict
import json
import statistics
import torch
import experiment as exp
from head_kd import HeadRouter, GROUPS
from teachers import SelectedSource
from state_checks import digest
from yolo_combine.factory import FusionModelFactory
from yolo_combine.hardware_contract import HardwareContractGuard

def flatten(x):
    if isinstance(x,torch.Tensor):return [x.detach().clone()]
    if isinstance(x,dict):return sum((flatten(v) for v in x.values()),[])
    if isinstance(x,(tuple,list)):return sum((flatten(v) for v in x),[])
    return []

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('native','kd','calibrate'),required=True)
    arm=parser.parse_args().arm;exp.ARM='native' if arm=='calibrate' else arm
    config,stage=exp.configure();exp.base.formal.seed_everything(1)
    session=exp.Session(config,device='0',run_name=arm+'-probe-v1')
    session.run_dir.mkdir(parents=True,exist_ok=False)
    model=FusionModelFactory(SelectedSource(),detect_data_yaml=config.detect_data,pose_data_yaml=config.pose_data).build(
        pose_head_checkpoint=config.pose_checkpoint,checkpoint_kind='float').model.to('cuda:0')
    _,loader,_=session._loaders(model)
    sample=torch.rand(1,3,160,160,device='cuda:0')
    with torch.inference_mode():before_detect=flatten(model.eval()(sample,task='detect'))
    opt,grouping=exp.base.policy.build_joint_optimizer(model,stage,optimizer_name='AdamW',
        weight_decay=config.weight_decay,beta1=config.beta1,beta2=config.beta2)
    exp.base.formal._apply_shared_bn_affine(model,trainable=False)
    assert all('.pose_head.' in n for n,p in model.named_parameters() if p.requires_grad)
    guard=HardwareContractGuard.capture(model)
    router=(HeadRouter(model,epochs=5,imgsz=640,mu=1) if arm=='calibrate' else
            exp.Router(model,epochs=5,imgsz=640) if arm=='kd' else exp.base.NativeTaskLossRouter(model,epochs=5,imgsz=640))
    teacher_hash=digest(router.teacher) if arm!='native' else None
    torch.cuda.reset_peak_memory_stats();iterator=iter(loader.loader)
    records=[]
    if arm=='calibrate':
        for _ in range(4):
            batch=loader.preprocess(next(iterator))
            with torch.autocast('cuda',dtype=torch.float16):native,kd,features=router.parts(batch)
            gn=torch.autograd.grad(native.raw_total,features,retain_graph=True)
            gk=torch.autograd.grad(kd,features,retain_graph=True)
            n=float(sum(g.float().square().sum() for g in gn).sqrt())
            k=float(sum(g.float().square().sum() for g in gk).sqrt())
            assert n>0 and k>0 and torch.isfinite(torch.tensor([n,k])).all()
            records.append({'native_norm':n,'kd_norm':k,'ratio':n/k})
            opt.zero_grad(set_to_none=True);(kd/len(batch['img'])).backward()
            for group in GROUPS:
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model.pose_head,group).parameters())
            assert all(p.grad is None for n,p in model.named_parameters() if '.pose_head.' not in n)
            opt.zero_grad(set_to_none=True)
            del native,kd,features,gn,gk
        mu=.05*statistics.median(r['ratio'] for r in records)
        assert 0<mu<1e6 and all(.0125<mu/r['ratio']<.2 for r in records)
        with (exp.HERE/'artifacts/calibration-v1.json').open('x') as f:
            json.dump({'status':'passed','mu':mu,'records':records,'target':'head feature gradient 5% native',
                'all_four_head_groups_live_kd_gradient':True,'no_parameter_updates':True,'validation_used':False},f,indent=2)
    else:
        ema=exp.PoseOnlyEMA(model)
        engine=exp.base.MacroStepEngine(model=model,
            losses=exp.base.impl._AutocastTaskLossRouter(router,device=torch.device('cuda:0'),enabled=True),
            optimizer=opt,reference_batch_size=config.reference_batch_size,task_weights={'detect':1,'pose':1},
            scaler=torch.amp.GradScaler('cuda',init_scale=1024),ema=ema,max_grad_norm=10,max_amp_retries=16,
            preprocess=lambda t,b:loader.preprocess(b))
        active={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        for _ in range(2):records.append(asdict(engine.run(detect_batches=(),pose_batches=(next(iterator),))))
        params=dict(model.named_parameters());assert any(not torch.equal(p,params[n]) for n,p in active.items())
        for m in (model,ema.ema):
            guard.assert_unchanged(m);state=m.state_dict()
            assert all(torch.equal(v,state[n].detach().cpu()) for n,v in session.fixed.items())
        state_path=session.run_dir/'criteria-state.pt';torch.save(router.state_dict(),state_path)
        router.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True))
    guard.assert_unchanged(model)
    if teacher_hash:
        assert teacher_hash==digest(router.teacher) and all(p.grad is None for p in router.teacher.parameters())
    with torch.inference_mode():after=flatten(model.eval()(sample,task='detect'))
    assert len(before_detect)==len(after) and all(torch.equal(a,b) for a,b in zip(before_detect,after))
    with (session.run_dir/'summary.json').open('x') as f:
        json.dump({'status':'passed','arm':arm,'detect_output_exact':True,'fixed_non_pose_state':True,
            'teacher_unchanged':teacher_hash,'records':records,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()},f,indent=2)
    print('JOB_DONE: Pose head '+arm+' preflight passed',flush=True)

if __name__=='__main__':main()
