"""Canonical train-only KD 倍率、全模型 live 梯度與更新驗證。"""
import json
import statistics
import torch
from experiment import configure, ProbeSession, runtime
from teachers import HERE, SelectedSource, SELECTED_SHA, sha256
from spatial_kd import SpatialRouter
from yolo_combine.factory import FusionModelFactory
from yolo_combine.stage_policy import build_joint_optimizer
from yolo_combine.hardware_contract import HardwareContractGuard
from yolo_combine.contracts import Task

def digest(model):
    import hashlib
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    config,stage=configure();runtime.formal.seed_everything(1)
    out=HERE/'artifacts/kd-calibration-v1';out.mkdir(exist_ok=False)
    session=ProbeSession(config,device='0',run_name='kd-calibration-v1')
    model=FusionModelFactory(SelectedSource(),detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint,checkpoint_kind='float').model.to('cuda:0')
    dl,pl,_=session._loaders(model)
    opt,_=build_joint_optimizer(model,stage,optimizer_name='AdamW',weight_decay=config.weight_decay,
                                beta1=config.beta1,beta2=config.beta2)
    guard=HardwareContractGuard.capture(model)
    router=SpatialRouter(model,epochs=5,imgsz=640,mu={'detect':1,'pose':1})
    hashes={t.value:digest(m) for t,m in router.teachers.items()}
    assert all(not m.training for m in router.teachers.values())
    rows={};mus={};last=None
    torch.cuda.reset_peak_memory_stats()
    for task,loader in ((Task.DETECT,dl),(Task.POSE,pl)):
        iterator=iter(loader.loader);records=[]
        for _ in range(4):
            batch=loader.preprocess(next(iterator))
            with torch.autocast('cuda',dtype=torch.float16):
                native,kd,features=router.parts(task,batch)
            gn=torch.autograd.grad(native.raw_total,features,retain_graph=True)
            gk=torch.autograd.grad(kd,features,retain_graph=True)
            n=float(sum(x.float().square().sum() for x in gn).sqrt())
            k=float(sum(x.float().square().sum() for x in gk).sqrt())
            assert n>0 and k>0 and torch.isfinite(torch.tensor([n,k])).all()
            records.append({'native_feature_gradient':n,'kd_feature_gradient':k,'ratio':n/k,
                            'images':batch['img'].shape[0],'kd_per_image':float(kd.detach())/len(batch['img'])})
            del native,kd,features,gn,gk
        mu=.1*statistics.median(r['ratio'] for r in records)
        assert 0<mu<1e6
        ratios=[mu/r['ratio'] for r in records]
        assert all(.025<=r<=.4 for r in ratios),ratios
        rows[task.value]=records;mus[task.value]=mu
        # 最後一個真實 train batch 做 KD-only 上游梯度及實際一步更新，更新丟棄。
        with torch.autocast('cuda',dtype=torch.float16):native,kd,features=router.parts(task,batch)
        opt.zero_grad(set_to_none=True)
        (mu*kd/len(batch['img'])).backward()
        active={n:p for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert active and all(torch.isfinite(p.grad).all() for p in active.values())
        assert any(n.startswith('graph.model.19.') and p.grad.abs().sum()>0 for n,p in active.items())
        before={n:p.detach().clone() for n,p in active.items()}
        torch.nn.utils.clip_grad_norm_(active.values(),10);opt.step();opt.zero_grad(set_to_none=True)
        assert any(not torch.equal(p,before[n]) for n,p in active.items() if n.startswith('graph.model.19.'))
        # 還原這次 probe 更新，兩任務倍率都以相同學生參數計算。
        with torch.no_grad():
            for n,p in active.items():p.copy_(before[n])
        opt.state.clear()
        del native,kd,features,before,active
    guard.assert_unchanged(model)
    assert hashes=={t.value:digest(m) for t,m in router.teachers.items()}
    state=router.state_dict();router.load_state_dict(state);assert router.state_dict()==state
    result={'status':'passed','mu':mus,'student_sha256':SELECTED_SHA,'records':rows,
        'teacher_state_unchanged':hashes,'kd_only_shared_update':True,'updates_discarded':True,
        'calibration_target':'0.1 times native gradient at shared Neck features; not full parameter gradient ratio',
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'validation_used':False,
        'remaining':'actual mixed macro AMP/snapshot/export checks before training'}
    with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2)
    print('JOB_DONE: KD train-only calibration '+json.dumps(mus),flush=True)

if __name__=='__main__':main()
