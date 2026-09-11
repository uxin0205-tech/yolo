"""K0／雙教師空間 KD：相同 qSiLU 學生、AdamW、五 epoch 預算。"""
import argparse
from dataclasses import asdict
import json
from experiment import configure, runtime
from teachers import HERE, SelectedSource, SELECTED_SHA, DETECT_SHA, POSE_SHA
from spatial_kd import SpatialRouter
from calibrate_kd import digest
from yolo_combine.joint_loss import NativeTaskLossRouter
import smoke_joint
import torch

ARM='k0'
ACTIVE=[]

class TrackedRouter(SpatialRouter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.initial_teacher_hashes={t.value:digest(m) for t,m in self.teachers.items()}
        ACTIVE.append(self)

def install():
    config,stage=configure()
    runtime.impl.NativeTaskLossRouter=TrackedRouter if ARM=='spatial' else NativeTaskLossRouter
    return config,stage

class Session(runtime.formal.FormalJointTrainingSession):
    def __init__(self,config,*,device='0',run_name=None):
        name=f'{ARM}-smoke-v1' if run_name=='j1-smoke-v1' else f'{ARM}-e5-seed1-v1'
        super().__init__(config,device=device,run_name=name)

    def _resolved_config(self):
        result=super()._resolved_config()
        result['stage_policies']['j3'].update(epochs=5,patience=0,warmup_epochs=1,
            learning_rates=dict(runtime.impl.JOINT_STAGES['j3'].learning_rates))
        result['dual_teacher_kd']={'arm':ARM,'student_sha256':SELECTED_SHA,'detect_teacher_sha256':DETECT_SHA,
            'pose_teacher_sha256':POSE_SHA,'method':'channel-mean square spatial attention transfer; not QK ranking',
            'sites':[16,19,22],'mu':json.loads((HERE/'artifacts/kd-calibration-v1/summary.json').read_text())['mu'] if ARM=='spatial' else {},
            'optimizer':'AdamW','musgd_recipe_rejected':True,'warmup_epochs':1,'mu_fixed':True,
            'teacher_exported':False,'original_fusion_gate_reported_separately':True}
        return result

    def _save_selected(self,labels,**kwargs):
        outputs=super()._save_selected(labels,**kwargs)
        failures={k:kwargs['metrics'][k]-v for k,v in self.config.load_baseline().items()
                  if kwargs['metrics'][k]-v < -.05}
        if failures:
            with (self.run_dir/'safety-stop.json').open('x') as f:json.dump(failures,f,indent=2)
            raise runtime.SafetyStop(str(failures))
        return outputs

def verify_router_after_smoke():
    assert len(ACTIVE)==1
    router=ACTIVE[0]
    assert router.initial_teacher_hashes=={t.value:digest(m) for t,m in router.teachers.items()}
    assert all(p.grad is None for m in router.teachers.values() for p in m.parameters())
    model=router.model
    assert not any('teacher' in n or 'projector' in n for n in model.state_dict())
    path=HERE/'artifacts/runs/spatial-smoke-v1/kd-state-roundtrip.pt'
    # 僅 state_dict，無 pickle model；測試新 criterion 額外資料的可保存性。
    torch.save({'criteria':router.state_dict(),'model':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    router.load_state_dict(payload['criteria'])
    assert router.state_dict()==payload['criteria']
    assert all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in payload['model'].items())
    assert not any(m._forward_hooks for m in model.graph.model[:23])
    with path.with_suffix('.json').open('x') as f:
        json.dump({'status':'passed','teachers_unchanged':True,'teacher_gradients_absent':True,
            'temporary_hooks_removed':True,'safe_state_roundtrip':True,'student_state_contains_no_teacher':True},f,indent=2)

def main():
    global ARM
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('k0','spatial'),required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();ARM=args.arm
    if args.smoke:
        smoke_joint.install=install;smoke_joint.Session=Session;smoke_joint.AdaptedBridgeSource=SelectedSource
        smoke_joint.NativeTaskLossRouter=TrackedRouter if ARM=='spatial' else NativeTaskLossRouter
        smoke_joint.main()
        if ARM=='spatial':verify_router_after_smoke()
        return
    config,_=install()
    for arm in ('k0','spatial'):
        assert json.loads((config.run_root/f'{arm}-smoke-v1/summary.json').read_text())['status']=='passed'
    assert json.loads((config.run_root/'spatial-smoke-v1/kd-state-roundtrip.json').read_text())['status']=='passed'
    session=Session(config);json.dumps(session._resolved_config())
    report=session.run()
    if ARM=='spatial':
        router=ACTIVE[0]
        assert router.initial_teacher_hashes=={t.value:digest(m) for t,m in router.teachers.items()}
    with (session.run_dir/'summary.json').open('x') as f:json.dump(asdict(report),f,indent=2,default=str)
    print('JOB_DONE: '+ARM+' paired five epochs completed',flush=True)

if __name__=='__main__':main()
