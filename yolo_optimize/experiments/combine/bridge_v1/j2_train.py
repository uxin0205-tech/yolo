"""J2-only 正式接續，固定校準 weight0.07、backbone9+小 LR。"""
from dataclasses import asdict, replace
import json
import sys
import torch
import j2_stage
import balanced_joint
import runtime
import smoke_joint
from safe_source import HERE


def install():
    config,stage=j2_stage.install()
    result=json.loads((HERE/'artifacts/j2-task-weight-calibration-v1/summary.json').read_text())
    assert result['status']=='passed' and result['selected_pose_weight']==.07
    config=replace(config,pose_weight=.07)
    config._validate()
    return config,stage


class Session(balanced_joint.Session):
    def __init__(self,config,*,device='0',run_name='balanced-j2-v1'):
        if run_name=='j1-smoke-v1':
            run_name='balanced-j2-smoke-v1'
        runtime.formal.FormalJointTrainingSession.__init__(self,config,device=device,run_name=run_name)
        self.pose_start=torch.load(j2_stage.START,map_location='cpu',weights_only=True)['metadata']['metrics']

    def _resolved_config(self):
        result=runtime.formal.FormalJointTrainingSession._resolved_config(self)
        stage=runtime.impl.JOINT_STAGES['j2']
        result['stage_policies']['j2'].update(epochs=stage.epochs,patience=stage.patience,
            warmup_epochs=1,learning_rates=dict(stage.learning_rates))
        result['j2_local_policy']={'source':str(j2_stage.START),'source_sha256':j2_stage.START_SHA,
            'pose_weight':.07,'fresh_optimizer_and_loss_horizon':40,
            'final_coco_drop':.005,'final_pose_drop':.02,
            'training_safety':'same as balanced J1; Pose drop relative to J1 best',
            'masf_lr_held_at_j1':1e-6,'backbone_start_layer':9}
        return result


def main():
    if '--smoke' in sys.argv:
        smoke_joint.install=install
        smoke_joint.Session=Session
        smoke_joint.AdaptedBridgeSource=j2_stage.J2Source
        smoke_joint.main()
        return
    config,_=install()
    assert json.loads((HERE/'artifacts/fusion/balanced-j2-smoke-v1/summary.json').read_text())['status']=='passed'
    session=Session(config)
    # 在正式run前觸發配置序列化，避免重演J1的super整合錯誤。
    json.dumps(session._resolved_config())
    try:
        report=session.run()
    except runtime.SafetyStop:
        print('JOB_DONE: J2 saved safety stop; review required',flush=True)
        return
    with (session.run_dir/'summary.json').open('x') as f:
        json.dump(asdict(report),f,indent=2,default=str)
    print('JOB_DONE: J2 complete; final verification required',flush=True)


if __name__=='__main__':
    main()
