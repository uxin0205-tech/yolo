"""最後J3-only低LR聯合微調，完成後集中驗收，不自動延長。"""
from dataclasses import asdict,replace
import json
import sys
import torch
import j3_stage
import balanced_joint
import runtime
import smoke_joint
from safe_source import HERE


def install():
    config,stage=j3_stage.install()
    result=json.loads((HERE/'artifacts/j3-task-weight-calibration-v1/summary.json').read_text())
    assert result['status']=='passed' and result['selected_pose_weight']==.215
    config=replace(config,pose_weight=.215)
    config._validate()
    return config,stage


class Session(balanced_joint.Session):
    def __init__(self,config,*,device='0',run_name='balanced-j3-v1'):
        if run_name=='j1-smoke-v1': run_name='balanced-j3-smoke-v1'
        runtime.formal.FormalJointTrainingSession.__init__(self,config,device=device,run_name=run_name)
        self.pose_start=torch.load(j3_stage.START,map_location='cpu',weights_only=True)['metadata']['metrics']

    def _resolved_config(self):
        result=runtime.formal.FormalJointTrainingSession._resolved_config(self)
        stage=runtime.impl.JOINT_STAGES['j3']
        result['stage_policies']['j3'].update(epochs=stage.epochs,patience=stage.patience,
            warmup_epochs=1,learning_rates=dict(stage.learning_rates))
        result['j3_local_policy']={'source':str(j3_stage.START),'source_sha256':j3_stage.START_SHA,
            'pose_weight':.215,'fresh_optimizer_and_loss_horizon':20,'explicit_J3_only':True,
            'enable_j3_append_false_to_avoid_duplicate_stage':True,
            'final_coco_drop':.005,'final_pose_drop':.02,'no_automatic_extension':True,
            'training_safety':'same as balanced J1; Pose drop relative to J2 best'}
        return result


def main():
    if '--smoke' in sys.argv:
        smoke_joint.install=install
        smoke_joint.Session=Session
        smoke_joint.AdaptedBridgeSource=j3_stage.J3Source
        smoke_joint.main()
        return
    config,_=install()
    assert json.loads((HERE/'artifacts/fusion/balanced-j3-smoke-v1/summary.json').read_text())['status']=='passed'
    session=Session(config)
    json.dumps(session._resolved_config())
    try: report=session.run()
    except runtime.SafetyStop:
        print('JOB_DONE: J3 saved safety stop; consolidated verification next',flush=True)
        return
    with (session.run_dir/'summary.json').open('x') as f: json.dump(asdict(report),f,indent=2,default=str)
    print('JOB_DONE: J3 complete; consolidated verification next',flush=True)


if __name__=='__main__': main()
