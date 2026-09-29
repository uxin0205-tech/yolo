"""qSiLU E2 固定共享層的 Pose-only 五輪；只寫新研究目錄。"""
from pathlib import Path
from dataclasses import replace
from types import MappingProxyType
import json
import head_kd
from teachers import SelectedSource, SELECTED, SELECTED_SHA
import pose_first as base
from j0_runtime import PoseOnlyEMA, J0Gate
from activation_lab.training.full35 import _fp32_bbox_iou
import ultralytics.utils.loss as loss_module
import torch
from state_checks import digest

HERE=Path(__file__).resolve().parent
ARM='native'
ROUTERS=[]

class Router(head_kd.HeadRouter):
    def __init__(self,*args,**kwargs):
        mu=json.loads((HERE/'artifacts/calibration-v1.json').read_text())['mu']
        super().__init__(*args,mu=mu,**kwargs)
        self.initial_teacher_hash=digest(self.teacher)
        ROUTERS.append(self)

def configure():
    config,stage=base.install()
    baseline=HERE.parents[0]/'dual_task_v1/artifacts/student-baseline-v1.json'
    config=replace(config,seed=1,run_root=HERE/'artifacts/runs',baseline_metrics_path=baseline,
        maximum_map_drop=.001,optimizer='AdamW',pose_weight=1.0)
    stage=replace(stage,epochs=5,patience=0,warmup_epochs=1,
        learning_rates=MappingProxyType({**dict(stage.learning_rates),'pose_head':1e-5}))
    base.impl.JOINT_STAGES=MappingProxyType({**dict(base.impl.JOINT_STAGES),'j0':stage})
    base.impl.SourceBundle=SelectedSource
    base.impl.NativeTaskLossRouter=Router if ARM=='kd' else base.NativeTaskLossRouter
    loss_module.bbox_iou=_fp32_bbox_iou(loss_module.bbox_iou)
    config._validate();assert config.preflight().ready
    assert config.pose_batch_size==16 and not config.shared_bn_affine_trainable
    return config,stage

class Session(base.formal.FormalJointTrainingSession):
    def _loaders(self,model):
        self.fixed={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if '.pose_head.' not in k}
        return super()._loaders(model)

    def _resolved_config(self):
        result=super()._resolved_config()
        result['stage_policies']['j0'].update(epochs=5,patience=0,warmup_epochs=1,
            learning_rates=dict(base.impl.JOINT_STAGES['j0'].learning_rates))
        result['pose_focus']={'arm':ARM,'student_sha256':SELECTED_SHA,'epochs':5,'warmup':1,
            'physical_batch':16,'optimizer':'AdamW','pose_lr':1e-5,'fixed_all_non_pose_state':True,
            'kd_mu':json.loads((HERE/'artifacts/calibration-v1.json').read_text())['mu'] if ARM=='kd' else 0,
            'source':str(SELECTED),'teacher_only_training':True,'shared_feature_trial_requires_separate_branch':True}
        return result

    def _save_selected(self,labels,**kwargs):
        for model in (kwargs['model'],kwargs['ema'].ema):
            state=model.state_dict()
            assert all(torch.equal(v,state[n].detach().cpu()) for n,v in self.fixed.items()), 'non-Pose state changed'
        saved=super()._save_selected(labels,**kwargs)
        failed={k:kwargs['metrics'][k]-v for k,v in self.config.load_baseline().items()
                if k.startswith('bbat/') and kwargs['metrics'][k]-v<-.02}
        if failed:
            with (self.run_dir/'safety-stop.json').open('x') as f:json.dump(failed,f,indent=2)
            raise RuntimeError('BBAT regression safety stop: '+str(failed))
        return saved
