"""報告最後的 J3 低 LR：明確的 J3-only 分支，不重跑 J0–J2。"""
from dataclasses import fields, replace
from types import MappingProxyType
import torch
import j2_stage
import runtime
from safe_source import HERE, SOURCE, sha256
from yolo_combine.joint_config import JointExperimentConfig
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from pwl_contract import verify_pwl

START=HERE/'artifacts/fusion/balanced-j2-v1/inference/best_pose.pt'
START_SHA='30df89e8791b06e89ac51e099317d810299973e7b6b5adaeb0a2ef82b44ccfcc'


class J3Source(j2_stage.J2Source):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert sha256(START)==START_SHA
        payload=torch.load(START,map_location='cpu',weights_only=True)
        self.record['detect_expected_metrics']={k:payload['metadata']['metrics'][k] for k in self.record['detect_expected_metrics']}

    def provenance(self,kind='float'):
        return {**super().provenance(kind),'joint_j2_best':str(START),'joint_j2_best_sha256':START_SHA,
            'source_kind':'trained_joint_J2_best_pose_E22','explicit_J3_only':True}

    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair=super().build_task_models('float',pose_head_checkpoint=pose_head_checkpoint)
        payload=torch.load(START,map_location='cpu',weights_only=True)
        assert payload['metadata']['stage']=='j2' and payload['metadata']['epoch']==21
        full=payload['state_dict']
        for task,model in [('detect',pair.detect),('pose',pair.pose)]:
            state={n:full['graph.model.23.'+task+'_head.'+n[len('model.23.'):]]
                if n.startswith('model.23.') else full['graph.'+n] for n in model.state_dict()}
            model.load_state_dict(state,strict=True)
            convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/f'configs/attention/{kind}-pwl-final.yaml'))
            verify_pwl(model)
        return pair


class Config(JointExperimentConfig):
    def _validate(self):
        JointExperimentConfig._validate(replace(self,stages=('j0','j1','j2')))
        # J3已明確列入stages；enable_j3只控制額外append，保持False避免重複。
        assert self.stages==('j3',) and not self.enable_j3


def install():
    original,_=j2_stage.install()
    values={f.name:getattr(original,f.name) for f in fields(original) if f.init}
    values['stages']=('j3',)
    config=Config(**values)
    config._validate()
    stage=replace(runtime.policy.JOINT_STAGES['j3'],warmup_epochs=1,
        learning_rates=MappingProxyType({'backbone':3.8e-7,'neck':1.9e-6,'masf':1e-6,
            'attention':5e-8,'detect_head':5e-6,'pose_head':5e-6}))
    runtime.impl.JOINT_STAGES=MappingProxyType({**dict(runtime.impl.JOINT_STAGES),'j3':stage})
    runtime.impl.SourceBundle=J3Source
    assert config.preflight().ready
    return config,stage


if __name__=='__main__':
    import calibrate_task_weight as probe
    probe.NAME='j3-task-weight-calibration-v1'
    probe.install=install
    probe.JointSource=J3Source
    probe.main()
