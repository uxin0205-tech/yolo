"""J1 best 後的獨立 J2：只新增 backbone layer9+ 的小 LR 適應。"""
from dataclasses import fields, replace
from types import MappingProxyType
import torch
from safe_source import HERE, SOURCE, sha256
import balanced_joint
import runtime
from merge_joint import JointSource
from yolo_combine.joint_config import JointExperimentConfig
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from pwl_contract import verify_pwl

START = HERE/'artifacts/fusion/balanced-j1-v2/inference/best_pose.pt'
START_SHA = '7ef02ce42ab9dfec984f60a79cc3b52002e6630dd35665ecf7c0001269902c50'


class J2Source(JointSource):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert sha256(START)==START_SHA
        payload=torch.load(START,map_location='cpu',weights_only=True)
        self.record['detect_expected_metrics']={k:payload['metadata']['metrics'][k] for k in self.record['detect_expected_metrics']}

    def provenance(self,kind='float'):
        return {**super().provenance(kind),'joint_j1_best':str(START),'joint_j1_best_sha256':START_SHA,
            'source_kind':'trained_joint_J1_best_pose_E6','new_stage_fresh_optimizer':True}

    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair=super().build_task_models('float',pose_head_checkpoint=pose_head_checkpoint)
        payload=torch.load(START,map_location='cpu',weights_only=True)
        assert payload['metadata']['stage']=='j1' and payload['metadata']['epoch']==5
        full=payload['state_dict']
        for task,model in [('detect',pair.detect),('pose',pair.pose)]:
            state={}
            for name in model.state_dict():
                key=('graph.model.23.'+task+'_head.'+name[len('model.23.'):]
                     if name.startswith('model.23.') else 'graph.'+name)
                state[name]=full[key]
            model.load_state_dict(state,strict=True)
            convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/f'configs/attention/{kind}-pwl-final.yaml'))
            verify_pwl(model)
        return pair


class Config(JointExperimentConfig):
    def _validate(self):
        JointExperimentConfig._validate(replace(self,stages=('j0','j1','j2')))
        assert self.stages==('j2',) and not self.enable_j3


def install():
    original,_=balanced_joint.install()
    values={f.name:getattr(original,f.name) for f in fields(original) if f.init}
    values['stages']=('j2',)
    config=Config(**values)
    config._validate()
    stage=replace(runtime.policy.JOINT_STAGES['j2'],epochs=40,warmup_epochs=1,
        learning_rates=MappingProxyType({'backbone':1.5e-6,'neck':7.5e-6,'masf':1e-6,
            'attention':0.,'detect_head':2e-5,'pose_head':2e-5}))
    runtime.impl.JOINT_STAGES=MappingProxyType({**dict(runtime.impl.JOINT_STAGES),'j2':stage})
    runtime.impl.SourceBundle=J2Source
    assert config.preflight().ready
    return config,stage


if __name__=='__main__':
    import calibrate_task_weight as probe
    probe.NAME='j2-task-weight-calibration-v1'
    probe.install=install
    probe.JointSource=J2Source
    probe.main()
