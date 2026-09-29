"""與既有 activation recipe 同預算的 SiLU／qSiLU 短恢復。"""
import argparse
from dataclasses import asdict, fields, replace
from types import MappingProxyType
import json
import torch
from evaluate import HERE, ROOT, Source, START, SHA
import j3_stage
import runtime
import smoke_joint
from yolo_combine.metrics import AccuracyGate
from activation_lab.training.full35 import _fp32_bbox_iou
import ultralytics.utils.loss as loss_module

ARM = 'silu'


class ActivationConfig(j3_stage.Config):
    @property
    def shared_bn_affine_trainable(self):
        return True


class ActivationSource(Source):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,activation=ARM,**kwargs)


def install():
    config,stage = j3_stage.install()
    config = ActivationConfig(**{f.name:getattr(config,f.name) for f in fields(config) if f.init})
    baseline_path = HERE/'artifacts/activation-baseline-v1.json'
    baseline = json.loads((HERE/'artifacts/zero-shot-v1/summary.json').read_text())['results']['silu']['bittrue']
    if not baseline_path.exists():
        with baseline_path.open('x') as f:
            json.dump({'metrics':{k:v for k,v in baseline.items() if k.endswith('map50_95')},
                'source':str(START),'sha256':SHA,'not_original_fusion_acceptance':True},f,indent=2)
    config = replace(config,seed=1,pose_weight=.25,maximum_map_drop=.015,detect_microbatch_size=16,
        baseline_metrics_path=baseline_path,run_root=HERE/'artifacts/runs')
    stage = replace(stage,epochs=10,patience=0,warmup_epochs=1,learning_rates=MappingProxyType({
        'backbone':3.8e-7,'neck':1.9e-6,'masf':3.8e-6,'attention':5e-8,
        'detect_head':5e-6,'pose_head':5e-6}))
    runtime.impl.JOINT_STAGES = MappingProxyType({**dict(runtime.impl.JOINT_STAGES),'j3':stage})
    runtime.impl.SourceBundle = ActivationSource
    runtime.impl.AccuracyGate = AccuracyGate
    loss_module.bbox_iou = _fp32_bbox_iou(loss_module.bbox_iou)
    config._validate()
    assert config.preflight().ready
    return config,stage


class Session(runtime.formal.FormalJointTrainingSession):
    def __init__(self,config,*,device='0',run_name=None):
        name = f'{ARM}-smoke-b16-v2' if run_name == 'j1-smoke-v1' else f'{ARM}-short-e10-seed1-v1'
        super().__init__(config,device=device,run_name=name)

    def _resolved_config(self):
        result = super()._resolved_config()
        stage = runtime.impl.JOINT_STAGES['j3']
        result['stage_policies']['j3'].update(epochs=10,patience=0,warmup_epochs=1,
            learning_rates=dict(stage.learning_rates))
        result['activation_recovery'] = {'activation':ARM,'start':str(START),'sha256':SHA,
            'paired_seed':1,'epochs':10,'warmup':1,'pose_weight':.25,
            'shared_bn_affine_trainable':True,'shared_bn_running_fixed':True,
            'fp32_bbox_iou':True,'activation_gate':.015,'best_joint_is_activation_relative_only':True,
            'original_fusion_gate_still_reported_separately':True}
        return result

    def _save_selected(self,labels,**kwargs):
        saved = super()._save_selected(labels,**kwargs)
        delta = {k:kwargs['metrics'][k]-v for k,v in self.config.load_baseline().items()}
        failures = {k:v for k,v in delta.items() if v < -.08}
        if failures:
            with (self.run_dir/'safety-stop.json').open('x') as f:
                json.dump({'status':'SAFETY_STOP','delta':failures,
                    'note':'catastrophic stop, not activation final gate'},f,indent=2)
            raise runtime.SafetyStop(str(failures))
        return saved


def main():
    global ARM
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation',choices=('silu','qsilu_pq'),required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args(); ARM = args.activation
    if args.smoke:
        smoke_joint.install = install
        smoke_joint.Session = Session
        smoke_joint.AdaptedBridgeSource = ActivationSource
        smoke_joint.main()
        return
    config,_ = install()
    assert json.loads((config.run_root/f'{ARM}-smoke-b16-v2/summary.json').read_text())['status'] == 'passed'
    session = Session(config)
    json.dumps(session._resolved_config())
    try: report = session.run()
    except runtime.SafetyStop:
        print('JOB_DONE: activation safety stop saved; review required',flush=True)
        return
    with (session.run_dir/'summary.json').open('x') as f: json.dump(asdict(report),f,indent=2,default=str)
    print('JOB_DONE: activation paired short recovery complete '+ARM,flush=True)


if __name__ == '__main__': main()
