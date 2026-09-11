"""J3 後的最小 BBAT 恢復試驗：只更新 Pose head，固定共享層與 Detect。"""
from dataclasses import asdict, replace
from types import MappingProxyType
import json
import sys
import torch
import pose_first as base
from safe_source import HERE, BridgeSource, SOURCE, sha256
from compare_old_combine import NEW, HASHES
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from pwl_contract import verify_pwl

NAME = 'j3-pose-head-recovery-v1'
SMOKE = 'j3-pose-head-recovery-smoke-v1'


class RecoverySource(BridgeSource):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert sha256(NEW) == HASHES[NEW]
        payload = torch.load(NEW,map_location='cpu',weights_only=True)
        self.record['detect_expected_metrics'] = {k:payload['metadata']['metrics'][k] for k in self.record['detect_expected_metrics']}

    def provenance(self,kind='float'):
        return {**super().provenance(kind),'source_kind':'trained_J3_E12_pose_head_recovery',
            'joint_start':str(NEW),'joint_start_sha256':HASHES[NEW]}

    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair = super().build_task_models('float',pose_head_checkpoint=pose_head_checkpoint)
        assert sha256(NEW) == HASHES[NEW]
        full = torch.load(NEW,map_location='cpu',weights_only=True)['state_dict']
        for task,model in [('detect',pair.detect),('pose',pair.pose)]:
            state = {n:full['graph.model.23.'+task+'_head.'+n[len('model.23.'):]]
                if n.startswith('model.23.') else full['graph.'+n] for n in model.state_dict()}
            model.load_state_dict(state,strict=True)
            convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/f'configs/attention/{kind}-pwl-final.yaml'))
            verify_pwl(model)
        return pair


def install():
    config, stage = base.install()
    path = HERE/'artifacts/j3-pose-head-recovery-baseline-v1.json'
    baseline = config.load_baseline()
    start = torch.load(NEW,map_location='cpu',weights_only=True)['metadata']['metrics']
    baseline = {k:start[k] if k.startswith('coco/') else v for k,v in baseline.items()}
    if not path.exists():
        with path.open('x') as f:
            json.dump({'metrics':baseline,'note':'COCO 僅作精確不變檢查；BBAT 仍比原獨立 Pose，容許下降 0.02。'},f,indent=2)
    else:
        assert json.loads(path.read_text())['metrics'] == baseline
    config = replace(config,baseline_metrics_path=path,maximum_map_drop=.02)
    stage = replace(stage,epochs=10,patience=0,warmup_epochs=1,
        learning_rates=MappingProxyType({**dict(stage.learning_rates),'pose_head':2e-5}))
    base.impl.JOINT_STAGES = MappingProxyType({**dict(base.impl.JOINT_STAGES),'j0':stage})
    base.impl.SourceBundle = RecoverySource
    return config,stage


class Session(base.formal.FormalJointTrainingSession):
    def __init__(self,*args,**kwargs):
        if kwargs.get('run_name') == 'j0-extend-smoke-v1': kwargs['run_name'] = SMOKE
        super().__init__(*args,**kwargs)

    def _resolved_config(self):
        result = super()._resolved_config()
        result['bbat_recovery'] = {'source':str(NEW),'epochs':10,'patience':4,'warmup_epochs':1,
            'optimizer':'AdamW','pose_head_lr':2e-5,'physical_pose_batch':16,
            'fixed_shared_and_detect':True,'new_optimizer_and_native_loss_horizon':10,
            'acceptance':'improve vs starting J3; retain original COCO 0.005 / BBAT 0.02 final gates'}
        return result

    def _save_selected(self,labels,**kwargs):
        saved = super()._save_selected(labels,**kwargs)
        metrics = kwargs['metrics']
        if not hasattr(self,'_start_metrics'):
            self._start_metrics = torch.load(NEW,map_location='cpu',weights_only=True)['metadata']['metrics']
        keys = [k for k in self.config.load_baseline() if k.startswith('bbat/')]
        score = sum(metrics[k] for k in keys)/len(keys)
        initial = sum(self._start_metrics[k] for k in keys)/len(keys)
        previous = getattr(self,'_best_recovery',initial)
        self._stale = 0 if score > previous+.0001 else getattr(self,'_stale',0)+1
        self._best_recovery = max(previous,score)
        failures = {k:metrics[k]-self._start_metrics[k] for k in keys if metrics[k] < self._start_metrics[k]-.02}
        if failures or self._stale >= 4:
            with (self.run_dir/'recovery-stop.json').open('x') as f:
                json.dump({'status':'SAFETY_STOP' if failures else 'PLATEAU','failed_deltas':failures,
                    'best_score':self._best_recovery,'starting_score':initial,
                    'saved':{k:str(v) for k,v in saved.items()}},f,indent=2)
            raise base.PosePlateau()
        return saved


def main():
    config,stage = install()
    base.formal.seed_everything(config.seed)
    base.Session = Session
    base.AdaptedBridgeSource = RecoverySource
    if '--smoke' in sys.argv:
        base.smoke(config,stage)
        return
    assert json.loads((HERE/f'artifacts/fusion/{SMOKE}/summary.json').read_text())['status'] == 'passed'
    session = Session(config,device='0',run_name=NAME)
    json.dumps(session._resolved_config(),default=str)
    try:
        report = session.run()
    except base.PosePlateau:
        print('JOB_DONE: Pose head 恢復候選已保存，須依停止結果驗收。',flush=True)
        return
    with (session.run_dir/'summary.json').open('x') as f:
        json.dump(asdict(report),f,indent=2,default=str)
    print('JOB_DONE: Pose head 恢復訓練完成，未自動替換正式權重。',flush=True)


if __name__ == '__main__':
    main()
