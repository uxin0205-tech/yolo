"""新 qSiLU 學生的獨立 KD 設定；不從教師或舊學生接續 optimizer。"""
from dataclasses import replace
from types import MappingProxyType
import json
from teachers import HERE, SelectedSource, SELECTED, SELECTED_SHA
import train as activation_train
import runtime

def configure():
    config, stage = activation_train.install()
    baseline_path = HERE/'artifacts/student-baseline-v1.json'
    import torch
    baseline = torch.load(SELECTED, map_location='cpu', weights_only=True)['metadata']['metrics']
    if not baseline_path.exists():
        with baseline_path.open('x') as f:
            json.dump({'metrics':{k:v for k,v in baseline.items() if k.endswith('map50_95')},
                       'source':str(SELECTED), 'sha256':SELECTED_SHA}, f, indent=2)
    config = replace(config, run_root=HERE/'artifacts/runs', baseline_metrics_path=baseline_path,
                     maximum_map_drop=.001)
    stage = replace(stage, epochs=5, patience=0, warmup_epochs=1)
    runtime.impl.JOINT_STAGES = MappingProxyType({**dict(runtime.impl.JOINT_STAGES), 'j3':stage})
    runtime.impl.SourceBundle = SelectedSource
    config._validate()
    assert config.preflight().ready
    return config, stage

class ProbeSession(runtime.formal.FormalJointTrainingSession):
    pass
