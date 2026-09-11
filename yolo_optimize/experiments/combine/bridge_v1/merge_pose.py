"""已驗證 10% Pose trunk 的融合 J0；完整 Pose best 獨立保留。"""
from dataclasses import replace
from types import MappingProxyType
import json
import sys
import torch
import pose_first as base
from full_pose import FullPoseSource
from safe_source import HERE, SOURCE, sha256
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from pwl_contract import verify_pwl

POSE_BEST = HERE / 'artifacts/fusion/full-pose-gentle-v1/inference/best_pose.pt'
POSE_SHA = '68d6be769751aaf7ce15d552f6b8d7f1e6f4bb5e463a530bbc976e655959f117'
INITIAL = HERE / 'artifacts/full-pose-assembly-v1/summary.json'


class MergeSource(FullPoseSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        measured = json.loads(INITIAL.read_text())['results']['0.1']
        self.record['detect_expected_metrics'] = {k: measured[k] for k in self.record['detect_expected_metrics']}

    def provenance(self, kind='float'):
        return {**super().provenance(kind), 'source_kind': '10percent_pose_trunk_merge_initialization',
            'full_pose_best': str(POSE_BEST), 'full_pose_sha256': POSE_SHA,
            'pose_trunk_ratio': .1, 'source_export_is_original_anchor_not_merged_file': True}

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models('float', pose_head_checkpoint=pose_head_checkpoint)
        assert sha256(POSE_BEST) == POSE_SHA
        full = torch.load(POSE_BEST, map_location='cpu', weights_only=True)['state_dict']
        for model in (pair.detect, pair.pose):
            state = model.state_dict()
            for name, value in state.items():
                if int(name.split('.')[1]) >= 23:
                    continue
                target = full['graph.'+name]
                if value.dtype.is_floating_point and not torch.equal(value, target):
                    state[name] = value + .1*(target-value)
            model.load_state_dict(state, strict=True)
        prefix = 'graph.model.23.pose_head.'
        pair.pose.model[23].load_state_dict({k[len(prefix):]: v for k,v in full.items() if k.startswith(prefix)}, strict=True)
        for model in (pair.detect, pair.pose):
            convert_yolo26_model(model, VariantConfig.from_yaml(SOURCE / f'configs/attention/{kind}-pwl-final.yaml'))
            verify_pwl(model)
        return pair


original_install = base.install
original_session = base.Session


def install():
    config, stage = original_install()
    path = HERE / 'artifacts/full-pose-assembly-v1/j0-baseline.json'
    if not path.exists():
        original = config.load_baseline()
        measured = json.loads(INITIAL.read_text())['results']['0.1']
        values = {k: measured[k] if k.startswith('coco/') else v for k,v in original.items()}
        with path.open('x') as f:
            json.dump({'metrics': values, 'note': 'J0 invariance baseline only; joint COCO gate retains original Detect anchor'}, f, indent=2)
    config = replace(config, baseline_metrics_path=path)
    stage = replace(stage, epochs=20)
    base.impl.JOINT_STAGES = MappingProxyType({**dict(base.impl.JOINT_STAGES), 'j0': stage})
    base.impl.SourceBundle = MergeSource
    return config, stage


class Session(original_session):
    def __init__(self, *args, **kwargs):
        kwargs['run_name'] = {'j0-extend-smoke-v1': 'merge-j0-smoke-v1',
            'j0-pose-extend-v1': 'merge-j0-v1'}[kwargs['run_name']]
        super().__init__(*args, **kwargs)

    def _resolved_config(self):
        result = super()._resolved_config()
        result['stage_policies']['j0']['epochs'] = 20
        result['pose_first_extension'].update(epochs=20, source='full Pose E30 head plus 10% Pose trunk',
            original_detect_anchor_retained=True, initial_pose_ap=.84745105)
        return result


if __name__ == '__main__':
    base.install = install
    base.Session = Session
    base.AdaptedBridgeSource = MergeSource
    if '--smoke' not in sys.argv:
        assert json.loads((HERE / 'artifacts/fusion/merge-j0-smoke-v1/summary.json').read_text())['status'] == 'passed'
    base.main()
