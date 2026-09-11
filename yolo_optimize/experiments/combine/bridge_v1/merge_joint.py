"""完整 Pose 訓練及融合 J0 後，低 LR 聯合 J1。"""
from dataclasses import replace
from types import MappingProxyType
import json
import sys
import torch
from safe_source import HERE, sha256
from merge_pose import MergeSource
import runtime
import train_joint
import smoke_joint

J0 = HERE / 'artifacts/fusion/merge-j0-v1/inference/best_pose.pt'


class JointSource(MergeSource):
    def provenance(self, kind='float'):
        return {**super().provenance(kind), 'merged_pose_adaptation': str(J0),
            'merged_pose_adaptation_sha256': sha256(J0), 'phase': 'joint_J1_after_full_pose_and_merge_J0'}

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        payload = torch.load(J0, map_location='cpu', weights_only=True)
        assert payload['metadata']['epoch'] == 0 and payload['metadata']['stage'] == 'j0'
        prefix = 'graph.model.23.pose_head.'
        pair.pose.model[23].load_state_dict({n[len(prefix):]: v for n,v in payload['state_dict'].items()
            if n.startswith(prefix)}, strict=True)
        return pair


original_install = runtime.install
original_session = runtime.Session


def install():
    config, stage = original_install()
    stage = replace(stage, learning_rates=MappingProxyType({k:v*.1 for k,v in stage.learning_rates.items()}))
    runtime.impl.JOINT_STAGES = MappingProxyType({**dict(runtime.impl.JOINT_STAGES), 'j1': stage})
    runtime.impl.SourceBundle = JointSource
    return config, stage


class Session(original_session):
    def __init__(self, *args, **kwargs):
        kwargs['run_name'] = {'j1-smoke-v1':'merge-j1-smoke-v1', 'j1-bridge-v1':'merge-j1-v1'}[kwargs['run_name']]
        super().__init__(*args, **kwargs)

    def _resolved_config(self):
        result = super()._resolved_config()
        result['joint_after_full_pose'] = {'pose_source': str(J0), 'pose_source_sha256':sha256(J0),
            'all_j1_learning_rates_multiplier': .1,
            'coco_protection_against_original_anchor': True, 'independent_pose_checkpoint_preserved': True}
        return result


if __name__ == '__main__':
    train_joint.install = smoke_joint.install = install
    train_joint.Session = smoke_joint.Session = Session
    smoke_joint.AdaptedBridgeSource = JointSource
    if '--smoke' in sys.argv:
        smoke_joint.main()
    else:
        assert json.loads((HERE / 'artifacts/fusion/merge-j1-smoke-v1/summary.json').read_text())['status'] == 'passed'
        train_joint.main()
