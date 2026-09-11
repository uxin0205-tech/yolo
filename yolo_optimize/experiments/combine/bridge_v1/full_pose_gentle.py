"""完整 Pose 同起點單變因：全部 role LR ×0.1；保留首輪。"""
from dataclasses import replace
from types import MappingProxyType
import full_pose as base

original_install = base.install
original_session = base.Session


def install():
    config, stage = original_install()
    stage = replace(stage, learning_rates=MappingProxyType({k: v*.1 for k,v in stage.learning_rates.items()}))
    base.pose_first.impl.JOINT_STAGES = MappingProxyType({**dict(base.pose_first.impl.JOINT_STAGES), 'j0': stage})
    return config, stage


class Session(original_session):
    def __init__(self, *args, **kwargs):
        kwargs['run_name'] = {'full-pose-smoke-v2': 'full-pose-gentle-smoke-v1',
            'full-pose-adamw-v1': 'full-pose-gentle-v1'}[kwargs['run_name']]
        super().__init__(*args, **kwargs)

    def _resolved_config(self):
        result = super()._resolved_config()
        result['full_pose_policy'].update(head_lr=2e-5, neck_lr=7.5e-6, backbone_lr=1.5e-6,
            controlled_change='all role learning rates x0.1; same source, scope, seed, data, horizon')
        return result


if __name__ == '__main__':
    base.install = install
    base.Session = Session
    import sys
    import json
    if '--smoke' not in sys.argv:
        assert json.loads((base.HERE / 'artifacts/fusion/full-pose-gentle-smoke-v1/summary.json').read_text())['status'] == 'passed'
    base.main()
