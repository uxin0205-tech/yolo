"""P2 Pose MASF 八 epoch；不進行任何後續融合／activation。"""
from dataclasses import asdict
import json
from local_source import HERE
from j0_runtime import formal
from pose_p2_masf import install


def main():
    from j0_runtime import require_training_enabled
    require_training_enabled()
    config, stages = install()
    for name in ('pose-p2-probe-v1.json', 'fusion/j0-p2-smoke-v1/summary.json'):
        assert json.loads((HERE / 'artifacts' / name).read_text())['status'] == 'passed'
    class Session(formal.FormalJointTrainingSession):
        def _resolved_config(self):
            result = super()._resolved_config()
            result['stage_policies']['j0']['learning_rates'] = dict(stages['j0'].learning_rates)
            result['masf_location'] = 'P2 shared; existing backbone frozen, new MASF trainable'
            return result
    session = Session(config, device='0', run_name='j0-p2-masf-v1')
    report = session.run()
    with (session.run_dir / 'summary.json').open('x') as handle:
        json.dump(asdict(report), handle, ensure_ascii=False, indent=2, default=str)
    print('JOB_DONE: P2 Pose MASF 完成；需合併三組與 COCO 結果後停止', flush=True)


if __name__ == '__main__':
    main()
