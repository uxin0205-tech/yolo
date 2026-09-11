"""只跑成對 Pose MASF J0，結束即交回分析，不自動展開其他階段。"""
from dataclasses import asdict
import json
from local_source import HERE
from j0_runtime import install, formal, impl
from pose_masf import PoseMASFSource


def main():
    from j0_runtime import require_training_enabled
    require_training_enabled()
    config = install()
    assert json.loads((HERE / 'artifacts/pose-masf-probe-v1.json').read_text())['status'] == 'passed'
    assert json.loads((HERE / 'artifacts/fusion/j0-masf-smoke-v1/summary.json').read_text())['status'] == 'passed'
    impl.SourceBundle = PoseMASFSource
    session = formal.FormalJointTrainingSession(config, device='0', run_name='j0-pose-masf-v1')
    report = session.run()
    with (session.run_dir / 'summary.json').open('x') as handle:
        json.dump(asdict(report), handle, ensure_ascii=False, indent=2, default=str)
    print('JOB_DONE: Pose MASF 對照訓練結束；需呈現 COCO／ball／bat 並停止等待使用者決策', flush=True)


if __name__ == '__main__':
    main()
