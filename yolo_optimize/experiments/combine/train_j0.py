"""沿用 combine 正式 J0，完成後保留結果供階段分析。"""
from dataclasses import asdict
import json
from j0_runtime import install, formal
from local_source import HERE


def main():
    from j0_runtime import require_training_enabled
    require_training_enabled()
    config = install()
    smoke = HERE / 'artifacts/fusion/j0-smoke-v1/summary.json'
    assert json.loads(smoke.read_text())['status'] == 'passed'
    session = formal.FormalJointTrainingSession(config, device='0', run_name='j0-no-masf-v1')
    report = session.run()
    with (session.run_dir / 'summary.json').open('x') as handle:
        json.dump(asdict(report), handle, ensure_ascii=False, indent=2, default=str)
    print('JOB_DONE: J0 結束；需比較原 Pose baseline 後決定共享訓練', flush=True)


if __name__ == '__main__':
    main()
