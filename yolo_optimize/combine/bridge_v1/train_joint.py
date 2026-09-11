"""P3 bridge J1 正式聯合訓練；安全停在 checkpoint 邊界。"""
from dataclasses import asdict
import json
from safe_source import HERE
from runtime import install, Session, SafetyStop


def main():
    config, _ = install()
    assert json.loads((HERE / 'artifacts/fusion/j1-smoke-v1/summary.json').read_text())['status'] == 'passed'
    session = Session(config, device='0', run_name='j1-bridge-v1')
    try:
        report = session.run()
    except SafetyStop:
        print('JOB_DONE: SAFETY_STOP；該 epoch 已保存，需分析後再接續。', flush=True)
        return
    with (session.run_dir / 'summary.json').open('x') as handle:
        json.dump(asdict(report), handle, ensure_ascii=False, indent=2, default=str)
    print('JOB_DONE: J1 完成，接續 alpha 開關與階段分析。', flush=True)


if __name__ == '__main__':
    main()
