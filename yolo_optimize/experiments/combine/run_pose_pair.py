"""等待現有 control，接 MASF smoke／訓練；完成後不啟動其他工作。"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    from j0_runtime import require_training_enabled
    require_training_enabled()
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-pid', type=int, required=True)
    args = parser.parse_args()
    state = HERE / 'artifacts/pose-pair-queue-v1.json'
    assert not state.exists()
    control = HERE / 'artifacts/fusion/j0-no-masf-v1/summary.json'
    if not control.exists():
        # pidfd 綁定具體程序，不因 PID 重用等待到別的工作。
        with open(f'/proc/{args.control_pid}/cmdline', 'rb') as handle:
            assert b'/combine/train_j0.py' in handle.read()
        descriptor = os.pidfd_open(args.control_pid)
        try:
            while not select.select([descriptor], [], [], 600)[0]:
                pass
        finally:
            os.close(descriptor)
    report = json.loads(control.read_text())
    assert report['epochs_completed'] == 8 and report['completed_stages'] == ['j0']
    for name, script in [('j0-masf-smoke-v1', 'smoke_pose_masf.py'),
                         ('j0-pose-masf-v1', 'train_pose_masf.py')]:
        subprocess.run([sys.executable, str(HERE / 'monitor.py'), '--name', name, '--',
                        sys.executable, str(HERE / script)], cwd=HERE, check=True)
    with state.open('x') as handle:
        json.dump({'status': 'ALL_DONE_REQUIRES_COMPARISON_AND_USER_DECISION',
            'control': str(control), 'candidate': str(HERE / 'artifacts/fusion/j0-pose-masf-v1/summary.json'),
            'followup_training_allowed': False, 'wait_seconds': 600}, handle, indent=2)
    print('ALL_DONE: Pose MASF 配對完成；呈現結果後停止，等使用者決定', flush=True)


if __name__ == '__main__':
    main()
