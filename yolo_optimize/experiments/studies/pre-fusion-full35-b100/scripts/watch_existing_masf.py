"""以 Linux pidfd 接回孤立的既有工作；只等待，不重啟或發送信號。"""
import argparse
import json
import os
from pathlib import Path
import select
from common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    destination = ROOT / 'artifacts/masf-task-monitor-recovery-v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    descriptor = os.pidfd_open(args.pid)
    command = Path(f'/proc/{args.pid}/cmdline').read_bytes().split(b'\0')
    assert str(ROOT / 'scripts/train_masf_task_bridge.py').encode() in command
    assert b'masf-task-bridge-v1' in command
    report = {'status': 'waiting', 'pid': args.pid, 'wait_seconds': 600,
              'pidfd_identity_pinned': True, 'job_restarted': False}
    write_json(destination, report)
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    print('MONITOR_ATTACHED: 既有 MASF 工作，pidfd 最多等待 600 秒', flush=True)
    try:
        while not poller.poll(600000):
            pass
    finally:
        os.close(descriptor)
    # 非本程序 child，無法取得 exit code；須核對實際完成摘要。
    path = ROOT / 'artifacts/masf-task-bridge-v1/summary.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    if current.get('status') != 'complete':
        report.update(status='needs_diagnosis', observed_summary_status=current.get('status'),
                      child_exit_code_known=False)
        write_json(destination, report)
        print('ERROR_OR_REVIEW: 工作已退出但未完整完成，需核對必要資訊', flush=True)
        return 1
    baseline = json.loads((ROOT / 'artifacts/masf-head-fork-v1/summary.json').read_text())
    assert current['first_macro_trace_sha256'] == baseline['first_macro_trace_sha256']
    assert [r['epoch'] for r in current['epochs']] == list(range(6, 11))
    assert all(r['images'] == 118287 and r['macros'] == 925 for r in current['epochs'])
    comparisons = [{'epoch': b['epoch'], 'native': a['ema'], 'bridge': b['ema'],
                    'delta': {k: b['ema'][k] - a['ema'][k] for k in a['ema']}}
                   for a, b in zip(baseline['epochs'], current['epochs'])]
    report.update(status='awaiting_analysis', comparisons=comparisons, child_exit_code_known=False)
    write_json(destination, report)
    print('ALL_DONE: 已核對 E6–E10 全量摘要並保存配對結果；等待分析', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
