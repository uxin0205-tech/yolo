#!/usr/bin/env python3
"""安靜等待指定 Linux PID 結束；不取樣 GPU、不讀訓練 log。

使用 pidfd 固定程序身分。正常執行只由 kernel 等待，每次最多 600 秒。
此模式不憑空推定 STALLED；需要工作本身退出或另外的明確狀態事件。
"""
import argparse
import json
import os
import select
import sys

sys.dont_write_bytecode = True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--old-supervisor', type=int, required=True)
    args = parser.parse_args()
    # 在暫停舊監督器前取得穩定身分，避免 PID 重用。
    job = os.pidfd_open(args.pid)
    supervisor = os.pidfd_open(args.old_supervisor)
    import signal
    command = open(f'/proc/{args.old_supervisor}/cmdline', 'rb').read()
    if b'supervise_recovery.py' not in command:
        raise RuntimeError('舊 supervisor 身分不符，拒絕發送信號')
    try:
        signal.pidfd_send_signal(supervisor, signal.SIGSTOP)
        while not select.select([job], [], [], 600)[0]:
            pass
        print(json.dumps({'event': 'JOB_EXIT', 'pid': args.pid,
                          'exit_code': None, 'requires_result_classification': True}), flush=True)
    finally:
        # 讓原 parent 回收 child 並保存真正 exit code；只在事件或本監測結束時執行。
        try:
            signal.pidfd_send_signal(supervisor, signal.SIGCONT)
        except ProcessLookupError:
            pass
        os.close(job)
        os.close(supervisor)

if __name__ == '__main__':
    main()
