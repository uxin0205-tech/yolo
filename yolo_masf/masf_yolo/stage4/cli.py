"""Stage4 CLI；重用 Attention QueueExecutor，背景監測每 600 秒。"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import PROJECT, read, write_json


def supervise(output):
    from binaryqk.run import GPUInUseError, require_idle_gpu

    from .workflow import verify_source
    verify_source(output)
    with (output / 'monitor.lock').open('a') as lock, (output / 'monitor.jsonl').open('a', buffering=1) as stream:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if (output / 'STOP').exists():
                return 130
            try:
                require_idle_gpu()
                break
            except GPUInUseError as error:
                status = {'pid': os.getpid(), 'status': 'waiting-for-gpu', 'reason': str(error),
                          'time_unix': time.time(), 'monitor_interval_seconds': 600}
                write_json(output / 'supervisor.json', status)
                stream.write(json.dumps(status) + '\n')
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if (output / 'STOP').exists():
                        return 130
                    time.sleep(5)
        command = [sys.executable, '-m', 'masf_yolo.cli', 'stage4', 'worker', '--output', str(output), '--execute']
        with (output / 'worker.log').open('a') as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            started, next_check = time.time(), 0
            write_json(output / 'supervisor.json', {'pid': os.getpid(), 'child_pid': child.pid, 'status': 'running',
                       'started_unix': started, 'stage': 4, 'monitor_interval_seconds': 600})
            while child.poll() is None:
                now = time.time()
                if (output / 'STOP').exists():
                    os.killpg(child.pid, signal.SIGTERM)
                if now >= next_check:
                    try:
                        query = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                               timeout=15, check=False)
                        gpu = query.stdout.strip() or query.stderr.strip()
                    except subprocess.TimeoutExpired:
                        gpu = 'nvidia-smi timeout'
                    paths = [p for p in [output / 'progress.json', *output.glob('*/status.json')] if p.exists()]
                    latest = max(paths, key=lambda p: p.stat().st_mtime) if paths else None
                    event = {'time_unix': now, 'child_pid': child.pid, 'gpu': gpu,
                             'progress': read(latest) if latest else {'kind': 'loading'}}
                    last = max(started, latest.stat().st_mtime) if latest else started
                    if now - last > 1800:
                        os.killpg(child.pid, signal.SIGTERM)
                        event['action'] = 'terminate-own-child-after-1800s-no-progress'
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                    next_check = now + 600
                time.sleep(5)
            code = child.wait()
            complete = (output / 'comparison.json').exists() and read(output / 'comparison.json').get('ready')
            write_json(output / 'supervisor.json', {'pid': os.getpid(), 'stage': 4,
                       'status': 'completed' if code == 0 and complete else 'failed', 'exit_code': code,
                       'finished_unix': time.time(), 'elapsed_seconds': time.time() - started,
                       'next_stage_started': False})
            return code


def main(argv=None):
    import torch
    from yolo_attention.queue_executor import QueueExecutor
    from yolo_attention.queue_model import JobStatus
    from yolo_attention.queue_store import QueueStore

    from .workflow import Backend, initialize, verify_source
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('diagnostics', 'init', 'extension-init', 'status', 'run', 'worker', 'plot'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.action == 'diagnostics':
        from .diagnostics import run
        print(json.dumps(run(), ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        parser.error('--output required')
    output = args.output.resolve()
    if args.action == 'extension-init':
        from .extension_workflow import initialize as initialize_extension
        initialize_extension(output)
        return 0
    if args.action == 'init':
        initialize(output)
        return 0
    if args.action == 'plot':
        if (output / 'extension.json').exists():
            from .extension_report import plot
        else:
            from .report import plot
        plot(output)
        return 0
    store = QueueStore(output)
    if args.action == 'status':
        print(json.dumps([{'job': j.id, 'status': j.status.value} for j in store.load().jobs], indent=2))
        return 0
    executor = QueueExecutor(store, backend=Backend())
    if not args.execute:
        print(json.dumps(executor.preview_next().to_dict()))
        return 0
    if args.action == 'run':
        return supervise(output)
    verify_source(output)
    executor.run_all(execute=True)
    if any(j.status is not JobStatus.SUCCEEDED for j in store.load().jobs):
        raise RuntimeError('Stage4 queue incomplete')
    return 0
