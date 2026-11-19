"""第 3 階段 CLI 接線與 600 秒監測；job 執行仍交給既有 QueueExecutor。"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from .queue_backend import ResearchQueueBackend
from .queue_executor import QueueExecutor
from .queue_model import JobStatus
from .queue_store import QueueStore
from .stage3_checkpoint import PROJECT
from .stage3_diagnostics import numeric_reference, write_json
from .stage3_workflow import initialize, verify_source


def add_parser(subparsers):
    parser = subparsers.add_parser('stage3', help='Integrated S1 → normalization/PWL/reciprocal research')
    parser.add_argument('action', choices=('reference', 'init', 'status', 'run', 'worker', 'plot'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    kd = subparsers.add_parser('stage3-kd', help='D1 KD recovery within Stage 3')
    kd.add_argument('action', choices=('init', 'status', 'run', 'worker', 'plot'))
    kd.add_argument('--output', type=Path, required=True)
    kd.add_argument('--execute', action='store_true')


def supervise(output, command_name='stage3', verifier=verify_source):
    from binaryqk.run import GPUInUseError, require_idle_gpu
    verifier(output)
    with (output/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output/'monitor.jsonl').open('a', buffering=1) as stream:
            while True:
                if (output/'STOP').exists():
                    return 130
                try:
                    require_idle_gpu()
                    break
                except GPUInUseError as error:
                    status = {'pid': os.getpid(), 'status': 'waiting-for-gpu', 'reason': str(error),
                              'time_unix': time.time(), 'monitor_interval_seconds': 600}
                    write_json(output/'supervisor.json', status)
                    stream.write(json.dumps(status)+'\n')
                    deadline = time.monotonic()+600
                    while time.monotonic() < deadline:
                        if (output/'STOP').exists():
                            return 130
                        time.sleep(5)
            command = [sys.executable, '-m', 'yolo_attention.cli', command_name, 'worker',
                       '--output', str(output), '--execute']
            with (output/'worker.log').open('a') as log:
                child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.time()
                write_json(output/'supervisor.json', {'pid': os.getpid(), 'child_pid': child.pid,
                           'status': 'running', 'started_unix': started, 'stage': 3,
                           'monitor_interval_seconds': 600, 'next_stage_started': False})
                next_check = 0
                while child.poll() is None:
                    now = time.time()
                    if now >= next_check:
                        try:
                            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                                  '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                                 timeout=15, check=False)
                            gpu_status = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
                        except subprocess.TimeoutExpired:
                            gpu_status = 'nvidia-smi timeout; child progress monitoring continues'
                        paths = [output/'progress.json']
                        if command_name == 'stage3-kd':
                            paths.extend((output/'smoke/status.json', output/'training/status.json'))
                        existing = [p for p in paths if p.exists()]
                        path = max(existing, key=lambda p: p.stat().st_mtime) if existing else paths[0]
                        progress = json.loads(path.read_text()) if path.exists() else {'kind': 'loading'}
                        event = {'time_unix': now, 'child_pid': child.pid, 'gpu': gpu_status,
                                 'progress': progress}
                        last = max(started, path.stat().st_mtime) if path.exists() else started
                        if now-last > 1800:
                            os.killpg(child.pid, signal.SIGTERM)
                            event['action'] = 'terminate-own-child-after-1800s-no-progress'
                        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
                        next_check = now+600
                    time.sleep(5)
                code = child.wait()
                decision = json.loads((output/'comparison.json').read_text()) if (output/'comparison.json').exists() else {}
                status = ('completed' if decision.get('ready') else 'needs-followup') if code == 0 else 'failed'
                write_json(output/'supervisor.json', {'pid': os.getpid(), 'stage': 3, 'status': status,
                           'exit_code': code, 'finished_unix': time.time(), 'elapsed_seconds': time.time()-started,
                           'next_stage_started': False})
                return code


def run(args):
    torch.set_num_threads(2)
    initializer, verifier = initialize, verify_source
    if args.command == 'stage3-kd':
        from .stage3_kd_workflow import initialize as initializer
        from .stage3_kd_workflow import verify_source as verifier
    if args.action == 'reference':
        result = numeric_reference()
        print(json.dumps({'passed': result['passed'], 'reciprocal_backends': list(result['reciprocal'])}))
        return 0
    if args.output is None:
        raise ValueError('--output is required for this Stage 3 command')
    output = args.output.resolve()
    if args.action == 'init':
        initializer(output)
        print(json.dumps({'initialized': str(output), 'gpu_started': False}))
        return 0
    if args.action == 'plot':
        if args.command == 'stage3-kd':
            from .stage3_kd_workflow import plot
        else:
            from .stage3_report import plot
        plot(output)
        return 0
    store = QueueStore(output)
    if args.action == 'status':
        print(json.dumps({'jobs': [{'id': j.id, 'status': j.status.value} for j in store.load().jobs]}, indent=2))
        return 0
    executor = QueueExecutor(store, backend=ResearchQueueBackend(project_root=PROJECT))
    if not args.execute:
        print(json.dumps(executor.preview_next().to_dict()))
        return 0
    if args.action == 'run':
        return supervise(output, args.command, verifier)
    verifier(output)
    executor.run_all(execute=True)
    if any(job.status is not JobStatus.SUCCEEDED for job in store.load().jobs):
        raise RuntimeError('Stage 3 queue did not reach complete terminal states')
    return 0
