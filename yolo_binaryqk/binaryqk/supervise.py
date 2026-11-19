"""第 1 階段：三分支初篩→winner 恢復；600 秒監測，完成後停止。"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import checkpoint
from .basis import BASES
from .model import ROOT, sha256
from .protocol import recipe_for, should_stop, verify_reference_review
from .run import GPUInUseError, require_idle_gpu, write_json


def verify_queue_source(output, extra_paths=()):
    current = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
    current.update({str(path.relative_to(ROOT)): sha256(path) for path in extra_paths})
    path = output / 'source.json'
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError('queue source changed; review before continuing experiments')
    else:
        write_json(path, current)


def candidate_action(destination, variant, smoke=False, phase='screening', parent=None, policy=None):
    """已完成者須通過 recipe、完整評估與 checkpoint；未完者由 trainer 驗證 resume。"""
    if not destination.exists():
        return 'fresh'
    status_file = destination / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    if status.get('kind') == 'completed':
        result = json.loads((destination / 'result.json').read_text())
        recipe = policy.recipe(phase) if policy else recipe_for(phase)
        if (result.get('variant') != variant or result.get('smoke') != smoke
                or result.get('phase') != phase or result.get('recipe') != recipe):
            raise ValueError('completed candidate identity/recipe mismatch')
        history = result['history']
        last = len(history) - 1
        if [row['epoch'] for row in history] != list(range(last + 1)) or last < 1:
            raise ValueError('completed candidate has incomplete epoch history')
        valid_end = last == (1 if smoke else recipe['epochs'])
        if not smoke and phase == 'recovery' and result['stop_reason'] == 'patience':
            valid_end |= last <= recipe['epochs'] and should_stop(recipe, last, result['best']['epoch'])
        if not valid_end:
            raise ValueError('completed candidate did not meet budget/stopping rule')
        if not smoke and any(row['evaluation'].get('images') != 5000
                             or row['evaluation'].get('formal') is not True for row in history):
            raise ValueError('completed candidate lacks full formal evaluation')
        name = result['best']['checkpoint']
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('invalid best checkpoint path')
        _, manifest = (policy.codec if policy else checkpoint).load(destination / name)
        identity = manifest.get('scale_candidate', manifest['variant'])
        if identity != variant or manifest['lineage']['phase'] != phase:
            raise ValueError('completed checkpoint variant/phase mismatch')
        if parent is not None and manifest['lineage']['parent_sha256'] != sha256(parent):
            raise ValueError('completed recovery parent mismatch')
        return 'skip-completed'
    if (destination / 'resume.pt').is_file() and (destination / 'resume.json').is_file():
        return 'resume'
    raise ValueError(f'incomplete candidate without verified epoch resume: {destination}')


def wait_for_gpu(output, variant, monitor):
    while True:
        if (output / 'STOP').exists():
            return False
        try:
            require_idle_gpu()
            return True
        except GPUInUseError as error:
            event = {'time_unix': time.time(), 'pid': os.getpid(), 'status': 'waiting-for-gpu',
                     'variant': variant, 'reason': str(error), 'monitor_interval_seconds': 600,
                     'next_stage_started': False}
            write_json(output / 'supervisor.json', event)
            monitor.write(json.dumps(event, ensure_ascii=False) + '\n')
        next_check = time.monotonic() + 600
        while time.monotonic() < next_check:
            if (output / 'STOP').exists():
                return False
            time.sleep(max(0, min(5, next_check - time.monotonic())))


def run_child(output, variant, phase, smoke, monitor, parent=None, policy=None):
    if policy:
        verify_queue_source(output, policy.source_paths)
    else:
        verify_queue_source(output)
    destination = output / (variant if phase == 'screening' or policy else 'recovery')
    action = (candidate_action(destination, variant, smoke, phase, parent, policy=policy) if policy
              else candidate_action(destination, variant, smoke, phase, parent))
    if action == 'skip-completed':
        monitor.write(json.dumps({'time_unix': time.time(), 'variant': variant, 'phase': phase,
                                  'event': 'verified-completed-checkpoint-skip'}) + '\n')
        return 0
    command = ([sys.executable, '-m', 'binaryqk.scale', 'train', variant, '--output', str(destination)] if policy else
               [sys.executable, '-m', 'binaryqk.train', variant, '--output', str(destination), '--phase', phase])
    if parent is not None:
        command.extend(['--parent', str(parent)])
    if smoke:
        command.append('--smoke')
    if action == 'resume':
        command.append('--resume')
    while True:
        if not wait_for_gpu(output, variant, monitor):
            return 130
        with (output / f'{destination.name}.log').open('a') as log:
            log.write(f'\nQueue launch {time.time()} action={action} phase={phase}\n')
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            write_json(output / 'supervisor.json', {'pid': os.getpid(), 'child_pid': child.pid,
                       'status': 'running', 'variant': variant, 'phase': phase,
                       'monitor_interval_seconds': 600, 'smoke': smoke, 'next_stage_started': False})
            child_started, next_check = time.time(), 0.
            while child.poll() is None:
                now = time.time()
                if (output / 'STOP').exists() and destination.exists():
                    (destination / 'STOP').touch(exist_ok=True)
                if now >= next_check:
                    try:
                        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                              '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                             timeout=15, check=False)
                        gpu_status = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
                    except subprocess.TimeoutExpired:
                        gpu_status = 'nvidia-smi timeout; child progress still monitored'
                    status_path = destination / 'status.json'
                    status = json.loads(status_path.read_text()) if status_path.exists() else {'kind': 'loading'}
                    event = {'time_unix': now, 'variant': variant, 'phase': phase, 'child_pid': child.pid,
                             'gpu': gpu_status, 'progress': status}
                    monitor.write(json.dumps(event, ensure_ascii=False) + '\n')
                    last_progress = max(child_started, status_path.stat().st_mtime) if status_path.exists() else child_started
                    if now - last_progress > 1800:
                        os.killpg(child.pid, signal.SIGTERM)
                        event['action'] = 'terminate-own-child-after-1800s-no-progress'
                        monitor.write(json.dumps(event, ensure_ascii=False) + '\n')
                    next_check = now + 600
                time.sleep(5)
            code = child.wait()
            monitor.write(json.dumps({'time_unix': time.time(), 'variant': variant, 'phase': phase,
                                      'exit_code': code}) + '\n')
        if code == 75:
            continue
        return code


def screening_selection(output, smoke):
    if smoke:
        selected = 'B3'
    else:
        results = {v: json.loads((output / v / 'result.json').read_text()) for v in BASES}
        if any(r['phase'] != 'screening' or r['smoke'] or len(r['history']) != 11 for r in results.values()):
            raise ValueError('selection requires all three complete 10-epoch screening results')
        selected = max(BASES, key=lambda v: (results[v]['best']['AP50_95'], -len(BASES[v])))
        score = results[selected]['best']['AP50_95']
        floor = verify_reference_review()['source_results']['native_official'] * .90
        write_json(output / 'screening-selection.json', {'variant': selected, 'AP50_95': score,
                   'required_minimum': floor, 'passed': score >= floor,
                   'rule': 'best official AP; ties fewer bases then B1/B2/B3 order',
                   'scores': {v: r['best']['AP50_95'] for v, r in results.items()}})
        if score < floor:
            raise ValueError('screening quality gate failed; diagnose before full-model recovery')
    result = json.loads((output / selected / 'result.json').read_text())
    return selected, output / selected / result['best']['checkpoint']


def supervise(output, smoke=False, continue_queue=False):
    verify_reference_review()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'yolo_binaryqk/artifacts/stage1'):
        raise ValueError('supervisor output must stay inside stage1 artifacts')
    if continue_queue and not output.is_dir():
        raise FileNotFoundError('continuation requires an existing queue')
    output.mkdir(parents=True, exist_ok=continue_queue)
    with (output / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _supervise_locked(output, smoke, continue_queue)


def _supervise_locked(output, smoke, continue_queue):
    started = time.time()
    previous_path = output / 'supervisor.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    write_json(output / 'supervisor.json', {'pid': os.getpid(), 'started_unix': started,
               'stage': 1, 'monitor_interval_seconds': 600, 'smoke': smoke, 'status': 'starting',
               'continuation': continue_queue, 'next_stage_started': False})
    with (output / 'monitor.jsonl').open('a' if continue_queue else 'x', buffering=1) as monitor:
        if continue_queue:
            monitor.write(json.dumps({'time_unix': started, 'event': 'queue-continuation',
                                      'previous_supervisor': previous}) + '\n')
        try:
            for variant in BASES:
                code = run_child(output, variant, 'screening', smoke, monitor)
                if code:
                    raise RuntimeError(f'{variant} screening exit={code}')
            selected, parent = screening_selection(output, smoke)
            code = run_child(output, selected, 'recovery', smoke, monitor, parent)
            if code:
                raise RuntimeError(f'{selected} recovery exit={code}')
            if not smoke:
                from .report import summarize
                summarize(output)
            write_json(output / 'supervisor.json', {'pid': os.getpid(), 'status': 'completed',
                       'finished_unix': time.time(), 'elapsed_seconds': time.time() - started,
                       'stage': 1, 'smoke': smoke, 'next_stage_started': False})
        except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
            stopped = (output / 'STOP').exists()
            write_json(output / 'supervisor.json', {'pid': os.getpid(), 'status': 'stopped' if stopped else 'failed',
                       'error': f'{type(error).__name__}: {error}', 'next_stage_started': False})
            return 130 if stopped else 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--continue-queue', action='store_true')
    args = parser.parse_args()
    raise SystemExit(supervise(args.output, args.smoke, args.continue_queue))


if __name__ == '__main__':
    main()
