"""第 2 階段入口：S1／S2 共用訓練與監測 engine，完成即停止。"""

import argparse
import fcntl
import os
import time
from pathlib import Path

import torch

from .model import ROOT
from .run import GPUInUseError, write_json
from .scale_checkpoint import CANDIDATES
from .scale_study import ScaleStudy, verify_reference
from .supervise import run_child, verify_queue_source
from .train import run


def supervise(output, smoke=False, continue_queue=False):
    review = verify_reference()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'yolo_binaryqk/artifacts/stage2'):
        raise ValueError('Stage 2 output must stay in stage2 artifacts')
    if continue_queue and not output.is_dir():
        raise FileNotFoundError(output)
    output.mkdir(parents=True, exist_ok=continue_queue)
    with (output / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        policy = ScaleStudy()
        verify_queue_source(output, policy.source_paths)
        write_json(output / 'supervisor.json', {'pid': os.getpid(), 'stage': 2, 'status': 'starting',
                   'started_unix': started, 'smoke': smoke, 'next_stage_started': False})
        with (output / 'monitor.jsonl').open('a' if continue_queue else 'x', buffering=1) as monitor:
            try:
                for candidate in CANDIDATES:
                    code = run_child(output, candidate, 'scale_bias', smoke, monitor,
                                     parent=Path(review['parent']), policy=policy)
                    if code:
                        raise RuntimeError(f'{candidate} exit={code}')
                if not smoke:
                    from .scale_report import summarize
                    summarize(output)
                write_json(output / 'supervisor.json', {'pid': os.getpid(), 'stage': 2, 'status': 'completed',
                           'finished_unix': time.time(), 'elapsed_seconds': time.time() - started,
                           'smoke': smoke, 'next_stage_started': False})
            except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
                stopped = (output / 'STOP').exists()
                write_json(output / 'supervisor.json', {'pid': os.getpid(), 'stage': 2,
                           'status': 'stopped' if stopped else 'failed', 'error': f'{type(error).__name__}: {error}',
                           'next_stage_started': False})
                return 130 if stopped else 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    queue = sub.add_parser('queue')
    queue.add_argument('--continue-queue', action='store_true')
    train = sub.add_parser('train')
    train.add_argument('candidate', choices=CANDIDATES)
    train.add_argument('--parent', type=Path)
    train.add_argument('--resume', action='store_true')
    for command in (queue, train):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'queue':
        raise SystemExit(supervise(args.output, args.smoke, args.continue_queue))
    try:
        run(args.candidate, args.output, ROOT / 'yolo_binaryqk/artifacts/stage1/calibration.json',
            phase='scale_bias', parent=args.parent, smoke=args.smoke, resume=args.resume, policy=ScaleStudy())
    except GPUInUseError as error:
        print(str(error), flush=True)
        raise SystemExit(75) from error


if __name__ == '__main__':
    main()
