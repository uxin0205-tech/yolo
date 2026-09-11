"""融合區獨立事件監測，正常僅等待 child，最多600秒一次。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.name.replace('-', '').isalnum()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    root = Path(__file__).resolve().parent
    logs = root / 'artifacts/logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f'{args.name}.log').open('x') as log, (logs / f'{args.name}.events.jsonl').open('x') as events:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        def emit(event, **extra):
            data = {'event': event, 'name': args.name, 'pid': child.pid, 'supervisor_pid': os.getpid(),
                    'time': datetime.now(timezone.utc).isoformat(), **extra}
            rendered = json.dumps(data, ensure_ascii=False)
            events.write(rendered + '\n')
            events.flush()
            print(rendered, flush=True)
        emit('JOB_STARTED')
        while True:
            try:
                code = child.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                pass
        emit('JOB_DONE' if code == 0 else 'ERROR', exit_code=code)
        return code


if __name__ == '__main__':
    raise SystemExit(main())
