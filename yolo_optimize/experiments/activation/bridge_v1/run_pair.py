"""配對 activation 串列工作：正常只 child.wait(600)，錯誤不啟動後續。"""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def main():
    root = HERE/'artifacts/paired-queue-v1'
    root.mkdir(exist_ok=False)
    with (root/'events.jsonl').open('x') as events:
        def emit(event,**data):
            rendered = json.dumps({'event':event,'time':datetime.now(timezone.utc).isoformat(),**data})
            events.write(rendered+'\n'); events.flush(); print(rendered,flush=True)
        for arm in ('silu','qsilu_pq'):
            run = HERE/f'artifacts/runs/{arm}-short-e10-seed1-v1'
            assert not run.exists(), f'禁止覆寫已有 run: {run}'
            with (root/(arm+'.log')).open('x') as log:
                child = subprocess.Popen([sys.executable,str(HERE/'train.py'),'--activation',arm],
                    cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                emit('JOB_STARTED',job=arm,pid=child.pid)
                while True:
                    try:
                        code = child.wait(timeout=600)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if code or not (run/'summary.json').exists() or (run/'safety-stop.json').exists():
                    emit('ERROR',job=arm,exit_code=code,log=str(root/(arm+'.log')),
                        reason='nonzero exit or incomplete/safety-stopped experiment; next job not started')
                    return 1
                report = json.loads((run/'summary.json').read_text())
                if report['epochs_completed'] != 10:
                    emit('ERROR',job=arm,reason='paired 10 epochs not completed')
                    return 1
                emit('JOB_DONE',job=arm,epochs=10,summary=str(run/'summary.json'))
        emit('ALL_DONE',note='SiLU/qSiLU paired short recovery only; review results before KD')
    return 0


if __name__ == '__main__': raise SystemExit(main())
