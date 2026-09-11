"""配對 smoke／訓練序列；成功檢查後接續、正常每次 wait 最多600秒。"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');smoke=p.parse_args().smoke
    root=HERE/('artifacts/smoke-queue-v1' if smoke else 'artifacts/training-queue-v1')
    root.mkdir(exist_ok=False)
    with (root/'events.jsonl').open('x') as events:
        def emit(event,**data):
            text=json.dumps({'event':event,'time':datetime.now(timezone.utc).isoformat(),**data})
            events.write(text+'\n');events.flush();print(text,flush=True)
        for arm in ('k0','spatial'):
            suffix='smoke-v1' if smoke else 'e5-seed1-v1'
            run=HERE/'artifacts/runs'/f'{arm}-{suffix}'
            assert not run.exists()
            with (root/(arm+'.log')).open('x') as log:
                command=[sys.executable,str(HERE/'train_pair.py'),'--arm',arm]+(['--smoke'] if smoke else [])
                child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                emit('JOB_STARTED',arm=arm,pid=child.pid)
                while True:
                    try:code=child.wait(timeout=600);break
                    except subprocess.TimeoutExpired:pass
            if code or not (run/'summary.json').exists():
                emit('ERROR',arm=arm,exit_code=code);return 1
            summary=json.loads((run/'summary.json').read_text())
            assert (summary['status']=='passed') if smoke else (summary['epochs_completed']==5)
            emit('JOB_DONE',arm=arm)
        emit('ALL_DONE',scope='smoke' if smoke else 'paired-five-epoch-training')
    return 0

if __name__=='__main__':raise SystemExit(main())
