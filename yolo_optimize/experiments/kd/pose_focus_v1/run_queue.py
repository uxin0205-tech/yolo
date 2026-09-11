"""前置／正式配對 queue；每次安靜 wait 最多600秒，錯誤停止接續。"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');pre=p.parse_args().preflight
    root=HERE/'artifacts'/('preflight-queue-v1' if pre else 'training-queue-v1');root.mkdir(parents=True,exist_ok=False)
    with (root/'events.jsonl').open('x') as events:
        def emit(event,**kwargs):
            s=json.dumps({'event':event,'time':datetime.now(timezone.utc).isoformat(),**kwargs})
            events.write(s+'\n');events.flush();print(s,flush=True)
        for arm in (('calibrate','native','kd') if pre else ('native','kd')):
            run=HERE/'artifacts/runs'/(arm+('-probe-v1' if pre else '-e5-seed1-v1'))
            assert not run.exists()
            with (root/(arm+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,str(HERE/('probe.py' if pre else 'train.py')),'--arm',arm],
                    cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                emit('JOB_STARTED',arm=arm,pid=child.pid)
                while True:
                    try:code=child.wait(timeout=600);break
                    except subprocess.TimeoutExpired:pass
            if code or not (run/'summary.json').exists():emit('ERROR',arm=arm,exit_code=code);return 1
            d=json.loads((run/'summary.json').read_text())
            assert (d['status']=='passed') if pre else (d['epochs_completed']==5)
            emit('JOB_DONE',arm=arm)
        emit('ALL_DONE',scope='preflight' if pre else 'pose-head-pair')
    return 0

if __name__=='__main__':raise SystemExit(main())
