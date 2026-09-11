"""序列前置；正常 child.wait(600)，不讀 log 或查 GPU。"""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'artifacts/optimizer-probe-queue-v1';out.mkdir(exist_ok=False)
    with (out/'events.jsonl').open('x') as events:
        def emit(event,**kwargs):
            text=json.dumps({'event':event,'time':datetime.now(timezone.utc).isoformat(),**kwargs})
            events.write(text+'\n');events.flush();print(text,flush=True)
        for arm in ('adamw','mu-initial','mu-confirm'):
            with (out/(arm+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,str(HERE/'probe_optimizer.py'),'--arm',arm],
                                       stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
                emit('JOB_STARTED',arm=arm,pid=child.pid)
                while True:
                    try:code=child.wait(timeout=600);break
                    except subprocess.TimeoutExpired:pass
            if code:
                emit('ERROR',arm=arm,exit_code=code);return code
            result=json.loads((HERE/'artifacts/optimizer-probe-v1'/f'{arm}.json').read_text())
            if result['status']=='recipe_rejected':
                emit('ALL_DONE',recipe_accepted=False,reason=result['reason']);return 0
            assert result['status']=='probe_completed' and len(result['reports'])==16
            emit('JOB_DONE',arm=arm)
        emit('ALL_DONE',recipe_accepted=result['recipe_accepted'],ratios=result['group_update_ratios'])
    return 0

if __name__=='__main__':raise SystemExit(main())
