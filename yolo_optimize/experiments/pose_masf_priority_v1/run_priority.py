"""GPU 第一順位的獨立比較；需要明確 --execute，結束仍維持原訓練暫停。"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ATTENTION=HERE.parent/'attention_recovery_v1'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        if (HERE/'artifacts/final-audit-v1.json').exists():
            assert json.loads((HERE/'artifacts/final-audit-v1.json').read_text())['status']=='passed'
            print('COMPLETED Pose MASF 比較與分析已完成；原 Attention 訓練仍暫停。')
        else:
            print('READY GPU 第一順位：Pose MASF 比較；未啟動新工作。')
        return 0
    assert json.loads((HERE/'artifacts/preflight-v1.json').read_text())['status']=='passed'
    assert json.loads((ATTENTION/'artifacts/queue-v1/pause-outcome-v1.json').read_text())['status']=='PAUSED'
    lock=(ATTENTION/'artifacts/queue-v1/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=HERE/'artifacts/queue-v1';out.mkdir(parents=True,exist_ok=True)
    with (out/'events.jsonl').open('a') as events:
        def emit(status,**kw):
            line=json.dumps({'status':status,'time':datetime.now(timezone.utc).isoformat(),**kw},ensure_ascii=False)
            events.write(line+'\n');events.flush();print(line,flush=True)
        for case in ('baseline','pose_masf','pose_alpha_zero','report'):
            result=HERE/'artifacts/summary-v1.json' if case=='report' else HERE/'artifacts/comparison-v1'/case/'summary.json'
            if result.exists():
                assert json.loads(result.read_text())['status'] in ('passed','completed');continue
            attempt=1
            while (out/f'{case}-{attempt}.log').exists():attempt+=1
            log=out/f'{case}-{attempt}.log'
            with log.open('x') as stream:
                child=subprocess.Popen([sys.executable,str(HERE/'compare.py'),case],stdout=stream,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                emit('JOB_STARTED',job=case,pid=child.pid,priority=1,log=str(log))
                while True:
                    try:code=child.wait(timeout=60);break
                    except subprocess.TimeoutExpired:pass
            if code or not result.exists():emit('ERROR',job=case,exit_code=code,log=str(log));return code or 1
            assert json.loads(result.read_text())['status'] in ('passed','completed')
            emit('JOB_DONE',job=case,result=str(result))
        emit('ALL_DONE',scope='Pose MASF 推論比較',original_attention_training_still_paused=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
