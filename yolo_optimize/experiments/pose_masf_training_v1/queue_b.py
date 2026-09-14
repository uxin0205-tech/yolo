"""B 組獨立 GPU 佇列；先等空閒，600 秒監測，不終止任何外部工作。"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ART=HERE/'artifacts/queue-b-v1'
ATTENTION=HERE.parent/'attention_recovery_v1/artifacts/queue-v1'

def gpu_users():
    result=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=20)
    return [int(p.strip()) for p in result.stdout.splitlines() if p.strip()]

def wait_interval():
    # 每次阻塞不超過 60 秒；十段由 Python 自行等待，不需要模型輪詢。
    until=time.monotonic()+600
    while time.monotonic()<until:time.sleep(max(0,min(60,until-time.monotonic())))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:print('READY：B only，GPU 空閒後 smoke → 5 epoch → alpha-off 分析');return 0
    assert json.loads((HERE/'artifacts/cpu-preflight-v1/summary.json').read_text())['status']=='passed'
    check=json.loads((HERE/'artifacts/queue-preflight-v1.json').read_text())
    assert check['status']=='passed'
    def verify_sources():
        for name,digest in check['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,'排程程式已變更，必須先重新檢查：'+name
    verify_sources()
    assert (ATTENTION/'pause-request.json').exists(),'原 Attention 暫停保護不存在'
    ART.mkdir(parents=True,exist_ok=True)
    own=(ART/'queue.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def emit(status,**kw):
        data={'status':status,'time':datetime.now(timezone.utc).isoformat(),'queue_pid':os.getpid(),**kw}
        line=json.dumps(data,ensure_ascii=False)
        with (ART/'events.jsonl').open('a') as f:f.write(line+'\n')
        tmp=ART/'state.tmp';tmp.write_text(line+'\n');tmp.replace(ART/'state.json')
        print(line,flush=True)
    shared=(ATTENTION/'worker.lock').open('a')
    waiting=False
    while True:
        try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
        except BlockingIOError:locked=False
        users=gpu_users()
        if locked and not users:break
        if locked:fcntl.flock(shared,fcntl.LOCK_UN)
        if not waiting:emit('WAITING_GPU',external_pids=users,interval_seconds=600);waiting=True
        wait_interval()
    jobs=[('smoke',HERE/'artifacts/smoke-v1/summary.json'),('train',HERE/'artifacts/b-e5-seed1-v1/summary.json'),
          ('analyze',HERE/'artifacts/b-e5-seed1-v1/analysis/summary.json'),
          ('finalize',HERE/'artifacts/b-e5-seed1-v1/final-audit.json')]
    for job,result in jobs:
        verify_sources()
        if result.exists():
            assert json.loads(result.read_text())['status']=='completed';continue
        # 外部程序不使用共享 lock；每個新 job 前再做一次 admission，不與它競爭。
        if job!='finalize' and gpu_users():
            emit('WAITING_GPU',job=job,interval_seconds=600)
            while gpu_users():wait_interval()
        attempt=1
        while (ART/f'{job}-{attempt}.log').exists():attempt+=1
        log=ART/f'{job}-{attempt}.log'
        with log.open('x') as f:
            command=[sys.executable,'-B',str(HERE/'finalize_b.py')] if job=='finalize' else [sys.executable,'-B',str(HERE/'training_b.py'),job]
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'' if job=='finalize' else '0'})
            emit('JOB_STARTED',job=job,pid=child.pid,log=str(log))
            last_monitor=time.monotonic();stalled=False
            while True:
                try:code=child.wait(timeout=60);break
                except subprocess.TimeoutExpired:pass
                if time.monotonic()-last_monitor>=600:
                    last_monitor=time.monotonic()
                    # 不讀 log；僅以輸出／macro 心跳的 mtime 判斷 30 分鐘無活動。
                    heart=HERE/('artifacts/smoke-v1/heartbeat' if job=='smoke' else 'artifacts/b-e5-seed1-v1/heartbeat')
                    latest=max(p.stat().st_mtime for p in (log,heart) if p.exists())
                    if time.time()-latest>1800 and not stalled:
                        emit('STALLED',job=job,pid=child.pid,log=str(log));stalled=True
                    elif stalled and time.time()-latest<=1800:
                        emit('RECOVERED',job=job,pid=child.pid);stalled=False
        if code or not result.exists():emit('ERROR',job=job,exit_code=code,log=str(log));return code or 1
        assert json.loads(result.read_text())['status']=='completed'
        emit('JOB_DONE',job=job,result=str(result))
    emit('ALL_DONE',report=str(HERE/'RESULTS.md'),original_attention_paused=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
