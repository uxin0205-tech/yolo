"""使用者取消原生對照後的背景 queue：只跑 KD，再整理；無模型 API 呼叫。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent

def main():
    root=HERE/'artifacts/direct-queue-v1';root.mkdir(exist_ok=False)
    assert not (HERE/'artifacts/runs/kd-e5-seed1-v1').exists()
    jobs=[('pose-head-kd',[sys.executable,str(HERE/'train.py'),'--arm','kd']),
          ('summarize',[sys.executable,str(HERE/'summarize_direct.py')])]
    with (root/'events.jsonl').open('x') as events:
        def emit(event,**data):
            s=json.dumps({'event':event,'time':datetime.now(timezone.utc).isoformat(),**data})
            events.write(s+'\n');events.flush();print(s,flush=True)
        for name,command in jobs:
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                emit('JOB_STARTED',job=name,pid=child.pid)
                while True:
                    try:code=child.wait(timeout=600);break
                    except subprocess.TimeoutExpired:pass
            if code:
                emit('ERROR',job=name,exit_code=code,note='保留結果並停止後續；需下次模型介入診斷');return code
            emit('JOB_DONE',job=name)
        emit('ALL_DONE',note='KD及結果整理完成，尚未執行共享層修正或独立匯出重驗')
    return 0

if __name__=='__main__':raise SystemExit(main())
