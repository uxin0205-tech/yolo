#!/usr/bin/env python3
"""單一 recovery job 的安靜監督器：正常只等待，最多 600 秒一個 wait。

不讀 log、不查 GPU、不推定 STALLED；退出時才讀 summary 分類。
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('smoke','train','masf-diagnostic','qk-fp10','qk-fp22'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('rep17','masf_off','heads'),default='rep17')
    args=parser.parse_args()
    root=args.output.resolve()
    if not root.is_relative_to(ROOT) or root==ROOT or root.exists():
        raise ValueError('輸出必須為全新 workspace 子目錄')
    logs=root.parent/'logs'; logs.mkdir(parents=True,exist_ok=True)
    child_log=logs/f'{root.name}.log'
    events=logs/f'{root.name}.events.jsonl'
    if events.exists() or child_log.exists(): raise FileExistsError('禁止覆寫既有 log')
    command=([sys.executable,'-u',str(ROOT/'scripts/verify_recovery_safety.py'),
             '--variant',args.variant,'--output',str(root)] if args.stage=='smoke' else
             [sys.executable,'-u',str(ROOT/'scripts/run_recovery.py'),'train',
              '--variant',args.variant,'--physical-batch','32','--ema-age-mode','parent',
              '--validate-live','--output',str(root)])
    if args.stage == 'masf-diagnostic':
        command=[sys.executable,'-u',str(ROOT/'scripts/run_recovery.py'),'validate',
                 '--candidate','masf_off','--backend','both','--output',str(root)]
    if args.stage in ('qk-fp10','qk-fp22'):
        command=[sys.executable,'-u',str(ROOT/'scripts/run_recovery.py'),'validate',
                 '--candidate',args.stage.replace('-','_'),'--backend','bittrue','--output',str(root)]
    with events.open('x') as event_file,child_log.open('x') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
        def emit(event,**kwargs):
            value={'event':event,'time':datetime.now(timezone.utc).isoformat(),
                   'pid':child.pid,'supervisor_pid':os.getpid(),'output':str(root),**kwargs}
            event_file.write(json.dumps(value,ensure_ascii=False)+'\n'); event_file.flush()
            print(json.dumps(value,ensure_ascii=False),flush=True)
        emit('JOB_STARTED',command=command)
        while True:
            try:
                code=child.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                pass
        try:
            summary=json.loads((root/'summary.json').read_text())
            status=summary.get('status')
            if args.stage in ('masf-diagnostic','qk-fp10','qk-fp22'):
                status=json.loads((root/'status.json').read_text()).get('phase')
            ok=code==0 and status in ('passed','complete','paused_for_analysis')
            emit('JOB_DONE' if ok else 'ERROR',exit_code=code,status=status)
        except (OSError,ValueError) as error:
            emit('ERROR',exit_code=code,error=str(error)); ok=False
        return 0 if ok else 1

if __name__=='__main__': raise SystemExit(main())
