#!/usr/bin/env python3
"""啟動獨立 session 的 quiet supervisor，自己只以 pidfd 等待其退出。"""
import argparse
import os
from pathlib import Path
import select
import subprocess
import sys
import json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','masf-diagnostic','qk-fp10','qk-fp22'),default='train')
    parser.add_argument('--variant',choices=('rep17','masf_off','heads'),default='rep17')
    args=parser.parse_args()
    root=args.output.resolve()
    if not root.is_relative_to(ROOT) or root==ROOT or root.exists():
        raise ValueError('輸出必須為全新 workspace 子目錄')
    logs=root.parent/'logs'; logs.mkdir(parents=True,exist_ok=True)
    with (logs/f'{root.name}.supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/quiet_recovery_supervisor.py'),
             '--stage',args.stage,'--variant',args.variant,'--output',str(root)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
             env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
    descriptor=os.pidfd_open(child.pid)
    print(json.dumps({'event':'MONITOR_STARTED','supervisor_pid':child.pid,'output':str(root)}),flush=True)
    try:
        while not select.select([descriptor],[],[],600)[0]: pass
        code=child.wait()
        print(json.dumps({'event':'SUPERVISOR_DONE' if code==0 else 'ERROR',
                          'exit_code':code,'output':str(root)}),flush=True)
        return code
    finally: os.close(descriptor)

if __name__=='__main__': raise SystemExit(main())
