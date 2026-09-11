"""持久化單一工作的開始／結束事件；正常狀態只做最多 600 秒 wait。"""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from common import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum():
        raise ValueError('job name 格式不符')
    command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:raise ValueError('缺少工作指令')
    logs=ROOT/'artifacts/logs';logs.mkdir(parents=True,exist_ok=True)
    with (logs/f'{args.name}.events.jsonl').open('x') as events, (logs/f'{args.name}.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
        def emit(kind,**extra):
            payload={'event':kind,'time':datetime.now(timezone.utc).isoformat(),
                'pid':child.pid,'supervisor_pid':os.getpid(),**extra}
            events.write(json.dumps(payload,ensure_ascii=False)+'\n');events.flush()
            print(json.dumps(payload,ensure_ascii=False),flush=True)
        emit('JOB_STARTED',command=command)
        while True:
            try:code=child.wait(timeout=600);break
            except subprocess.TimeoutExpired:pass
        emit('JOB_DONE' if code==0 else 'ERROR',exit_code=code)
        return code


if __name__=='__main__':raise SystemExit(main())
