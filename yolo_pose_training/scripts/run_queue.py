"""背景循序執行與每 600 秒記錄 GPU；失敗立即停止相依階段。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT/'artifacts/queue-status.json'

def status(state, **kwargs):
    payload = {'state':state, 'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'queue_pid':os.getpid(), **kwargs}
    tmp = STATUS.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2)); tmp.replace(STATUS)
    print(json.dumps(payload),flush=True)


def monitor(stage, proc, logfile):
    try:
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
        row={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'stage':stage,'pid':proc.pid,'gpu':gpu.stdout.strip(),
             'log_age_seconds':round(time.time()-logfile.stat().st_mtime,1)}
        row['warning'] = '日誌超過 30 分鐘未更新，需檢查' if row['log_age_seconds']>1800 else None
        with (ROOT/'logs/monitor.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    except Exception as e:
        print(f'監測異常：{e}',flush=True)


def execute(name, command, interval, allow_oom=False):
    logfile=ROOT/'logs'/f'{name}.log'
    with logfile.open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        status('running',stage=name,child_pid=proc.pid,log=str(logfile))
        deadline=0
        while proc.poll() is None:
            if time.monotonic()>=deadline:
                monitor(name,proc,logfile); deadline=time.monotonic()+interval
            time.sleep(5)
        if allow_oom and proc.returncode == 42:
            return False
        if proc.returncode:
            raise RuntimeError(f'{name} 結束碼 {proc.returncode}，請查閱 {logfile}')
        return True


def main():
    os.chdir(ROOT)
    lock=(ROOT/'artifacts/queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=yaml.safe_load((ROOT/'configs/queue.yaml').read_text())
    for config in queue['stages']:
        cfg=yaml.safe_load((ROOT/config).read_text())
        done=ROOT/'runs'/cfg['name']/'completed.json'
        if done.is_file():
            continue
        run = ROOT/'runs'/cfg['name']
        checkpoint = run/'weights/last-resumable.pt'
        if run.exists() and not checkpoint.is_file():
            raise RuntimeError(f"已有未完成 run 且無完整 checkpoint：{cfg['name']}，停止避免覆寫")
        assert Path(cfg['model']).is_file(), cfg['model']
        batch = cfg['batch']
        print(f"{cfg['name']} 固定 batch={batch}", flush=True)
        command=[sys.executable,'-u',str(ROOT/'scripts/train_stage.py'),str(ROOT/config),str(batch)]
        if not execute(cfg['name'],command,queue['monitor_seconds'],allow_oom=True):
            raise RuntimeError(f"{cfg['name']} batch={batch} 顯存不足，保留 checkpoint 並停止；不自動變更 batch")
    execute('comparison',[sys.executable,'-u',str(ROOT/'scripts/evaluate.py')],queue['monitor_seconds'])
    status('completed',report=str(ROOT/'artifacts/final/report.md'))

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=str(exc)); traceback.print_exc(); sys.exit(1)
