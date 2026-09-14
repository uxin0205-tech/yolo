"""MASF 報告完成後啟動；避免以 queue.py 名稱遮蔽 Python 標準函式庫。"""
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
ART=HERE/'artifacts/queue-v1'

def users():
    r=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    return [int(p.strip()) for p in r.stdout.splitlines() if p.strip()]

def sleep600():
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:time.sleep(max(0,min(60,deadline-time.monotonic())))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:print('READY：scale_bias → Rep17 → Rep20，皆 smoke→5E→驗證分析；原生QK重用既有結果');return 0
    assert json.loads((HERE.parent/'pose_masf_training_v1/artifacts/b-e5-seed1-v1/final-audit.json').read_text())['status']=='completed'
    assert (HERE.parent/'pose_masf_training_v1/RESULTS.md').exists() and (HERE/'README.md').exists()
    assert json.loads((HERE/'artifacts/preflight-v1.json').read_text())['status']=='passed'
    plan=json.loads((HERE/'queue-plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    ART.mkdir(parents=True,exist_ok=True)
    lock=(ART/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shared=(HERE.parent/'attention_recovery_v1/artifacts/queue-v1/worker.lock').open('a')
    def emit(status,**kwargs):
        row={'status':status,'time':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**kwargs}
        text=json.dumps(row,ensure_ascii=False)
        with (ART/'events.jsonl').open('a') as f:f.write(text+'\n')
        tmp=ART/'state.tmp';tmp.write_text(text+'\n');tmp.replace(ART/'state.json');print(text,flush=True)
    waiting=False
    while True:
        try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);ok=True
        except BlockingIOError:ok=False
        occupied=users()
        if ok and not occupied:break
        if ok:fcntl.flock(shared,fcntl.LOCK_UN)
        if not waiting:emit('WAITING_GPU',external_pids=occupied);waiting=True
        sleep600()
    for arm in plan['arms']:
        for phase in ('smoke','train'):
            result=HERE/'artifacts'/arm/phase/'summary.json'
            if result.exists():assert json.loads(result.read_text())['status']=='completed';continue
            if users():
                emit('WAITING_GPU',arm=arm,phase=phase)
                while users():sleep600()
            attempt=1
            while (ART/f'{arm}-{phase}-{attempt}.log').exists():attempt+=1
            log=ART/f'{arm}-{phase}-{attempt}.log'
            with log.open('x') as f:
                child=subprocess.Popen([sys.executable,'-B',str(HERE/'run_arm.py'),arm,phase],stdout=f,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'0'})
                emit('JOB_STARTED',arm=arm,phase=phase,child_pid=child.pid,log=str(log))
                checkpoint=time.monotonic();stalled=False
                while True:
                    try:code=child.wait(timeout=60);break
                    except subprocess.TimeoutExpired:pass
                    if time.monotonic()-checkpoint>=600:
                        checkpoint=time.monotonic();heartbeat=HERE/'artifacts'/arm/phase/'heartbeat'
                        latest=max(p.stat().st_mtime for p in (heartbeat,log) if p.exists())
                        if time.time()-latest>1800 and not stalled:emit('STALLED',arm=arm,phase=phase,child_pid=child.pid);stalled=True
                        elif stalled and time.time()-latest<=1800:emit('RECOVERED',arm=arm,phase=phase);stalled=False
            if code or not result.exists():emit('ERROR',arm=arm,phase=phase,exit_code=code,log=str(log));return code or 1
            assert json.loads(result.read_text())['status']=='completed'
            emit('JOB_DONE',arm=arm,phase=phase,result=str(result))
    sys.path.insert(0,str(HERE.parents[1]))
    from tools.restructure_layout import edit
    base=json.loads((HERE.parent/'pose_masf_training_v1/artifacts/b-e5-seed1-v1/analysis/summary.json').read_text())['e5']
    summaries={a:json.loads((HERE/'artifacts'/a/'train/summary.json').read_text()) for a in plan['arms']}
    lines=['# MASF 之後：三候選完整比較', '',
           '所有已排工作完成；不自動升版、不再加訓。沒有同預算未改架構對照，差值不代表方法的獨立因果收益。', '',
           '| AP50–95 (%) | 原生 QK B5 | scale/bias E5 | Rep17 E5 | Rep20 E5 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for k,v in base.items():
        if k.endswith('/map50_95'):
            values=[v]+[summaries[a]['metrics_by_epoch']['5'][k] for a in plan['arms']]
            lines.append('| '+k+' | '+' | '.join(f'{100*x:.4f}' for x in values)+' |')
    lines+=['','方法、硬體邊界及layer17/20理由見 README.md；每組的逐回合、部署常數與Float/BitTrue驗證見其train目錄。']
    for a,summary in summaries.items():
        lines.append(f"- {a} 工程門檻：{'通過（僅候選）' if summary['gate_passed'] else '未通過'}；E5匯出SHA：{summary['inference_sha256']}")
    edit(HERE/'RESULTS.md','\n'.join(lines))
    emit('ALL_DONE',reports=[str(HERE/'artifacts'/a/'train/RESULTS.md') for a in plan['arms']],default_model_unchanged=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
