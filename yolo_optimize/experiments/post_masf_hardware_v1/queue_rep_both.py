"""既有三組全完成後接雙層組；不重啟／終止既有 queue 或 GPU job。"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ART=HERE/'artifacts/queue-rep-both-v1'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:print('READY：原三組完成後 → Rep17+20 smoke → 5E → 分析');return 0
    plan=json.loads((HERE/'queue-rep-both-plan.json').read_text())
    assert json.loads((HERE/'artifacts/rep-both-preflight-v1.json').read_text())['status']=='passed'
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    ART.mkdir(parents=True,exist_ok=True)
    own=(ART/'worker.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def emit(status,**kwargs):
        row={'status':status,'time':datetime.now(timezone.utc).isoformat(),'queue_pid':os.getpid(),**kwargs};text=json.dumps(row,ensure_ascii=False)
        with (ART/'events.jsonl').open('a') as f:f.write(text+'\n')
        tmp=ART/'state.tmp';tmp.write_text(text+'\n');tmp.replace(ART/'state.json');print(text,flush=True)
    predecessor=plan['predecessor_pid'];stat=Path(f'/proc/{predecessor}/stat')
    if stat.exists():
        ticks=stat.read_text().split(') ',1)[1].split()[19]
        assert ticks==plan['predecessor_start_ticks'],'原 PID 已換成其他程序，禁止等待錯誤工作'
        try:fd=os.pidfd_open(predecessor)
        except ProcessLookupError:fd=None
        if fd is not None:
            emit('WAITING_PREDECESSOR',predecessor_pid=predecessor)
            try:
                while not select.select([fd],[],[],60)[0]:pass
            finally:os.close(fd)
    for arm in ('scale_bias','rep17','rep20'):
        result=HERE/'artifacts'/arm/'train/summary.json'
        if not result.exists() or json.loads(result.read_text())['status']!='completed':
            emit('ERROR',reason='PREDECESSOR_NOT_COMPLETED',arm=arm);return 1
    from run_queue import users,sleep600
    shared=(HERE.parent/'attention_recovery_v1/artifacts/queue-v1/worker.lock').open('a')
    waiting=False
    while True:
        try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
        except BlockingIOError:locked=False
        occupied=users()
        if locked and not occupied:break
        if locked:fcntl.flock(shared,fcntl.LOCK_UN)
        if not waiting:emit('WAITING_GPU',external_pids=occupied);waiting=True
        sleep600()
    for phase in ('smoke','train'):
        result=HERE/'artifacts/rep17_20'/phase/'summary.json'
        if result.exists():assert json.loads(result.read_text())['status']=='completed';continue
        if users():
            emit('WAITING_GPU',phase=phase)
            while users():sleep600()
        attempt=1
        while (ART/f'{phase}-{attempt}.log').exists():attempt+=1
        log=ART/f'{phase}-{attempt}.log'
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'run_rep_both.py'),'rep17_20',phase],stdout=f,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'0'})
            emit('JOB_STARTED',arm='rep17_20',phase=phase,child_pid=child.pid,log=str(log))
            last=time.monotonic();stalled=False
            while True:
                try:code=child.wait(timeout=60);break
                except subprocess.TimeoutExpired:pass
                if time.monotonic()-last>=600:
                    last=time.monotonic();heart=HERE/'artifacts/rep17_20'/phase/'heartbeat'
                    changed=max(p.stat().st_mtime for p in (log,heart) if p.exists())
                    if time.time()-changed>1800 and not stalled:emit('STALLED',phase=phase,child_pid=child.pid);stalled=True
                    elif stalled and time.time()-changed<=1800:emit('RECOVERED',phase=phase);stalled=False
        if code or not result.exists():emit('ERROR',phase=phase,exit_code=code,log=str(log));return code or 1
        assert json.loads(result.read_text())['status']=='completed'
        emit('JOB_DONE',phase=phase,result=str(result))
    sys.path.insert(0,str(HERE.parents[1]))
    from tools.restructure_layout import edit
    arms=('scale_bias','rep17','rep20','rep17_20')
    base=json.loads((HERE.parent/'pose_masf_training_v1/artifacts/b-e5-seed1-v1/analysis/summary.json').read_text())['e5']
    rows={a:json.loads((HERE/'artifacts'/a/'train/summary.json').read_text()) for a in arms}
    lines=['# 四候選完整比較（含 Rep17＋20）','',
        '原三候選結果不覆寫；此表加入使用者追加的雙層組。均從同一 B5 起點，固定5epoch／warmup1，無未改架構加訓對照。','',
        '| AP50–95 (%) | 原生 QK B5 | scale/bias | Rep17 | Rep20 | Rep17＋20 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,value in base.items():
        if key.endswith('/map50_95'):
            values=[value]+[rows[a]['metrics_by_epoch']['5'][key] for a in arms]
            lines.append('| '+key+' | '+' | '.join(f'{100*v:.4f}' for v in values)+' |')
    lines+=['','沒有自動採用或加訓；layer17／20理由、負面結果與硬體限制見README，各組完整state與獨立驗證在artifacts。']
    edit(HERE/'RESULTS-4arms.md','\n'.join(lines))
    emit('ALL_DONE',report=str(HERE/'RESULTS-4arms.md'),default_model_unchanged=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
