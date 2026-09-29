"""以 overall/person 為唯一決策項，兩臂各從完整 E1 接續至 E3。"""
import json
import subprocess
import sys
from common import ROOT,write_json


def main():
    state_path=ROOT/'artifacts/a0-pair-v2-state.json'
    if state_path.exists():raise FileExistsError('禁止覆寫既有 queue')
    a=json.loads((ROOT/'artifacts/a0-control-continue-smoke-v2/summary.json').read_text())
    b=json.loads((ROOT/'artifacts/a0-qk-continue-smoke-v2/summary.json').read_text())
    assert a['first_macro_trace_sha256']==b['first_macro_trace_sha256']
    assert a['optimizer_steps']==b['optimizer_steps']==926
    assert a['initial_ema_updates']==b['initial_ema_updates']==925
    state={'status':'running','jobs':[],'monitor_wait_seconds':600,'accuracy_winner':None,
        'decision_metrics':['coco/box/map50_95','coco/person/box/map50_95'],
        'planned_endpoint_epoch':3,'ball_bat_gate':False}
    write_json(state_path,state)
    for arm in ('control','qk'):
        name=f'a0-{arm}-recovery-v2'
        job={'name':name,'arm':arm,'status':'running'};state['jobs'].append(job);write_json(state_path,state)
        snapshot=ROOT/f'artifacts/a0-{arm}-recovery-v1/epoch-01-resume.pt'
        command=[sys.executable,str(ROOT/'scripts/monitor.py'),'--name',name,'--',sys.executable,'-u',
            str(ROOT/'scripts/continue_a0.py'),'--arm',arm,'--name',name,'--resume-from',str(snapshot),
            '--max-epochs','3','--patience','6']
        process=subprocess.Popen(command)
        while True:
            try:code=process.wait(timeout=600);break
            except subprocess.TimeoutExpired:pass
        if code:
            job['status']='error';state['status']='error';write_json(state_path,state);return code
        summary=json.loads((ROOT/'artifacts'/name/'summary.json').read_text())
        if summary['status'] not in ('complete','patience_stop','paused_for_analysis'):raise RuntimeError('未知終態')
        job.update(status=summary['status'],last_epoch=summary['epochs'][-1]['epoch'])
        write_json(state_path,state)
    state['status']='awaiting_analysis';write_json(state_path,state)
    print('ALL_DONE: E3 階段完成，檢查 overall/person 趨勢後決定延長，不代表已增準',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
