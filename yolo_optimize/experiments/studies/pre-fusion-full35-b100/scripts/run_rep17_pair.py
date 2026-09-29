"""前置整圖／COCO 驗證後，真實 macro 校準，再串行兩組各5回合。"""
import json
import sys
from common import ROOT,write_json
from run_scope_pair import wait_job
from export_model import export
import torch


def main():
    state_path=ROOT/'artifacts/rep17-pair-v1-state.json'
    if state_path.exists():raise FileExistsError('禁止覆寫既有 queue')
    state={'status':'preflight','jobs':[],'monitor_wait_seconds':600,'epochs_per_arm':5,
        'decision_metrics':['coco/box/map50_95','coco/person/box/map50_95'],'accuracy_winner':None}
    write_json(state_path,state)
    code=wait_job([sys.executable,str(ROOT/'scripts/monitor.py'),'--name','rep17-preflight-v1','--',
        sys.executable,'-u',str(ROOT/'scripts/probe_rep17.py')])
    if code:state['status']='error';write_json(state_path,state);return code
    assert json.loads((ROOT/'artifacts/rep17-preflight-v1/summary.json').read_text())['status']=='passed'
    proof={};torch.set_num_threads(4)
    for smoke in (True,False):
        for variant in ('control','rep'):
            name=f'rep17-{variant}-'+('smoke-v1' if smoke else 'v1')
            job={'name':name,'variant':variant,'status':'running','smoke':smoke}
            state['jobs'].append(job);write_json(state_path,state)
            command=[sys.executable,str(ROOT/'scripts/monitor.py'),'--name',name,'--',sys.executable,'-u',
                str(ROOT/'scripts/train_rep17.py'),'--variant',variant,'--name',name]
            if smoke:command.append('--smoke')
            code=wait_job(command)
            if code:state['status']='error';job['status']='error';write_json(state_path,state);return code
            output=ROOT/'artifacts'/name;summary=json.loads((output/'summary.json').read_text())
            job['status']=summary['status'];write_json(state_path,state)
            if smoke:
                assert summary['status']=='passed' and summary['smoke_images']==128
                assert summary['optimizer_steps']==1 and summary['ema_updates']==7401
                assert 0<summary['first_update_ratio']<.001
                exported=export(output/'epoch-01-resume.pt',output/'smoke-inference.pt')
                proof[variant]={'trace':summary['first_macro_trace_sha256'],
                    'loss_sum':summary['first_macro_native_loss_sum'],'first_update_ratio':summary['first_update_ratio'],
                    'ema_updates':7401,'exported':str(exported),'export_cpu160_reload_exact':True}
            else:
                assert summary['status'] in ('complete','paused_for_analysis')
                job['last_epoch']=summary['epochs'][-1]['epoch'];write_json(state_path,state)
                if summary['status']=='paused_for_analysis':
                    state['status']='awaiting_analysis';write_json(state_path,state)
                    print('STAGE_REVIEW: Rep17 主要 AP 安全門檻觸發',flush=True);return 0
        if smoke:
            assert proof['control']['trace']==proof['rep']['trace']
            assert proof['control']['loss_sum']==proof['rep']['loss_sum']
            write_json(ROOT/'artifacts/rep17-smoke-proof-v1.json',{'status':'passed','arms':proof})
            state['status']='training';write_json(state_path,state)
            print('REP17_PASSED: 同資料／初始loss、單點更新、固定live/EMA與匯出',flush=True)
    state['status']='awaiting_analysis';write_json(state_path,state)
    print('ALL_DONE: 單點 RepConv 兩組各5回合完成，接續主要AP與MASF分析',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
