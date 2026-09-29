"""同一 QK E5 起點，比較 narrow 與後段解凍；先實際更新校準，再至 E8。"""
import json
import subprocess
import sys
from common import ROOT,write_json,sha256
import torch
import qk_challenger


def wait_job(command):
    child=subprocess.Popen(command)
    while True:
        try:return child.wait(timeout=600)
        except subprocess.TimeoutExpired:pass


def main():
    state_path=ROOT/'artifacts/a0-scope-pair-v1-state.json'
    if state_path.exists():raise FileExistsError('禁止覆寫既有 queue')
    snapshot=ROOT/'artifacts/a0-qk-recovery-v3/epoch-05-resume.pt'
    state={'status':'smoke','jobs':[],'monitor_wait_seconds':600,'accuracy_winner':None,
        'parent':str(snapshot),'parent_sha256':sha256(snapshot),'planned_endpoint_epoch':8,
        'decision_metrics':['coco/box/map50_95','coco/person/box/map50_95'],
        'variable':'trainable scope, with new-group optimizer state and one-epoch LR ramp',
        'inference_graph_changed':False,'ball_bat_gate':False}
    write_json(state_path,state)
    proof={}
    for smoke in (True,False):
        for scope in ('narrow','late'):
            name=f'a0-scope-{scope}-'+('smoke-v1' if smoke else 'v1')
            job={'name':name,'scope':scope,'status':'running','smoke':smoke}
            state['jobs'].append(job);write_json(state_path,state)
            command=[sys.executable,str(ROOT/'scripts/monitor.py'),'--name',name,'--',sys.executable,'-u',
                str(ROOT/'scripts/continue_a0.py'),'--arm','qk','--name',name,'--scope',scope,
                '--resume-from',str(snapshot),'--max-epochs','8','--patience','6']
            if smoke:command.append('--smoke')
            code=wait_job(command)
            if code:
                state['status']='error';job['status']='error';write_json(state_path,state);return code
            output=ROOT/'artifacts'/name
            summary=json.loads((output/'summary.json').read_text())
            job['status']=summary['status'];write_json(state_path,state)
            if smoke:
                assert summary['status']=='passed' and summary['smoke_images']==128
                assert summary['initial_ema_updates']==4625 and summary['optimizer_steps']==4626
                payload=torch.load(output/'epoch-06-resume.pt',map_location='cpu',weights_only=False)
                assert payload['ema_updates']==4626 and payload['model'].criterion.updates==5
                for group in payload['optimizer']['param_groups']:
                    expected=1 if group['role']=='adapt' else 4626
                    steps={int(payload['optimizer']['state'][i]['step']) for i in group['params']}
                    assert steps=={expected},(group['role'],steps)
                if scope=='late':
                    assert summary['adaptive_parameters']>0
                    assert all(0<x<0.001 for x in summary['first_adaptive_update_ratios'].values())
                else:assert summary['adaptive_parameters']==0
                first=json.loads((output/'progress.jsonl').read_text().splitlines()[0])
                proof[scope]={'trace':summary['first_macro_trace_sha256'],'loss_sum':first['loss_sum'],
                    'ema_updates':4626,'criterion_updates':5,'adaptive_parameters':summary['adaptive_parameters'],
                    'update_ratios':summary.get('first_adaptive_update_ratios',{}),
                    'parent_sha256':summary['continued_from_sha256']}
                del payload
            else:
                assert summary['status'] in ('complete','patience_stop','paused_for_analysis')
                job['last_epoch']=summary['epochs'][-1]['epoch'];write_json(state_path,state)
                if summary['status']=='paused_for_analysis':
                    state['status']='awaiting_analysis';write_json(state_path,state)
                    print('STAGE_REVIEW: overall/person 安全門檻觸發',flush=True);return 0
        if smoke:
            assert proof['narrow']['trace']==proof['late']['trace']
            assert proof['narrow']['parent_sha256']==proof['late']['parent_sha256']==state['parent_sha256']
            assert abs(proof['narrow']['loss_sum']-proof['late']['loss_sum'])<1e-6*abs(proof['narrow']['loss_sum'])
            write_json(ROOT/'artifacts/scope-proof-v1.json',{'status':'passed','arms':proof,
                'exact_uninterrupted_dataloader_replay':False,'paired_new_stream':True})
            state['status']='training';write_json(state_path,state)
            print('SCOPE_PASSED: 同權重／資料／首更新前 loss、optimizer／EMA 與解凍更新比例',flush=True)
    state['status']='awaiting_analysis';write_json(state_path,state)
    print('ALL_DONE: E8 narrow／late 完成，接續 overall/person 與推論分析',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
