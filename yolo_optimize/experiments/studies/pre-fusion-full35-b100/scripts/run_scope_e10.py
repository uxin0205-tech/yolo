"""narrow／late 各自 E8 接續至 E10，保留 late 已訓練三回合的 moments。"""
import json
import subprocess
import sys
from common import ROOT,write_json,sha256
from run_scope_pair import wait_job
import torch
import qk_challenger


def main():
    state_path=ROOT/'artifacts/a0-scope-pair-v2-state.json'
    if state_path.exists():raise FileExistsError('禁止覆寫既有 queue')
    proof={}
    state={'status':'smoke','jobs':[],'monitor_wait_seconds':600,'planned_endpoint_epoch':10,
        'decision_metrics':['coco/box/map50_95','coco/person/box/map50_95'],'accuracy_winner':None,
        'ball_bat_gate':False,'script_sha256':sha256(ROOT/'scripts/continue_a0.py')}
    write_json(state_path,state)
    for smoke in (True,False):
        for scope in ('narrow','late'):
            name=f'a0-scope-{scope}-'+('e8-smoke-v2' if smoke else 'v2')
            snapshot=ROOT/f'artifacts/a0-scope-{scope}-v1/epoch-08-resume.pt'
            job={'name':name,'scope':scope,'status':'running','smoke':smoke,'snapshot':str(snapshot)}
            state['jobs'].append(job);write_json(state_path,state)
            command=[sys.executable,str(ROOT/'scripts/monitor.py'),'--name',name,'--',sys.executable,'-u',
                str(ROOT/'scripts/continue_a0.py'),'--arm','qk','--name',name,'--scope',scope,
                '--resume-from',str(snapshot),'--max-epochs','10','--patience','6']
            if smoke:command.append('--smoke')
            code=wait_job(command)
            if code:
                state['status']='error';job['status']='error';write_json(state_path,state);return code
            output=ROOT/'artifacts'/name
            summary=json.loads((output/'summary.json').read_text())
            job['status']=summary['status'];write_json(state_path,state)
            if smoke:
                assert summary['status']=='passed' and summary['smoke_images']==128
                assert summary['initial_ema_updates']==7400 and summary['optimizer_steps']==7401
                payload=torch.load(output/'epoch-09-resume.pt',map_location='cpu',weights_only=False)
                assert payload['ema_updates']==7401 and payload['model'].criterion.updates==8
                steps_by_role={}
                for group in payload['optimizer']['param_groups']:
                    expected=2776 if group['role']=='adapt' else 7401
                    steps={int(payload['optimizer']['state'][i]['step']) for i in group['params']}
                    assert steps=={expected},(group['role'],steps)
                    steps_by_role[group['role']]=sorted(steps)
                assert summary['new_group_moments_restored']==(scope=='late')
                proof[scope]={'trace':summary['first_macro_trace_sha256'],'ema_updates':7401,
                    'criterion_updates':8,'optimizer_steps':steps_by_role,
                    'adaptive_start_epoch':summary['adaptive_start_epoch'],
                    'new_group_moments_restored':summary['new_group_moments_restored']}
                del payload
            else:
                assert summary['status'] in ('complete','patience_stop','paused_for_analysis')
                job['last_epoch']=summary['epochs'][-1]['epoch'];write_json(state_path,state)
                if summary['status']=='paused_for_analysis':
                    state['status']='awaiting_analysis';write_json(state_path,state)
                    print('STAGE_REVIEW: overall/person 安全門檻觸發',flush=True);return 0
        if smoke:
            assert proof['narrow']['trace']==proof['late']['trace']
            write_json(ROOT/'artifacts/scope-continuation-proof-v2.json',{'status':'passed','arms':proof,
                'paired_new_stream':True,'exact_uninterrupted_dataloader_replay':False})
            state['status']='training';write_json(state_path,state)
            print('CONTINUATION_PASSED: E8 原群與 late 群 moments／EMA／criterion／trace',flush=True)
    state['status']='awaiting_analysis';write_json(state_path,state)
    print('ALL_DONE: E10 對照完成，接續主要 AP／推論分析',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
