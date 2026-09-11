"""完整 E3 成對邊界驗證通過後，接續至 E5；正常只以 600 秒 wait 監測。"""
import json
import subprocess
import sys
from common import ROOT,write_json
import torch
import qk_challenger


def run(command):
    child=subprocess.Popen(command)
    while True:
        try:return child.wait(timeout=600)
        except subprocess.TimeoutExpired:pass


def main():
    state_path=ROOT/'artifacts/a0-pair-v3-state.json'
    if state_path.exists():raise FileExistsError('禁止覆寫既有 queue')
    state={'status':'smoke','jobs':[],'monitor_wait_seconds':600,
        'planned_endpoint_epoch':5,'accuracy_winner':None,'ball_bat_gate':False,
        'decision_metrics':['coco/box/map50_95','coco/person/box/map50_95']}
    write_json(state_path,state)
    reports={}
    for smoke in (True,False):
        for arm in ('control','qk'):
            name=f'a0-{arm}-'+('e3-smoke-v3' if smoke else 'recovery-v3')
            snapshot=ROOT/f'artifacts/a0-{arm}-recovery-v2/epoch-03-resume.pt'
            job={'name':name,'arm':arm,'status':'running','smoke':smoke}
            state['jobs'].append(job);write_json(state_path,state)
            command=[sys.executable,str(ROOT/'scripts/monitor.py'),'--name',name,'--',
                sys.executable,'-u',str(ROOT/'scripts/continue_a0.py'),'--arm',arm,
                '--name',name,'--resume-from',str(snapshot),'--max-epochs','5','--patience','6']
            if smoke:command.append('--smoke')
            code=run(command)
            if code:
                job['status']='error';state['status']='error';write_json(state_path,state);return code
            output=ROOT/'artifacts'/name
            summary=json.loads((output/'summary.json').read_text())
            job['status']=summary['status'];write_json(state_path,state)
            if smoke:
                assert summary['status']=='passed'
                payload=torch.load(output/'epoch-04-resume.pt',map_location='cpu',weights_only=False)
                steps={int(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s}
                assert steps=={2776} and payload['ema_updates']==2776
                assert summary['initial_ema_updates']==2775 and summary['optimizer_steps']==2776
                assert payload['model'].criterion.updates==3 and summary['smoke_images']==128
                assert summary['epochs'][0]['source_run']==f'a0-{arm}-recovery-v1'
                assert summary['epochs'][2]['source_run']==f'a0-{arm}-recovery-v2'
                reports[arm]={'optimizer_steps':sorted(steps),'ema_updates':2776,'criterion_updates':3,
                    'trace':summary['first_macro_trace_sha256'],'initial_stale_epochs':summary['initial_stale_epochs']}
                del payload
            else:
                assert summary['status'] in ('complete','patience_stop','paused_for_analysis')
                job['last_epoch']=summary['epochs'][-1]['epoch'];write_json(state_path,state)
                if summary['status']=='paused_for_analysis':
                    state['status']='awaiting_analysis';write_json(state_path,state)
                    print('STAGE_REVIEW: overall/person 安全門檻觸發',flush=True);return 0
        if smoke:
            assert reports['control']['trace']==reports['qk']['trace']
            old=json.loads((ROOT/'artifacts/continuation-proof-v2.json').read_text())
            assert reports['control']['trace']!=old['arms']['control']['trace']
            write_json(ROOT/'artifacts/continuation-proof-v3.json',{'status':'passed','arms':reports,
                'exact_uninterrupted_dataloader_replay':False,'paired_new_stream':True})
            state['status']='training';write_json(state_path,state)
            print('CONTINUATION_PASSED: E3 optimizer／EMA／criterion／paired trace',flush=True)
    state['status']='awaiting_analysis';write_json(state_path,state)
    print('ALL_DONE: E5 成對階段完成；尚未宣稱增準，接續分析訓練範圍',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
