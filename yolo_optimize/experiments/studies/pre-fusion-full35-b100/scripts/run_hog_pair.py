"""融合前 HOG：實際更新／strip 驗證後，原生5與HOG最多10（patience4）。"""
import json
import sys
from common import ROOT,write_json,sha256
from run_scope_pair import wait_job
from train_hog import PARENT,inference_graph
from verify_qk_challenger import tensors
import torch


def main():
    state_path=ROOT/'artifacts/prefusion-hog-pair-v1-state.json'
    if state_path.exists():raise FileExistsError('禁止覆寫既有 queue')
    calibration=json.loads((ROOT/'artifacts/prefusion-hog-calibration-v1/summary.json').read_text())
    assert calibration['status']=='passed' and calibration['parent_sha256']==sha256(PARENT)
    state={'status':'smoke','jobs':[],'monitor_wait_seconds':600,'accuracy_winner':None,
        'decision_metrics':['coco/box/map50_95','coco/person/box/map50_95'],
        'control_epochs':5,'hog_max_epochs':10,'hog_patience':4,'parent':str(PARENT)}
    write_json(state_path,state);proof={};torch.set_num_threads(4)
    for smoke in (True,False):
        for arm in ('control','hog'):
            name=f'prefusion-hog-{arm}-'+('smoke-v1' if smoke else 'v1')
            job={'name':name,'arm':arm,'status':'running','smoke':smoke}
            state['jobs'].append(job);write_json(state_path,state)
            command=[sys.executable,str(ROOT/'scripts/monitor.py'),'--name',name,'--',sys.executable,'-u',
                str(ROOT/'scripts/train_hog.py'),'--arm',arm,'--name',name]
            if smoke:command.append('--smoke')
            code=wait_job(command)
            if code:
                state['status']='error';job['status']='error';write_json(state_path,state);return code
            output=ROOT/'artifacts'/name;summary=json.loads((output/'summary.json').read_text())
            job['status']=summary['status'];write_json(state_path,state)
            if smoke:
                assert summary['status']=='passed' and summary['smoke_images']==128
                assert summary['optimizer_steps']==1 and summary['ema_updates']==7401
                payload=torch.load(output/'epoch-01-resume.pt',map_location='cpu',weights_only=False)
                for group in payload['optimizer']['param_groups']:
                    steps={int(payload['optimizer']['state'][i]['step']) for i in group['params']
                        if i in payload['optimizer']['state']}
                    assert steps==(set() if arm=='control' and group['role']=='hog' else {1})
                original=payload['model'].float().eval();stripped=inference_graph(original).float().eval()
                difference=set(original.state_dict())-set(stripped.state_dict())
                assert difference=={'hog_aux.projection.weight','hog_aux.projection.bias'}
                image=torch.rand(1,3,160,160,generator=torch.Generator().manual_seed(0))
                with torch.inference_mode():
                    expected=tensors(original(image));actual=tensors(stripped(image))
                assert len(expected)==len(actual) and all(torch.equal(x,y) for x,y in zip(expected,actual))
                proof[arm]={'trace':summary['first_macro_trace_sha256'],'strip_cpu160_exact':True,
                    'stripped_keys':sorted(difference),'p3_grad_norm':summary['p3_producer_grad_norm'],
                    'hog_grad_norm':summary['hog_grad_norm'],'optimizer_steps':1,'ema_updates':7401}
                del payload,original,stripped
            else:
                assert summary['status'] in ('complete','patience_stop','paused_for_analysis')
                job['last_epoch']=summary['epochs'][-1]['epoch'];write_json(state_path,state)
                if summary['status']=='paused_for_analysis':
                    state['status']='awaiting_analysis';write_json(state_path,state)
                    print('STAGE_REVIEW: 原生／HOG 主要AP安全門檻觸發',flush=True);return 0
        if smoke:
            assert proof['control']['trace']==proof['hog']['trace']
            assert proof['control']['hog_grad_norm'] is None and proof['hog']['hog_grad_norm']>0
            write_json(ROOT/'artifacts/prefusion-hog-proof-v1.json',{'status':'passed','arms':proof})
            state['status']='training';write_json(state_path,state)
            print('HOG_PASSED: 真實更新、固定state、同資料trace、CPU推論strip等價',flush=True)
    state['status']='awaiting_analysis';write_json(state_path,state)
    print('ALL_DONE: HOG矩陣完成，僅比較共同訓練預算並接續分析',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
