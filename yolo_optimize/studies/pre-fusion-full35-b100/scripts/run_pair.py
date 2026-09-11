"""既定 A0 control／QK 兩臂，子工作退出才讀結果；正常等待最多 600 秒。"""
import json
import subprocess
import sys
from common import ROOT,write_json


def main():
    state_path=ROOT/'artifacts/a0-pair-state.json'
    if state_path.exists():raise FileExistsError('已存在 queue，請稽核後另開明確版本，不自動重跑')
    a=json.loads((ROOT/'artifacts/a0-control-smoke/summary.json').read_text())
    b=json.loads((ROOT/'artifacts/a0-qk-smoke-v2/summary.json').read_text())
    assert a['first_macro_trace_sha256']==b['first_macro_trace_sha256']
    assert a['qk_optimizer_state_count']==0 and b['qk_optimizer_state_count']==12
    assert json.loads((ROOT/'artifacts/a0-qk-smoke/inference-bittrue.json').read_text())['status']=='passed'
    state={'status':'running','jobs':[],'monitor_wait_seconds':600,'accuracy_winner':None}
    write_json(state_path,state)
    for arm in ('control','qk'):
        name=f'a0-{arm}-recovery-v1'
        job={'name':name,'arm':arm,'status':'running'};state['jobs'].append(job);write_json(state_path,state)
        command=[sys.executable,str(ROOT/'scripts/monitor.py'),'--name',name,'--',sys.executable,'-u',
            str(ROOT/'scripts/recover_a0.py'),'--arm',arm,'--name',name]
        process=subprocess.Popen(command)
        while True:
            try:code=process.wait(timeout=600);break
            except subprocess.TimeoutExpired:pass
        if code:
            job['status']='error';state['status']='error';write_json(state_path,state);return code
        summary=json.loads((ROOT/'artifacts'/name/'summary.json').read_text())
        if summary['status'] not in ('complete','patience_stop','paused_for_analysis'):
            raise RuntimeError('子工作未到合法終態')
        job.update(status=summary['status'],epochs=len(summary['epochs']))
        write_json(state_path,state)
    state['status']='awaiting_analysis';write_json(state_path,state)
    print('ALL_DONE: 成對訓練已終止，待 AP／state／推論分析；不是增準成功',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
