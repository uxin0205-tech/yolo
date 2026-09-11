#!/usr/bin/env python3
"""CPU 稽核 heads-only 完整快照的逐分支參數、moment 與共享凍結。"""
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/direction1-20260908/heads-only-parent-recovery'


def main():
    torch.set_num_threads(2)
    summary=json.loads((RUN/'summary.json').read_text())
    parent=torch.load(summary['metadata']['parent'],map_location='cpu',weights_only=True,mmap=True)['state_dict']
    results=[]
    for epoch in range(1,6):
        snapshot=torch.load(RUN/f'checkpoints/epoch-{epoch:04d}.pt',map_location='cpu',weights_only=True,mmap=True)
        fixed=summary['metadata']['ema']['fixed_state_names']
        for bank in ('model_state','ema_state'):
            assert all(torch.equal(snapshot[bank][key],parent[key.removeprefix('base.')]) for key in fixed)
        branches={}
        for group in snapshot['optimizer_state']['param_groups']:
            for index,name in zip(group['params'],group['param_names']):
                if '.pose_head.' not in name and '.detect_head.' not in name: continue
                task='pose' if '.pose_head.' in name else 'detect'
                branch=name.split(f'.{task}_head.')[1].split('.')[0]
                row=branches.setdefault(f'{task}/{branch}',{'tensors':0,'changed':0,'optimizer_states':0,
                    'nonzero_moments':0,'step_min':None,'step_max':None,'delta_squared':0.0,'parent_squared':0.0})
                row['tensors']+=1
                before=parent[name.removeprefix('base.')].float()
                after=snapshot['model_state'][name].float()
                row['changed']+=int(not torch.equal(before,after))
                row['delta_squared']+=float((after-before).square().sum())
                row['parent_squared']+=float(before.square().sum())
                state=snapshot['optimizer_state']['state'].get(index)
                if state:
                    row['optimizer_states']+=1
                    row['nonzero_moments']+=int(bool(state['exp_avg'].ne(0).any()))
                    step=int(state['step'])
                    row['step_min']=step if row['step_min'] is None else min(row['step_min'],step)
                    row['step_max']=step if row['step_max'] is None else max(row['step_max'],step)
        for row in branches.values():
            row['relative_l2']=(row['delta_squared']/max(row['parent_squared'],1e-24))**.5
        results.append({'epoch':epoch,'fixed_states_exact':True,'branches':branches})
    output=RUN/'head-update-audit.json'
    with output.open('x') as handle:
        json.dump({'status':'complete','epochs':results,'note':'moment 為累積證據，不等於每一個 minibatch 梯度都非零'},handle,ensure_ascii=False,indent=2)
    print(json.dumps(results[-1],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
