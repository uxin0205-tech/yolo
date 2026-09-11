#!/usr/bin/env python3
"""從實際 summary／checkpoint 建立第一輪 CSV 與證據索引；不啟動 GPU。"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/direction1-20260908'
OUT=ROOT.parent/'proposals/integrated-roadmap/results'
RUNS={
    'native':'native-parent-ema-control-adopted',
    'quarter_lr':'native-quarter-lr-control',
    'hog':'hog-parent-ema-band-v1',
    'rep17':'rep17-parent-ema-control',
    'masf_off':'masf-off-parent-bridge',
    'heads':'heads-only-parent-recovery',
}
WEIGHTS={'coco/box/map50_95':.2,'coco/person/box/map50_95':.2,
    'bbat/box/map50_95':.2,'bbat/pose/map50_95':.4}


def read(path):
    return json.loads(path.read_text())


def joint(metrics):
    return sum(value*metrics[key] for key,value in WEIGHTS.items())


def main():
    parent=read(BASE/'best-joint-revalidation/summary.json')
    metrics=parent['metrics']['bittrue']
    keys=sorted(key for key in metrics if key.endswith('map50_95'))
    assert len(keys)==8
    origin=Path(parent['provenance']['checkpoint'])
    digest=hashlib.sha256(origin.read_bytes()).hexdigest()
    assert digest==parent['provenance']['checkpoint_sha256']
    summaries={name:read(BASE/run/'summary.json') for name,run in RUNS.items()}
    control={row['epoch']:row['metrics']['bittrue'] for row in summaries['native']['epochs']}
    rows=[];checks=[]
    for name,summary in summaries.items():
        assert summary['status'] in ('complete','paused_for_analysis')
        assert Path(summary['metadata']['parent']).resolve()==origin.resolve()
        assert summary['metadata']['warmup_epochs']==1
        assert summary['metadata']['physical_batch']==32
        for epoch in summary['epochs']:
            number=epoch['epoch']
            weights=BASE/RUNS[name]/f'inference/epoch-{number:04d}.pt'
            snapshot=BASE/RUNS[name]/f'checkpoints/epoch-{number:04d}.pt'
            assert weights.is_file() and weights.stat().st_size>0
            assert snapshot.is_file() and snapshot.stat().st_size>0
            for bank,values in (('ema',epoch['metrics']['bittrue']),('live',epoch['live_metrics'])):
                assert all(key in values and math.isfinite(values[key]) and 0<=values[key]<=1 for key in keys)
                row={'run':name,'epoch':number,'bank':bank,'joint':joint(values),
                    'joint_delta_parent':joint(values)-joint(metrics),
                    'joint_delta_native_same_epoch':joint(values)-joint(control[number]) if bank=='ema' else '',
                    'run_status':summary['status'],'source':str((BASE/RUNS[name]/'summary.json').relative_to(ROOT))}
                row.update({key:values[key] for key in keys})
                row.update({key+'_delta_parent':values[key]-metrics[key] for key in keys})
                rows.append(row)
        best=max(summary['epochs'],key=lambda e:joint(e['metrics']['bittrue']))
        b=best['metrics']['bittrue']
        checks.append({'run':name,'completed_epochs':len(summary['epochs']),
            'best_ema_epoch':best['epoch'],'best_ema_joint':joint(b),
            'best_parent_numeric_gate':all(b[k]>=metrics[k]-.001 for k in keys) and
                (joint(b)>=joint(metrics)+.001 or b['coco/box/map50_95']>=metrics['coco/box/map50_95']+.001),
            'checkpoint_files_present':True,'scope':summary['metadata']['scope']})
    OUT.mkdir(exist_ok=True)
    with (OUT/'epoch-comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'status':'evidence_tables_built_not_goal_complete','parent':str(origin),
        'parent_sha256_verified':digest,'parent_joint':joint(metrics),'metric_keys':keys,
        'rows':len(rows),'training_runs':checks,
        'limitations':['CSV 僅比較 EMA 對同 epoch native EMA；live 不混入 selector。',
            '不同長度只比較共同 epoch；本腳本核對 checkpoint 存在與大小，不聲稱重新載入每份完整快照。',
            'numeric gate 不代替同場景盲測、paired seed 或目標硬體驗證。']}
    (OUT/'evidence-index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'runs':[(c['run'],c['completed_epochs'],c['best_parent_numeric_gate']) for c in checks],
        'parent_sha256_verified':digest},ensure_ascii=False))


if __name__=='__main__':main()
