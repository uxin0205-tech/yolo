#!/usr/bin/env python3
"""以原 evaluator 的全量 flags 重建 AP，區分候選召回上限與排序差。"""
import json
from pathlib import Path
import numpy as np
from ultralytics.utils.metrics import ap_per_class, compute_ap

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/direction1-20260909/pose-ranking-audit'


def main():
    result = {}
    for name in ('parent','native_e5'):
        payload = json.loads((RUN/f'{name}.json').read_text())
        rows = [row['ranking'] for row in payload['records']]
        conf = np.concatenate([np.array(row['conf']) for row in rows])
        classes = np.concatenate([np.array(row['pred_cls']) for row in rows])
        targets = np.concatenate([np.array(row['target_cls']) for row in rows])
        result[name] = {}
        for kind, flag in (('box','tp'),('pose','tp_p')):
            tp = np.concatenate([np.array(row[flag],dtype=bool).reshape(-1,10) for row in rows])
            official = ap_per_class(tp, conf, classes, targets, plot=False)
            ap, ids = official[5], official[6]
            for index, cls in enumerate(ids):
                label = 'ball' if cls==0 else 'bat'
                expected = payload['metrics'][f'bbat/{label}/{kind}/map50_95']
                assert abs(float(ap[index].mean())-expected) < 1e-12
                mask = classes==cls
                count = int((targets==cls).sum())
                counts = tp[mask].sum(0)
                oracle = []
                for column in range(10):
                    # 每個 IoU 分別用 GT 把 TP 排前面；僅分析上界，絕不可作預測。
                    correct = np.sort(tp[mask,column])[::-1].astype(float)
                    tp_cum = correct.cumsum()
                    precision = tp_cum / np.arange(1,len(correct)+1)
                    oracle.append(float(compute_ap(tp_cum/count,precision)[0]))
                result[name][f'{label}_{kind}'] = {'ap':ap[index].tolist(),
                    'tp':counts.tolist(),'gt':count,'pred':int(mask.sum()),
                    'max_recall':(counts/count).tolist(),'label_oracle_ap':oracle,
                    'mean_sort_gap':float(np.mean(oracle)-ap[index].mean())}
    delta = {}
    for key,a in result['parent'].items():
        b = result['native_e5'][key]
        delta[key] = {'ap_by_iou':(np.array(b['ap'])-a['ap']).tolist(),
            'mean_ap':float(np.mean(b['ap'])-np.mean(a['ap'])),
            'mean_oracle_ap':float(np.mean(b['label_oracle_ap'])-np.mean(a['label_oracle_ap'])),
            'mean_sort_gap':b['mean_sort_gap']-a['mean_sort_gap'],
            'tp_by_iou':(np.array(b['tp'])-a['tp']).tolist()}
    report = {'status':'passed','official_ap_reconstruction_exact':True,'device':'cpu',
        'results':result,'delta_native_minus_parent':delta,
        'oracle_is_label_leakage_diagnostic_not_candidate':True,
        'interpretation':'AP差 = 同TP候選的oracle上界差 − oracle與實際排序差的變化；這是數值分解，不是可部署校準收益或訓練唯一根因。'}
    (RUN/'ranking-analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','official_ap_reconstruction_exact':True,'delta':delta},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
