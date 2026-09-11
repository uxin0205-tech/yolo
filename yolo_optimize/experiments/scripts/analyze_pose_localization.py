#!/usr/bin/env python3
"""CPU 重播已保存的 conf≥0.25 框，分開定位與信心排序證據。"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import box_iou

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/direction1-20260909/pose-error-audit'


def tensor(value, columns=None):
    result = torch.tensor(value, dtype=torch.float32)
    return result.reshape(-1, columns) if columns else result


def replay(record):
    pred, gt = record['predictions'], record['targets']
    pb, gb = tensor(pred['bboxes'], 4), tensor(gt['bboxes'], 4)
    pc, gc = tensor(pred['cls']), tensor(gt['cls'])
    iou = box_iou(gb, pb)
    correct = BaseValidator.match_predictions(SimpleNamespace(iouv=torch.linspace(.5,.95,10)), pc, gc, iou)
    counts = {str(c): correct[pc == c].sum(0).tolist() for c in (0,1)}
    objects = []
    for index in range(len(gc)):
        choices = torch.nonzero(pc == gc[index]).flatten()
        item = {'class': int(gc[index]), 'gt_index': index, 'iou': 0.0}
        if len(choices):
            chosen = int(choices[torch.argmax(iou[index, choices])])
            wh = (gb[index,2:]-gb[index,:2]).clamp_min(1e-6)
            pwh = (pb[chosen,2:]-pb[chosen,:2]).clamp_min(1e-6)
            center_error = ((pb[chosen,:2]+pb[chosen,2:]-gb[index,:2]-gb[index,2:])/2)/wh
            item.update(iou=float(iou[index,chosen]), conf=pred['conf'][chosen],
                center_error_normalized=float(center_error.norm()),
                size_log_error=float(torch.log(pwh/wh).abs().mean()))
        objects.append(item)
    return counts, objects


def main():
    data = {name: json.loads((RUN/f'{name}.json').read_text()) for name in ('parent','native_e5')}
    totals, objects, mismatches = {}, {}, []
    for name, payload in data.items():
        totals[name] = {'0':np.zeros(10,dtype=int),'1':np.zeros(10,dtype=int)}
        objects[name] = {}
        for row in payload['records']:
            counts, samples = replay(row)
            for cls in ('0','1'):
                totals[name][cls] += counts[cls]
                for col, key in ((0,'box50'),(5,'box75')):
                    expected = row['thresholds']['0.25'][cls][key]['tp']
                    if counts[cls][col] != expected:
                        mismatches.append({'model':name,'image':row['image'],'class':cls,'metric':key,
                            'replay':counts[cls][col],'official':expected})
            objects[name][row['image']] = samples
    paired = {str(c):[] for c in (0,1)}
    for path, rows in objects['parent'].items():
        other = objects['native_e5'][path]
        assert len(rows) == len(other)
        for a,b in zip(rows,other):
            assert (a['gt_index'],a['class']) == (b['gt_index'],b['class'])
            if min(a['iou'],b['iou']) >= .5:
                paired[str(a['class'])].append({key:b[key]-a[key] for key in
                    ('iou','conf','center_error_normalized','size_log_error')})
    report = {'status':'complete','device':'cpu','confidence':.25,
        'iou_thresholds':[round(.5+.05*i,2) for i in range(10)],
        'tp':{name:{cls:counts.tolist() for cls,counts in classes.items()} for name,classes in totals.items()},
        'official_50_75_replay_mismatches':mismatches,
        'paired_best_iou_descriptive_only':{cls:{'count':len(rows),
            'median_delta':{key:float(np.median([r[key] for r in rows])) for key in rows[0]}} if rows else {'count':0}
            for cls,rows in paired.items()},
        'limitations':['只保存 conf≥0.25，不可重建全 AP 或完整 confidence 排序。',
            '原圖座標經 clipping，與 letterbox official 匹配需核對，差異明列。',
            '逐 GT 的最佳 IoU 是描述性 oracle，非一對一官方 AP，不據此選閾值或部署。']}
    (RUN/'localization-analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
