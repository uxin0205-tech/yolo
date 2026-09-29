"""唯讀稽核 MASF 完成紀錄，輸出同回合差值與來源雜湊。"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'yolo_optimize/studies/pre-fusion-full35-b100/artifacts'
KEYS = ['coco/box/map50_95', 'coco/person/box/map50_95', 'coco/ball/box/map50_95', 'coco/bat/box/map50_95']
sources = []
def read(path):
    raw = path.read_bytes()
    sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)

names = ['masf-p3-control-v1', 'masf-p3-shared-v1', 'masf-p3-fork-v1', 'masf-head-control-v1', 'masf-head-fork-v1', 'masf-task-bridge-v1', 'masf-p2-control-v1', 'masf-p2-p2-v1']
runs = {n: read(BASE/n/'summary.json') for n in names}
for name, run in runs.items():
    assert run['status'] == 'complete' and len(run['epochs']) == 5
    for row in run['epochs']:
        assert row['images'] == 118287 and row['macros'] == 925
        assert (BASE/name/f"epoch-{row['epoch']:02d}-resume.pt").is_file()
        assert all(math.isfinite(row[bank][k]) and 0 <= row[bank][k] <= 1 for bank in ['ema', 'live'] for k in KEYS)

pairs = [('masf-p3-shared-v1', 'masf-p3-control-v1'), ('masf-p3-fork-v1', 'masf-p3-control-v1'), ('masf-head-fork-v1', 'masf-head-control-v1'), ('masf-task-bridge-v1', 'masf-head-fork-v1'), ('masf-task-bridge-v1', 'masf-head-control-v1'), ('masf-p2-p2-v1', 'masf-p2-control-v1')]
rows = []
for candidate, control in pairs:
    a,b = runs[candidate],runs[control]
    assert a['parent_sha256'] == b['parent_sha256']
    assert a['first_macro_trace_sha256'] == b['first_macro_trace_sha256']
    for x,y in zip(a['epochs'], b['epochs']):
        assert x['epoch'] == y['epoch']
        row = {'candidate': candidate, 'control': control, 'epoch': x['epoch']}
        row.update({k+'_delta_pp': (x['ema'][k]-y['ema'][k])*100 for k in KEYS})
        rows.append(row)

p2,c = runs['masf-p2-p2-v1'],runs['masf-p2-control-v1']
queue=read(BASE/'masf-p2-queue-v1-state.json')
checks=[]
for x,y,q in zip(p2['epochs'],c['epochs'],queue['comparisons']):
    delta={k:x['ema'][k]-y['ema'][k] for k in KEYS}
    eligible=all(delta[k]>=0 for k in KEYS[:2]) and max(delta[k] for k in KEYS[:2])>=.001 and all(x['ema'][k]>=p2['reference'][k] for k in KEYS[:2])
    assert eligible==q['eligible'] and all(abs(delta[k]-q['delta'][k])<1e-12 for k in KEYS)
    checks.append(eligible)
assert not any(checks) and queue['threshold_passed'] is False
for name in ['masf-p2-control-v1','masf-p2-p2-v1']:
    path=BASE/name/'progress.jsonl'; raw=path.read_bytes()
    sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()})
    events=[json.loads(line) for line in raw.splitlines()]
    assert len(events)==4625 and collections.Counter(e['epoch'] for e in events)=={i:925 for i in range(1,6)}
    assert events[-1]['optimizer_steps']==4625

qat=read(ROOT/'yolo_quantize/artifacts/queues/full-model-continuous-0907/qat-recovery-summary.json')
qat_results=[]
for job in qat['jobs']:
    if not job['job_id'].startswith('masf-'):continue
    completion=read(Path(job['completion']['path']))
    assert sources[-1]['sha256']==job['completion']['sha256'] and completion['epochs_completed']==5
    for epoch in job['epochs']:
        metrics=read(Path(epoch['source']['path']))
        assert sources[-1]['sha256']==epoch['source']['sha256']
    e=job['epochs'][-1]
    qat_results.append({'run':job['job_id'],'epochs':len(job['epochs']), 'worst_map50_drop_pp': max(0,-min(v for k,v in e['total_deltas'].items() if k.endswith('/map50')))*100,'worst_map50_95_drop_pp':max(0,-min(v for k,v in e['total_deltas'].items() if k.endswith('/map50_95')))*100})
result={'verified_prefusion_epochs':40,'matched_comparison_rows':len(rows),'p2_gate_passed':any(checks),'p2_mean_epoch_seconds':{n:statistics.mean(e['elapsed_seconds'] for e in runs[n]['epochs']) for n in ['masf-p2-control-v1','masf-p2-p2-v1']},'p2_alpha_ema':[e['alpha_ema'] for e in p2['epochs']], 'qat':qat_results,'sources':sources,'limits':['未載入模型或重跑 GPU 驗證；checkpoint 只核對存在。','首 macro trace 一致不代表本次逐張稽核完整資料順序。','本次 gate 重算以既有 EMA summary 為依據；無多 seed 或獨立 test。']}
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (OUT/'paired-deltas.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False,indent=2))
