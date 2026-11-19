"""獨立低 LR 候選：沿用背景監測，完成後與既有統一評估比較。"""
import fcntl
import json
from pathlib import Path
import sys
import traceback
import yaml
import run_queue as queue

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/candidates/stage4_musgd_lr00038.yaml'
OUTPUT = ROOT/'artifacts/candidates/stage4_musgd_lr00038'
queue.STATUS = OUTPUT/'status.json'


def evaluate():
    from ultralytics import YOLO
    cfg = yaml.safe_load(CONFIG.read_text())
    weight = ROOT/'runs'/cfg['name']/'weights/best.pt'
    metrics = YOLO(weight).val(data=cfg['data'], split='val', imgsz=960, batch=8, device=0,
                              workers=8, plots=True, project=str(ROOT/'evaluations'),
                              name=cfg['name'], exist_ok=True)
    candidate = {'stage':cfg['name'], 'checkpoint':str(weight),
                 'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
                 'per_class':{name:{'box_map50_95':float(metrics.box.maps[i]),
                                    'pose_map50_95':float(metrics.pose.maps[i])}
                              for i,name in metrics.names.items()},
                 'selection_score':(float(metrics.box.map)+float(metrics.pose.map))/2}
    baseline = json.loads((OUTPUT/'baseline-comparison.json').read_text())
    control = next(row for row in baseline if row['stage']=='stage4_musgd_960')
    deltas = {k:candidate['metrics'][k]-control['metrics'][k] for k in candidate['metrics']}
    result = {'candidate':candidate,'control':control,'metric_deltas':deltas,
              'selection_score_delta':candidate['selection_score']-control['selection_score']}
    (OUTPUT/'comparison.json').write_text(json.dumps(result,indent=2))
    lines = ['# MuSGD 低 LR 候選比較','',
             '相同 Stage 3 best.pt 初始化；原 Stage 4 lr0=0.001，候選 lr0=0.00038，其餘訓練參數一致。',
             '固定 bbat5-v1 formal val 683 張、960、batch8。控制組使用既有統一評估結果，候選為本次評估；沒有重跑控制組。','',
             '| 指標 | 原 Stage 4 | 低 LR 候選 | 差值 |','| --- | --- | --- | --- |']
    for k in candidate['metrics']:
        if k=='fitness': continue
        lines.append(f"| {k} | {control['metrics'][k]:.6f} | {candidate['metrics'][k]:.6f} | {deltas[k]:+.6f} |")
    lines += ['',f"Box／Pose mAP50–95 平均差值：{result['selection_score_delta']:+.6f}。",'',
              f"候選權重：{weight}；同目錄保留 last.pt 與 last-resumable.pt。",'',
              '困難與解法：執行日誌見 logs/；本表不代表跨 seed 穩定性或獨立 test 成績。',
              '未解事項：單 seed、相同 formal val 選模，差異不能視為統計顯著；既有 artifacts/final 權重未替換。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'artifacts/queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=yaml.safe_load(CONFIG.read_text())
    control=yaml.safe_load((ROOT/'configs/stage4_musgd_960.yaml').read_text())
    assert {k for k in cfg if cfg[k]!=control[k]}=={'name','lr0'}
    run=ROOT/'runs'/cfg['name']
    snapshot=OUTPUT/'baseline-comparison.json'
    if not snapshot.exists(): snapshot.write_bytes((ROOT/'artifacts/comparison.json').read_bytes())
    if not (run/'completed.json').exists():
        if run.exists() and not (run/'weights/last-resumable.pt').is_file():
            raise RuntimeError('候選已有未完成 run 且缺少完整 checkpoint，停止避免覆寫')
        queue.execute(cfg['name'],[sys.executable,'-u',str(ROOT/'scripts/train_stage.py'),str(CONFIG),str(cfg['batch'])],600)
    queue.execute(cfg['name']+'_comparison',[sys.executable,'-u',str(Path(__file__).resolve()),'--evaluate'],600)
    queue.status('completed',report=str(OUTPUT/'report.md'))

if __name__=='__main__':
    try:
        if '--evaluate' in sys.argv: evaluate()
        else: main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        queue.status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
