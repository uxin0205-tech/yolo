"""三階段基準LR遞減：独立排程、全程監測與同解析度比較。"""
import fcntl
import json
from pathlib import Path
import sys
import traceback
import yaml
import run_queue as queue

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/stagewise_lr'
CONFIG=ROOT/'configs/stagewise_lr/queue.yaml'
queue.STATUS=OUTPUT/'status.json'


def evaluate():
    from ultralytics import YOLO
    spec=yaml.safe_load(CONFIG.read_text());baseline=json.loads((OUTPUT/'baseline-comparison.json').read_text())
    rows=[dict(r,evaluation_source='saved baseline evaluation') for r in baseline]
    for path in spec['stages']:
        cfg=yaml.safe_load((ROOT/path).read_text());weight=ROOT/'runs'/cfg['name']/'weights/best.pt'
        for size in spec['evaluation_imgsz']:
            m=YOLO(weight).val(data=cfg['data'],split='val',imgsz=size,batch=8,device=0,workers=8,
                              plots=True,project=str(ROOT/'evaluations/stagewise_lr'),
                              name=f"{cfg['name']}_{size}",exist_ok=True)
            rows.append({'stage':cfg['name'],'group':'stagewise_lr','imgsz':size,'checkpoint':str(weight),
                         'metrics':{k:float(v) for k,v in m.results_dict.items()},
                         'selection_score':(float(m.box.map)+float(m.pose.map))/2,
                         'per_class':{name:{'box_map50_95':float(m.box.maps[i]),'pose_map50_95':float(m.pose.maps[i])}
                                      for i,name in m.names.items()}})
            (OUTPUT/'comparison.json').write_text(json.dumps(rows,indent=2))
    selections={}
    for size in spec['evaluation_imgsz']:
        pool=[r for r in rows if r['imgsz']==size]
        ref=max([r for r in baseline if r['imgsz']==size],key=lambda r:r['selection_score'])
        accepted=[ref]
        for row in pool:
            if row.get('group')!='stagewise_lr':continue
            safe=all(row['per_class'][name][key]>=ref['per_class'][name][key]-spec['per_class_ap_drop_tolerance']
                     for name in ['ball','bat'] for key in ['box_map50_95','pose_map50_95'])
            if safe and row['selection_score']>ref['selection_score']:accepted.append(row)
        selections[str(size)]={'reference':ref,'recommended':max(accepted,key=lambda r:r['selection_score']),
                               'best_average':max(pool,key=lambda r:r['selection_score']),
                               'best_pose':max(pool,key=lambda r:r['metrics']['metrics/mAP50-95(P)'])}
    (OUTPUT/'selection.json').write_text(json.dumps(selections,indent=2))
    lines=['# 逐階段降低 LR 訓練結果','',
           '從官方yolo26m-pose重新開始，P1／P2／P3基準LR為0.00038／0.00019／0.000095，各自cosine到0.5倍。',
           '固定bbat5-v1 formal val 683張、AP評估batch8，同解析度比較。歷史結果引用已完成的同口徑評估。','',
           '| 階段 | 解析度 | Box mAP50–95 | Pose mAP50–95 | Ball Box | Ball Pose | Bat Box | Bat Pose |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        if row.get('group')!='stagewise_lr' and row['stage'] not in ['combine_recipe_p3_640','stage4_musgd_960','refine_p3_960_lr0001']:continue
        m=row['metrics'];p=row['per_class'];vals=[m['metrics/mAP50-95(B)'],m['metrics/mAP50-95(P)']]+[p[n][k] for n in ['ball','bat'] for k in ['box_map50_95','pose_map50_95']]
        lines.append(f"| {row['stage']} | {row['imgsz']} | "+' | '.join(f'{v:.5f}' for v in vals)+' |')
    for size,s in selections.items():
        lines += ['',f"{size}綜合最高：{s['best_average']['stage']}；Pose最高：{s['best_pose']['stage']}；通過類別保護的建議：{s['recommended']['stage']}。"]
    lines += ['', '類別保護要求新方案平均AP提高，且ball／bat各自Box／Pose AP退步不超過0.005；以同解析度歷史平均AP最高模型為基準，僅產生建議，不替換既有final。',
              '各階段weights/保留best.pt、last.pt、last-resumable.pt。',
              '困難與解法見工作紀錄。限制：單seed與formal val選模；舊P1第6輪曾延長排程，新P1從40輪規劃開始，歷史排程並非完全相同。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'artifacts/queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=yaml.safe_load(CONFIG.read_text())
    for config in spec['stages']:
        cfg=yaml.safe_load((ROOT/config).read_text());run=ROOT/'runs'/cfg['name']
        if (run/'completed.json').is_file():continue
        if run.exists() and not (run/'weights/last-resumable.pt').is_file():raise RuntimeError('run未完成且無checkpoint，停止避免覆寫')
        assert Path(cfg['model']).is_file(),cfg['model']
        queue.execute(cfg['name'],[sys.executable,'-u',str(ROOT/'scripts/train_stage.py'),str(ROOT/config),str(cfg['batch'])],spec['monitor_seconds'])
    queue.execute('stagewise_lr_comparison',[sys.executable,'-u',str(Path(__file__).resolve()),'--evaluate'],spec['monitor_seconds'])
    queue.status('completed',report=str(OUTPUT/'report.md'))

if __name__=='__main__':
    try:
        if '--evaluate' in sys.argv:evaluate()
        else:main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True,exist_ok=True);queue.status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
