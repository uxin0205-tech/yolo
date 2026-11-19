"""由新 P3 最佳權重分支精修，保存 AP 與像素誤差比較。"""
import fcntl
import json
from pathlib import Path
import sys
import traceback
import yaml
import run_queue as queue

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/refinement'
CONFIG=ROOT/'configs/refinement/queue.yaml'
queue.STATUS=OUTPUT/'status.json'


def per_class_safe(candidate,reference,tolerance):
    return all(candidate['per_class'][name][metric]>=reference['per_class'][name][metric]-tolerance
               for name in ['ball','bat'] for metric in ['box_map50_95','pose_map50_95'])


def evaluate():
    from ultralytics import YOLO
    spec=yaml.safe_load(CONFIG.read_text());configs=[yaml.safe_load((ROOT/p).read_text()) for p in spec['stages']]
    baseline=json.loads((OUTPUT/'baseline-comparison.json').read_text())
    rows=[dict(r,evaluation_source='saved previous evaluation') for r in baseline]
    for c in configs:
        weight=ROOT/'runs'/c['name']/'weights/best.pt'
        for size in spec['evaluation_imgsz']:
            m=YOLO(weight).val(data=c['data'],split='val',imgsz=size,batch=8,device=0,
                              workers=8,plots=True,project=str(ROOT/'evaluations/refinement'),
                              name=f"{c['name']}_{size}",exist_ok=True)
            rows.append({'stage':c['name'],'group':'refinement','imgsz':size,'checkpoint':str(weight),
                         'metrics':{k:float(v) for k,v in m.results_dict.items()},
                         'selection_score':(float(m.box.map)+float(m.pose.map))/2,
                         'per_class':{name:{'box_map50_95':float(m.box.maps[i]),'pose_map50_95':float(m.pose.maps[i])}
                                      for i,name in m.names.items()}})
            (OUTPUT/'comparison.json').write_text(json.dumps(rows,indent=2))
    selectors={}
    for size in spec['evaluation_imgsz']:
        pool=[r for r in rows if r['imgsz']==size]
        ref=max([r for r in baseline if r['imgsz']==size],key=lambda r:r['selection_score'])
        allowed=[ref]+[r for r in pool if r.get('group')=='refinement' and
                       r['selection_score']>ref['selection_score'] and per_class_safe(r,ref,spec['per_class_ap_drop_tolerance'])]
        selectors[str(size)]={'reference':ref,'recommended':max(allowed,key=lambda r:r['selection_score']),
                              'best_average':max(pool,key=lambda r:r['selection_score']),
                              'best_pose':max(pool,key=lambda r:r['metrics']['metrics/mAP50-95(P)']),
                              'best_ball':max(pool,key=lambda r:sum(r['per_class']['ball'].values())/2)}
    (OUTPUT/'selection.json').write_text(json.dumps(selectors,indent=2))


def finish_report():
    rows=json.loads((OUTPUT/'comparison.json').read_text());selectors=json.loads((OUTPUT/'selection.json').read_text())
    lines=['# 新 P3 精修比較','', '相同 P3 best.pt 出發；固定 bbat5-v1 formal val 683 張，AP評估batch8；640與960分開比較。',
           '歷史成績引用既有同口徑評估。訓練640用physical32、960用physical16，兩者nbs128；不是嚴格只差解析度的實驗。','',
           '| 模型 | 解析度 | Box mAP50–95 | Pose mAP50–95 | Ball Box | Ball Pose | Bat Box | Bat Pose |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        if r.get('group')!='refinement' and r['stage'] not in ['combine_recipe_p3_640','stage4_musgd_960']:continue
        m=r['metrics'];p=r['per_class']
        values=[m['metrics/mAP50-95(B)'],m['metrics/mAP50-95(P)'],p['ball']['box_map50_95'],p['ball']['pose_map50_95'],p['bat']['box_map50_95'],p['bat']['pose_map50_95']]
        lines.append(f"| {r['stage']} | {r['imgsz']} | "+' | '.join(f'{x:.5f}' for x in values)+' |')
    for size,selection in selectors.items():
        lines += ['',f"{size}建議：{selection['recommended']['stage']}。",f"綜合最高：{selection['best_average']['stage']}；Pose最高：{selection['best_pose']['stage']}；ball兩项AP平均最高：{selection['best_ball']['stage']}。"]
    lines += ['', '推薦條件：平均AP提升，且ball／bat各自Box／Pose AP相較該解析度原綜合最佳，任一項下降不得超過0.005（0.5個百分點）；僅產生建議，不替換舊final。',
              '', '## 固定操作點定位診斷','',
              'conf0.25、同類別IoU0.5匹配；像素誤差只計匹配且GT可見的點，PCK全體版本將漏檢視為失敗，不能當成AP。','',
              '| 模型 | 解析度 | 類別 | 漏檢／GT | Keypoint中位像素誤差 | Keypoint P90 | 全體PCK@0.1 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    folders=[OUTPUT/'baseline-errors']+[OUTPUT/(Path(p).stem+'-errors') for p in yaml.safe_load(CONFIG.read_text())['stages']]
    for folder in folders:
        d=json.loads((folder/'summary.json').read_text())
        for res in d['results']:
            for name,c in res['classes'].items():
                lines.append(f"| {Path(d['weights']).parents[1].name} | {res['imgsz']} | {name} | {c['fn']}/{c['gt']} | {c['keypoint_error_px']['median']} | {c['keypoint_error_px']['p90']} | {c['pck_at_0p1_all_visible_gt']} |")
    lines += ['', '所有run保留best.pt、last.pt及last-resumable.pt。困難與解法見工作紀錄。未解限制：單seed、formal val選模；像素誤差受成功匹配樣本組成影響，不能忽略漏檢或直接當作整體定位能力。資料缺少遮擋／動態模糊真值。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'artifacts/queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=yaml.safe_load(CONFIG.read_text())
    assert (OUTPUT/'baseline-errors/summary.json').is_file()
    for config in spec['stages']:
        cfg=yaml.safe_load((ROOT/config).read_text());run=ROOT/'runs'/cfg['name']
        if (run/'completed.json').is_file():continue
        if run.exists() and not (run/'weights/last-resumable.pt').is_file():raise RuntimeError('未完成run缺少checkpoint，停止避免覆寫')
        queue.execute(cfg['name'],[sys.executable,'-u',str(ROOT/'scripts/train_stage.py'),str(ROOT/config),str(cfg['batch'])],600)
    queue.execute('refinement_ap_comparison',[sys.executable,'-u',str(Path(__file__).resolve()),'--evaluate'],600)
    for config in spec['stages']:
        cfg=yaml.safe_load((ROOT/config).read_text());output=OUTPUT/(Path(config).stem+'-errors')
        queue.execute(cfg['name']+'_errors',[sys.executable,'-u',str(ROOT/'scripts/analyze_pose_errors.py'),
                      '--weights',str(ROOT/'runs'/cfg['name']/'weights/best.pt'),'--output',str(output)],600)
    finish_report();queue.status('completed',report=str(OUTPUT/'report.md'))

if __name__=='__main__':
    try:
        if '--evaluate' in sys.argv:evaluate()
        else:main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True,exist_ok=True);queue.status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
