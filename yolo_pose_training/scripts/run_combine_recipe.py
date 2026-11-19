"""官方 Pose 的 combine 式三階段訓練，獨立保存與雙解析度比較。"""
import fcntl
import json
from pathlib import Path
import sys
import traceback
import yaml
import run_queue as queue

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/combine_recipe/queue.yaml'
OUTPUT = ROOT/'artifacts/combine_recipe'
queue.STATUS = OUTPUT/'status.json'


def evaluate():
    from ultralytics import YOLO
    spec=yaml.safe_load(CONFIG.read_text())
    configs=[yaml.safe_load((ROOT/p).read_text()) for p in spec['stages']]
    results=[]
    # 歷史960數值已有相同formal val、batch8評估；只新增缺少的640對照。
    baselines=json.loads((OUTPUT/'baseline-960.json').read_text())
    for row in baselines:
        results.append(dict(row, imgsz=960, group='previous', evaluation_source='saved baseline'))
    controls=[yaml.safe_load((ROOT/f'configs/{name}.yaml').read_text()) for name in
              ['stage2_adamw_640','stage3_adamw_960','stage4_musgd_960']]
    jobs=[(c,size,'combine_recipe') for c in configs for size in spec['evaluation_imgsz']]
    jobs += [(c,640,'previous') for c in controls]
    for cfg,size,group in jobs:
        weight=ROOT/'runs'/cfg['name']/'weights/best.pt'
        m=YOLO(weight).val(data=cfg['data'],split='val',imgsz=size,batch=8,device=0,
                          workers=8,plots=True,project=str(ROOT/'evaluations/combine_recipe'),
                          name=f"{cfg['name']}_{size}",exist_ok=True)
        results.append({'stage':cfg['name'],'group':group,'imgsz':size,'checkpoint':str(weight),
                        'metrics':{k:float(v) for k,v in m.results_dict.items()},
                        'selection_score':(float(m.box.map)+float(m.pose.map))/2,
                        'per_class':{name:{'box_map50_95':float(m.box.maps[i]),
                                           'pose_map50_95':float(m.pose.maps[i])}
                                     for i,name in m.names.items()}})
        (OUTPUT/'comparison.json').write_text(json.dumps(results,indent=2))
    selections={str(size):max([r for r in results if r['imgsz']==size],key=lambda r:r['selection_score'])
                for size in spec['evaluation_imgsz']}
    (OUTPUT/'selection.json').write_text(json.dumps(selections,indent=2))
    lines=['# combine 式獨立 Pose 完成比較','',
           '官方 yolo26m-pose 初始化；固定 bbat5-v1 formal val 683 張；batch8，按解析度分開比較。',
           '歷史960成績引用既有同設定評估；本次新增三階段640／960及舊Stage2／3／4的640評估。','',
           '| 模型 | 解析度 | Box mAP50–95 | Pose mAP50–95 | 平均 |',
           '| --- | --- | --- | --- | --- |']
    for r in sorted(results,key=lambda x:(x['imgsz'],x['stage'])):
        m=r['metrics'];lines.append(f"| {r['stage']} | {r['imgsz']} | {m['metrics/mAP50-95(B)']:.5f} | {m['metrics/mAP50-95(P)']:.5f} | {r['selection_score']:.5f} |")
    lines += ['']+[f"{size}最佳：{row['stage']}，權重 {row['checkpoint']}。" for size,row in selections.items()]
    lines += ['', '每階段best.pt、last.pt、last-resumable.pt均保留在runs/<stage>/weights。既有artifacts/final未替換。',
              '困難與解法：見工作紀錄與logs。未解事項：單seed、formal val選模，非獨立test；整套recipe改動，不能歸因於單項超參數。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'artifacts/queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=yaml.safe_load(CONFIG.read_text())
    for config in spec['stages']:
        cfg=yaml.safe_load((ROOT/config).read_text());run=ROOT/'runs'/cfg['name']
        if (run/'completed.json').is_file():continue
        if run.exists() and not (run/'weights/last-resumable.pt').is_file():
            raise RuntimeError(f"{cfg['name']}未完成且無完整checkpoint，停止避免覆寫")
        assert Path(cfg['model']).is_file(),cfg['model']
        queue.execute(cfg['name'],[sys.executable,'-u',str(ROOT/'scripts/train_stage.py'),str(ROOT/config),str(cfg['batch'])],spec['monitor_seconds'])
    queue.execute('combine_recipe_comparison',[sys.executable,'-u',str(Path(__file__).resolve()),'--evaluate'],spec['monitor_seconds'])
    queue.status('completed',report=str(OUTPUT/'report.md'))

if __name__=='__main__':
    try:
        if '--evaluate' in sys.argv:evaluate()
        else:main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True,exist_ok=True);queue.status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
