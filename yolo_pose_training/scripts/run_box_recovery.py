"""保留Pose最佳模型，對照Box loss加權能否改善框定位。"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import traceback
import yaml
import run_queue as queue

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/box_recovery/queue.yaml'
OUTPUT=ROOT/'artifacts/box_recovery'
queue.STATUS=OUTPUT/'status.json'


def verify_preserved():
    provenance=json.loads((OUTPUT/'provenance.json').read_text());root=Path(provenance['preserved_bundle'])
    for entry in json.loads((root/'manifest.json').read_text()):
        assert hashlib.sha256((root/entry['file']).read_bytes()).hexdigest()==entry['sha256'],entry['file']


def acceptable(candidate,reference,policy):
    m=candidate['metrics'];r=reference['metrics']
    return (candidate['selection_score']>reference['selection_score']
            and m['metrics/mAP50-95(B)']>=r['metrics/mAP50-95(B)']+policy['min_box_ap_gain']
            and m['metrics/mAP50-95(P)']>=r['metrics/mAP50-95(P)']-policy['max_pose_ap_drop']
            and all(candidate['per_class'][n]['pose_map50_95']>=reference['per_class'][n]['pose_map50_95']-policy['max_pose_ap_drop'] for n in ['ball','bat'])
            and all(candidate['per_class'][n]['box_map50_95']>=reference['per_class'][n]['box_map50_95']-policy['max_per_class_ap_drop'] for n in ['ball','bat']))


def evaluate():
    from ultralytics import YOLO
    spec=yaml.safe_load(CONFIG.read_text());rows=json.loads((OUTPUT/'references.json').read_text())
    ref=next(row for row in rows if row['stage']=='stagewise_lr_p3_640')
    for config in spec['stages']:
        c=yaml.safe_load((ROOT/config).read_text());weight=ROOT/'runs'/c['name']/'weights/best.pt'
        m=YOLO(weight).val(data=c['data'],split='val',imgsz=640,batch=8,device=0,workers=8,plots=True,
                          project=str(ROOT/'evaluations/box_recovery'),name=c['name'],exist_ok=True)
        row={'stage':c['name'],'group':'box_recovery','imgsz':640,'checkpoint':str(weight),
             'metrics':{k:float(v) for k,v in m.results_dict.items()},
             'selection_score':(float(m.box.map)+float(m.pose.map))/2,
             'per_class':{name:{'box_map50_95':float(m.box.maps[i]),'pose_map50_95':float(m.pose.maps[i])} for i,name in m.names.items()}}
        row['meets_recovery_target']=acceptable(row,ref,spec);rows.append(row)
        (OUTPUT/'comparison.json').write_text(json.dumps(rows,indent=2))
    accepted=[r for r in rows if r.get('meets_recovery_target')]
    selected=max(accepted,key=lambda r:r['selection_score']) if accepted else ref
    (OUTPUT/'selection.json').write_text(json.dumps({'reference':ref,'suggested':selected,'policy':spec,'automatic_replacement':False},indent=2))
    lines=['# 保留Pose精度的框定位精修結果','',
           '相同逐階段降LR P3 best初始化、固定640／batch32訓練；對照Box loss7.5，候選10.0，其他參數相同。',
           '固定bbat5-v1 formal val 683張、評估640／batch8；兩個參考模型引用既有同口徑結果。','',
           '| 模型 | Box AP | Pose AP | Ball Box | Ball Pose | Bat Box | Bat Pose |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        m=row['metrics'];p=row['per_class'];v=[m['metrics/mAP50-95(B)'],m['metrics/mAP50-95(P)']]+[p[n][k] for n in ['ball','bat'] for k in ['box_map50_95','pose_map50_95']]
        lines.append(f"| {row['stage']} | "+' | '.join(f'{x:.6f}' for x in v)+' |')
    lines += ['',f"條件式建議：{selected['stage']}。",'',
              '條件：相對本次保留模型平均AP提高且Box AP至少提升0.001；整體及每類Pose AP下降不超過0.003；每類Box AP下降不超過0.005。這些是工程容許值，不是統計顯著門檻。',
              '不自動替換任何舊權重。各run保留best.pt、last.pt、last-resumable.pt；診斷summary另保存固定conf0.25的漏檢與像素誤差。',
              '困難與解法見工作紀錄。限制：單seed、formal val反覆選模；Control vs Box10可比較loss加權，但不能只以加權後train loss下降宣稱精度提高。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'artifacts/queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_preserved();spec=yaml.safe_load(CONFIG.read_text())
    for config in spec['stages']:
        c=yaml.safe_load((ROOT/config).read_text());run=ROOT/'runs'/c['name']
        if (run/'completed.json').is_file():continue
        if run.exists() and not (run/'weights/last-resumable.pt').is_file():raise RuntimeError('未完成run無checkpoint，停止避免覆寫')
        queue.execute(c['name'],[sys.executable,'-u',str(ROOT/'scripts/train_stage.py'),str(ROOT/config),str(c['batch'])],600)
    queue.execute('box_recovery_comparison',[sys.executable,'-u',str(Path(__file__).resolve()),'--evaluate'],600)
    source=json.loads((OUTPUT/'provenance.json').read_text())['source']
    diagnostics=[('preserved',source)]+[(Path(config).stem,str(ROOT/'runs'/yaml.safe_load((ROOT/config).read_text())['name']/'weights/best.pt')) for config in spec['stages']]
    for name,weight in diagnostics:
        queue.execute('box_recovery_'+name+'_errors',[sys.executable,'-u',str(ROOT/'scripts/analyze_pose_errors.py'),
                      '--weights',weight,'--output',str(OUTPUT/(name+'-errors')),'--sizes','640'],600)
    verify_preserved();queue.status('completed',report=str(OUTPUT/'report.md'))

if __name__=='__main__':
    try:
        if '--evaluate' in sys.argv:evaluate()
        else:main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True,exist_ok=True);queue.status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
