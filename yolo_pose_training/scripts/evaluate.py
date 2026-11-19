"""所有階段以相同 960 解析度評估；僅比較固定 formal val。"""
import json
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    queue = yaml.safe_load((ROOT / 'configs/queue.yaml').read_text())
    rows = []
    for config in queue['stages']:
        cfg = yaml.safe_load((ROOT / config).read_text())
        weight = ROOT / 'runs' / cfg['name'] / 'weights/best.pt'
        metrics = YOLO(weight).val(data=cfg['data'], split='val', imgsz=960, batch=8, device=0,
                                  workers=8, plots=True, project=str(ROOT/'evaluations'),
                                  name=cfg['name'], exist_ok=True)
        row = {'stage':cfg['name'], 'checkpoint':str(weight),
               'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
               'per_class':{cfgname:{'box_map50_95':float(metrics.box.maps[i]),
                                    'pose_map50_95':float(metrics.pose.maps[i])}
                            for i,cfgname in metrics.names.items()}}
        row['selection_score'] = (float(metrics.box.map)+float(metrics.pose.map))/2
        rows.append(row)
        (ROOT/'artifacts/comparison.json').write_text(json.dumps(rows,indent=2))
    best = max(rows,key=lambda r:r['selection_score'])
    final = ROOT/'artifacts/final'; final.mkdir(exist_ok=True)
    shutil.copy2(best['checkpoint'],final/'best.pt')
    shutil.copy2(ROOT/'runs'/rows[-1]['stage']/'weights/last.pt',final/'last.pt')
    shutil.copy2(ROOT/'runs'/rows[-1]['stage']/'weights/last-resumable.pt',final/'last-resumable.pt')
    (final/'selection.json').write_text(json.dumps(best,indent=2))
    lines=['# 訓練完成結果','', '固定 bbat5-v1 formal val（683 張），統一以 960 評估。沒有獨立 test split；結果為驗證集選模表現。', '',
           '| 階段 | Box mAP50–95 | Pose mAP50–95 | 兩者平均 |','| --- | --- | --- | --- |']
    for r in rows:
        m=r['metrics']; lines.append(f"| {r['stage']} | {m['metrics/mAP50-95(B)']:.5f} | {m['metrics/mAP50-95(P)']:.5f} | {r['selection_score']:.5f} |")
    lines += ['',f"最佳候選：{best['stage']}。以 Box 與 Pose mAP50–95 平均選擇；各類別結果見 comparison.json。",'',
              'best.pt 為上述最佳候選；last.pt 與 last-resumable.pt 為最後 MuSGD 階段的最後權重與完整續訓 checkpoint。', '',
              '困難與解法：啟動期間見工作紀錄；執行警告與錯誤見 logs/。未解風險：沒有獨立測試集，且只比較此條接續訓練路徑，非全面超參數搜尋。']
    (ROOT/'artifacts/final/report.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
