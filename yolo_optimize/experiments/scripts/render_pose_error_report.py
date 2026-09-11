#!/usr/bin/env python3
"""從已完成全量驗證產物生成 HTML／SVG 案例；不重跑推論、不改原影像。"""
import base64
import html
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/direction1-20260909/pose-error-audit'


def error_count(record):
    return sum(row[metric][kind] for row in record['thresholds']['0.25'].values()
               for metric in ('box50', 'pose75') for kind in ('fp', 'fn'))


def pane(record, target=False):
    path = Path(record['image'])
    with Image.open(path) as image:
        width, height = image.size
        mime = Image.MIME[image.format]
    encoded = base64.b64encode(path.read_bytes()).decode()
    content = [f'<svg viewBox="0 0 {width} {height}" role="img">',
        f'<image width="{width}" height="{height}" href="data:{mime};base64,{encoded}"/>']
    data = record['targets' if target else 'predictions']
    for i, (box, cls) in enumerate(zip(data['bboxes'], data['cls'])):
        color = '#00ff99' if int(cls) == 0 else '#ffcc00'
        x1, y1, x2, y2 = box
        content.append(f'<rect x="{x1}" y="{y1}" width="{max(0,x2-x1)}" height="{max(0,y2-y1)}" fill="none" stroke="{color}" stroke-width="2"/>')
        label = ('ball' if int(cls) == 0 else 'bat')
        if not target:
            label += f" {data['conf'][i]:.2f}"
        content.append(f'<text x="{x1}" y="{max(12,y1-3)}" fill="{color}" font-size="12" stroke="black" stroke-width=".25">{label}</text>')
        for kp in data['kpts'][i]:
            # GT visibility is an annotation; prediction confidence is not comparable to it.
            if target and len(kp) > 2 and kp[2] <= 0:
                continue
            content.append(f'<circle cx="{kp[0]}" cy="{kp[1]}" r="3" fill="{color}" stroke="black"/>')
    return ''.join(content) + '</svg>'


def main():
    parent = json.loads((RUN / 'parent.json').read_text())
    candidate = json.loads((RUN / 'native_e5.json').read_text())
    a = {row['image']: row for row in parent['records']}
    b = {row['image']: row for row in candidate['records']}
    assert a.keys() == b.keys() and len(a) == 683
    totals = {}
    for name, data in (('parent', a), ('native_e5', b)):
        totals[name] = {threshold: {cls: {
            key: sum(row['thresholds'][threshold][cls][key] for row in data.values())
            for key in ('gt', 'pred')} for cls in ('0', '1')} for threshold in ('0.001', '0.25')}
        for threshold in totals[name]:
            for cls in totals[name][threshold]:
                totals[name][threshold][cls].update({metric: {kind: sum(row['thresholds'][threshold][cls][metric][kind] for row in data.values())
                    for kind in ('tp', 'fp', 'fn')} for metric in ('box50', 'box75', 'pose50', 'pose75')})
    difference = {path: error_count(b[path])-error_count(a[path]) for path in a}
    groups = [('退化案例', sorted((p for p in a if difference[p] > 0), key=lambda p: (-difference[p], p))),
              ('改善案例', sorted((p for p in a if difference[p] < 0), key=lambda p: (difference[p], p))),
              ('共同失敗', sorted((p for p in a if error_count(a[p]) > 0 and error_count(b[p]) > 0),
                                key=lambda p: (-min(error_count(a[p]),error_count(b[p])), p)))]
    pages = ['<!doctype html><meta charset="utf-8"><title>BBAT5 同圖錯誤對照</title>',
        '<style>body{font-family:sans-serif;background:#171717;color:#eee;margin:24px}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}svg{width:100%;background:#000}section{margin-bottom:32px}pre{white-space:pre-wrap}</style>',
        '<h1>原 BEST／原生 E5：完整驗證後的同圖案例</h1>',
        '<p>固定 conf=0.25。綠色 ball、黃色 bat；GT 隱藏未標可見點，預測顯示所有點，不暗中另調 keypoint threshold。框錯誤用 IoU≥0.5，點位用官方 OKS≥0.75，各自匹配，不能把兩者 TP 相減當成同一物件退化。</p>',
        '<p>案例從完整 683 張 val 依錯誤差排序，各類最多 4 張並去重，只作開發診斷，不是新 split、訓練子集或獨立 test；排序分數含 box/pose 重複事件，不是新增 AP。</p>']
    manifest = []
    used = set()
    for title, paths in groups:
        selected = [p for p in paths if p not in used][:4]
        for path in selected:
            used.add(path)
            manifest.append({'group': title, 'image': path, 'error_delta': difference[path]})
            pages.append(f'<section><h2>{title}：{html.escape(Path(path).name)}</h2><p>錯誤計數變化 {difference[path]:+d}</p><div class="row">')
            for label, record, target in (('GT',a[path],True),('原 BEST',a[path],False),('Native E5 EMA',b[path],False)):
                pages.append(f'<div><h3>{label}</h3>{pane(record,target)}</div>')
            pages.append('</div></section>')
    pages.append('<h2>完整資料集計數</h2><pre>'+html.escape(json.dumps(totals,ensure_ascii=False,indent=2))+'</pre>')
    (RUN / 'comparison.html').write_text('\n'.join(pages),encoding='utf-8')
    (RUN / 'case-manifest.json').write_text(json.dumps({'cases':manifest,'totals':totals},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert len(manifest) == len(used)
    print(f'完成 {len(manifest)} 個同圖案例；683 張全量計數，未改原影像')


if __name__ == '__main__':
    main()
