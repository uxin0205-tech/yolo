"""第 1 階段小型結果／圖表；plot 僅需 JSON，不需 GPU 或 checkpoint。"""

import argparse
import csv
import json
from pathlib import Path

from .basis import BASES
from .model import sha256
from .run import write_json


def metric_row(folder, scale_slots=None):
    result = json.loads((folder / 'result.json').read_text())
    if result['smoke']:
        raise ValueError('smoke is not formal AP')
    best = result['best']
    metrics = next(row['evaluation'] for row in result['history'] if row['epoch'] == best['epoch'])
    official = metrics['official']
    row = {'variant': result['variant'], 'phase': result['phase'], 'epoch': best['epoch'],
           'AP50': official['AP50'], 'AP50_95': official['AP50_95'],
           'internal_AP50_95': metrics['internal']['metrics/mAP50-95(B)']}
    for short, name in (('person', 'person'), ('ball', 'sports_ball'), ('bat', 'baseball_bat')):
        for key in ('AP50', 'AP50_95'):
            row[f'{short}_{key}'] = official['classes'][name][key]
    path = folder / best['checkpoint']
    row.update(checkpoint=str(path), sha256=sha256(path), checkpoint_bytes=path.stat().st_size,
               model_parameters=json.loads((folder / 'recipe.json').read_text())['model_parameters'],
               scale_slots=8 * len(BASES[result['variant']]) if scale_slots is None else scale_slots,
               training_seconds=sum(r.get('training_seconds', 0) for r in result['history']),
               peak_allocated_bytes=max(r.get('peak_allocated_bytes', 0) for r in result['history']))
    return row


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    payload = json.loads((output / 'comparison.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for curve in payload['curves']:
        axis = axes[0 if curve['phase'] == 'screening' else 1]
        axis.plot(curve['epochs'], curve['official_AP50_95'], marker='.', label=curve['variant'])
    for ax, title in zip(axes, ('Equal-budget branch screening', 'Selected branch recovery')):
        ax.set(xlabel='Epoch', ylabel='COCO bbox AP50-95', title=title)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(output / 'curves.svg')
    plt.close(fig)


def summarize(output):
    selection = json.loads((output / 'screening-selection.json').read_text())
    if not selection['passed']:
        raise ValueError('cannot hand off failed screening')
    folders = [output / v for v in BASES] + [output / 'recovery']
    rows = [metric_row(folder) for folder in folders]
    if rows[-1]['variant'] != selection['variant']:
        raise ValueError('recovery changed selected branch')
    curves = []
    for folder in folders:
        result = json.loads((folder / 'result.json').read_text())
        curves.append({'variant': result['variant'], 'phase': result['phase'],
                       'epochs': [r['epoch'] for r in result['history']],
                       'official_AP50_95': [r['evaluation']['official']['AP50_95'] for r in result['history']]})
    write_json(output / 'comparison.json', {'rows': rows, 'curves': curves,
               'branch_selected_by_screening': selection['variant'],
               'status': 'stage1-completed-awaiting-user-review; no-stage2-start',
               'limitations': 'single seed; only screening winner receives full recovery; analytical hardware costs only'})
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final = rows[-1]
    write_json(output / 'handoff.json', {'stage': 1, 'next_stage_started': False,
               'checkpoint': final['checkpoint'], 'sha256': final['sha256'], 'metrics': final,
               'manifest_embedded': True, 'selection': selection,
               'resume': str(output / 'recovery/resume.pt'), 'recipe': str(output / 'recovery/recipe.json')})
    lines = ['# 第 1 階段完成，等待使用者檢視', '',
             '三組各 10 epochs 初篩，依預定規則選一組接全模型恢復；未啟動第 2 階段。', '',
             '| 候選 | 階段 | Best epoch | 官方 AP50–95 | 球 AP50–95 | 棒 AP50–95 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        phase = '共同初篩' if row['phase'] == 'screening' else '選定分支恢復'
        lines.append(f"| {row['variant']} | {phase} | {row['epoch']} | {row['AP50_95']:.6f} | {row['ball_AP50_95']:.6f} | {row['bat_AP50_95']:.6f} |")
    lines += ['', '採用理由：依同預算初篩之官方整體 AP50–95 選分支；同分優先較少 bases。',
              '未採用候選只有初篩結果，不能宣稱其完整恢復後仍較差。恢復保留 epoch 0 作回退。',
              '完整八項 AP、參數契約、檔案大小、時間、顯存、權重 SHA 見 comparison.json 與 handoff.json。',
              'scale slots 為 16／16／24；binary score 成本為兩處 heads×bases×N²×D；residual／Hadamard 成本見 protocol。',
              '圖表由 comparison.json 重畫，不需 checkpoint 或 GPU。仍是單 seed／軟體 reference，沒有 RTL／HLS 或板端加速量測。']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    plot(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', type=Path, required=True)
    plot(parser.parse_args().plot)
