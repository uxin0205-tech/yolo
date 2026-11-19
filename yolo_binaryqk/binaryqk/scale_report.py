"""Stage 2 正式比較、整數常數與無 checkpoint 圖表重畫。"""

import argparse
import csv
import json
from pathlib import Path

from .report import metric_row
from .run import write_json
from .scale_checkpoint import CANDIDATES, load
from .scale_study import verify_reference


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    payload = json.loads((output / 'comparison.json').read_text())
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for curve in payload['curves']:
        axis.plot(curve['epochs'], curve['official_AP50_95'], marker='.', label=curve['candidate'])
    axis.axhline(payload['parent_AP50_95'], linestyle='--', color='gray', label='B2 parent')
    axis.set(xlabel='Epoch', ylabel='COCO bbox AP50-95', title='Stage 2: fixed Q10 vs power-of-two scale')
    axis.grid(alpha=.25)
    axis.legend()
    fig.savefig(output / 'curves.svg')
    plt.close(fig)


def summarize(output):
    review = verify_reference()
    rows, curves, manifests = [], [], {}
    for candidate in CANDIDATES:
        folder = output / candidate
        result = json.loads((folder / 'result.json').read_text())
        if result['smoke'] or len(result['history']) != 6 or result['stop_reason'] != 'budget':
            raise ValueError('Stage 2 comparison requires both full 5-epoch candidates')
        row = metric_row(folder, scale_slots=16)
        _, manifest = load(row['checkpoint'])
        if manifest['lineage']['parent_sha256'] != review['parent_sha256']:
            raise ValueError('Stage 2 comparison parent mismatch')
        manifests[candidate] = manifest
        write_json(folder / 'deployment-constants.json', manifest['constants'])
        row['scale_format'] = CANDIDATES[candidate].value
        row['bias_entries'] = 1008
        rows.append(row)
        curves.append({'candidate': candidate, 'epochs': [h['epoch'] for h in result['history']],
                       'official_AP50_95': [h['evaluation']['official']['AP50_95'] for h in result['history']]})
    selected = max(rows, key=lambda row: (row['AP50_95'], row['variant'] == 'S2'))
    write_json(output / 'comparison.json', {'stage': 2, 'rows': rows, 'curves': curves,
               'parent_AP50_95': review['parent_official_AP50_95'], 'selected': selected['variant'],
               'selection_rule': 'highest official AP; exact tie favors S2 shifts',
               'next_stage_started': False})
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'handoff.json', {'stage': 2, 'next_stage_started': False,
               'checkpoint': selected['checkpoint'], 'sha256': selected['sha256'], 'metrics': selected,
               'manifest': manifests[selected['variant']],
               'resume': str(output / selected['variant'] / 'resume.pt'),
               'recipe': str(output / selected['variant'] / 'recipe.json'),
               'parent_AP50_95': review['parent_official_AP50_95']})
    plot(output)
    lines = ['# 第 2 階段完成，等待檢視', '',
             'S1／S2 各從同一 B2 parent 獨立執行 5 epochs；沒有 baseline/control 加訓。', '',
             '| 候選 | Best epoch | 官方 AP50–95 | internal AP50–95 | 相對 B2 差值 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['variant']} | {row['epoch']} | {row['AP50_95']:.6f} | "
                     f"{row['internal_AP50_95']:.6f} | {row['AP50_95'] - review['parent_official_AP50_95']:+.6f} |")
    lines += ['', f"依事前規則推薦 {selected['variant']}，不預設二次冪勝出；第 3 階段未開始。",
              'Best 包含 epoch 0；若 best 為 0，表示該候選的適應訓練沒有改善初始替換結果。', '',
              '兩組均為 16 個 scale、1008 個 signed16 bias entries、最大 H/W32；Q10 score 累加需 signed18，reference 用 int32。',
              '每張640×640：2.56M次尺度計算，S1為整數乘法、S2為固定shift；共同bias表合計2016 bytes，未存完整NxN bias表。',
              '未打包二值reference、QKV/PV/projection及native Softmax仍保留；不宣稱整網定點或實測加速。',
              '完整八項AP、參數、檔案大小、訓練時間及顯存見 comparison.csv；部署整數常數見各候選 deployment-constants.json。',
              '載入必須依manifest重建，不得退回Stage1固定浮點scale；下一階段parent與SHA見 handoff.json。',
              '限制：單seed、沒有未改方法加訓control、bias從零開始；只有共同五輪預算，未做硬體綜合或板端量測。']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', type=Path, required=True)
    plot(parser.parse_args().plot)
