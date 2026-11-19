"""只讀小型數據繪圖；整合固定點契約、完整 AP 與交付選型。"""

import csv
import json
import math
from pathlib import Path

from .stage3_checkpoint import load, review, sha256
from .stage3_diagnostics import CANDIDATES, write_json


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    data = json.loads((output/'comparison.json').read_text())
    rows = [r for r in data['rows'] if r['formal']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar([r['candidate'] for r in rows], [r['AP50_95'] for r in rows])
    axes[0].axhline(data['parent_AP50_95'], color='black', linestyle='--', label='S1 parent')
    axes[0].set(ylabel='Official COCO AP50-95', title='Same parent, no adaptation training')
    axes[0].legend()
    complete = [r for r in rows if r['candidate'].startswith('D')]
    axes[1].bar([r['candidate'] for r in complete], [r['reciprocal_table_bytes'] for r in complete])
    axes[1].set(ylabel='Packed table bytes (analytical)', title='Reciprocal constants; exp table adds 42 bytes')
    fig.savefig(output/'comparison.svg')
    plt.close(fig)


def select(rows, parent_ap, tolerance=.001):
    evaluated = [r for r in rows if r['formal']]
    if not evaluated or any(not math.isfinite(r['AP50_95']) for r in evaluated):
        raise ValueError('selection requires finite formal metrics')
    deploy = [r for r in evaluated if r['candidate'] in {'D1', 'D2', 'D3'}]
    if len(deploy) != 3:
        raise ValueError('selection requires all three complete PWL candidates')
    best = max(deploy, key=lambda r: (r['AP50_95'], -r['reciprocal_table_bytes']))
    reference = max(evaluated, key=lambda r: r['AP50_95'])
    ready = best['AP50_95'] >= parent_ap-tolerance and best['AP50_95'] >= reference['AP50_95']-tolerance
    return {'selected': best['candidate'], 'best_comparison': reference['candidate'], 'ready': ready,
            'status': 'complete' if ready else 'needs-adaptation-or-other-normalization-extension',
            'rule': 'full-path highest official AP; exact tie smaller reciprocal ROM; must be within .001 of parent and best comparison',
            'parent_delta': best['AP50_95']-parent_ap,
            'comparison_delta': best['AP50_95']-reference['AP50_95'],
            'next_stage_started': False}


def summarize(output):
    output = Path(output)
    policy = review()
    rows = [json.loads((output/name/'result.json').read_text()) for name in CANDIDATES]
    decision = select(rows, policy['parent_official_AP50_95'])
    payload = {'stage': 3, 'rows': rows, 'parent_AP50_95': policy['parent_official_AP50_95'], **decision}
    write_json(output/'comparison.json', payload)
    fields = ['candidate', 'formal', 'AP50', 'AP50_95', 'person_AP50', 'person_AP50_95',
              'ball_AP50', 'ball_AP50_95', 'bat_AP50', 'bat_AP50_95', 'internal_AP50_95',
              'parameters', 'checkpoint_bytes', 'reciprocal_table_bytes', 'reason']
    with (output/'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    chosen = next(r for r in rows if r['candidate'] == decision['selected'])
    _, manifest = load(chosen['checkpoint'])
    write_json(output/'recommendation.json', {**decision, 'checkpoint': chosen['checkpoint'],
               'sha256': sha256(chosen['checkpoint']), 'manifest': manifest})
    if decision['ready']:
        write_json(output/'handoff.json', {'stage': 3, 'checkpoint': chosen['checkpoint'],
                   'sha256': sha256(chosen['checkpoint']), 'manifest': manifest, 'metrics': chosen,
                   'parent': policy['parent'], 'parent_sha256': policy['parent_sha256'],
                   'training': 'none; evaluated materialized deployment path', 'next_stage_started': False})
    plot(output)
    lines = ['# 第 3 階段評估與選型', '',
             '全部候選從同一 S1 Q10 parent 接續；沒有 exact Softmax 正式對照組或 control 加訓。',
             'N1～N9 先做共同 parent 的兩處 score/P/PV 診斷，再依事前門檻評估。D1～D3 固定 BitTrue PWL exp。', '',
             '| 候選 | 官方 AP50–95 | 相對 parent | 評估狀態 |', '| --- | ---: | ---: | --- |']
    for row in rows:
        value = f"{row['AP50_95']:.6f}" if row['formal'] else '未執行'
        delta = f"{row['AP50_95']-policy['parent_official_AP50_95']:+.6f}" if row['formal'] else '—'
        lines.append(f"| {row['candidate']} | {value} | {delta} | {row['reason']} |")
    lines += ['', f"完整路徑候選推薦：{decision['selected']}；選型狀態：{decision['status']}。",
              '採用門檻為官方 AP 與 parent／最佳比較候選的差距各不超過 0.001；完整路徑內選 AP 最高，完全同分才比 ROM。',
              '若門檻未通過，recommendation 只是待處理候選，不產生正式 handoff；先診斷或完成其他 normalization 的整數分母，再考慮少量適應訓練。', '',
              'D1：256-entry reciprocal LUT；D2：16 段 reciprocal PWL；D3：64-entry LUT＋一次 Newton。',
              'reciprocal 使用 Q1.16 mantissa 與獨立 exponent，probability 為 UQ1.24。沒有最後浮點重正規化。',
              'exp 為 Q8.8→UQ1.15，[-10,0]/20，exp(0)=1；最大 1024 tokens 的分母需要 uint26。',
              '三組均保留 S1 的 binary Q/K、Q10 scales、signed16 bias、activation call-sites；載入不允許默默回退。',
              '八項官方 AP、internal AP、檔案大小與解析 ROM 見 comparison.csv；每個 candidate 的 metrics／profile／checkpoint 可追溯。',
              'QKV、P·V、projection、Conv、SiLU 仍為浮點；這是定點 reference，沒有 RTL／HLS 或板端實測加速。',
              '只用一個既有 seed parent，沒有 control 加訓；16 張真實資料診斷不能取代完整 val5000 或代表所有輸入。',
              '第 4 階段未啟動。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return decision
