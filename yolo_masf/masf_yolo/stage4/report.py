"""小型數據、免 GPU 重畫及階段結果交接。"""

import csv
import shutil
from pathlib import Path

from .common import PROJECT, ROOT, parent, read, sha256, write_json
from .training import score, vector


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axes, ('detect', 'pose')):
        history = read(output / f'{task}-training/result.json')['history']
        ax.plot([r['epoch'] for r in history], [score(task, r['evaluation']) for r in history], marker='o')
        ax.set(title=task, xlabel='epoch', ylabel='primary AP50-95')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'training-curves.svg')
    plt.close(fig)


def summarize(output):
    output = Path(output)
    results = {}
    for task in ('detect', 'pose'):
        training = output / f'{task}-training'
        study = read(training / 'result.json')
        baseline = read(output / f'{task}-parent/metrics.json')
        on = read(output / f'{task}-verification/metrics.json')
        off = read(output / f'{task}-verification/alpha-off.json')
        # 兩者均列出；沒有額外 control，不把 head 適應全歸功 MASF。
        accepted = study['best']['epoch'] > 0 and score(task, on) > score(task, off)
        destination = PROJECT / f'result/stage4/{task}'
        destination.mkdir(parents=True, exist_ok=False)
        source = training / study['best']['checkpoint']
        candidate = destination / 'candidate.pt'
        shutil.move(source, candidate)
        shutil.move(source.with_suffix('.json'), candidate.with_suffix('.json'))
        source.symlink_to(candidate)
        source.with_suffix('.json').symlink_to(candidate.with_suffix('.json'))
        selected = candidate if accepted else parent(task)[0]
        data = {'task': task, 'adopt_masf_recommendation': accepted, 'checkpoint': str(selected),
                'sha256': sha256(selected), 'candidate': str(candidate), 'candidate_sha256': sha256(candidate),
                'best_epoch': study['best']['epoch'], 'parent': vector(task, baseline),
                'alpha_on': vector(task, on), 'alpha_off': vector(task, off),
                'reason': '對 parent 與同權重 alpha-off 均改善' if accepted else '未同時證實 parent 改善與 MASF 正貢獻，建議維持 parent',
                'resume': str(training / 'resume.pt'), 'resume_sha256': sha256(training / 'resume.pt'),
                'source_run': str(output), 'next_stage_started': False}
        write_json(destination / 'handoff.json', data)
        for filename in ('metrics.json', 'alpha-off.json', 'alpha.json'):
            shutil.copy2(output / f'{task}-verification' / filename, destination / filename)
        (destination / 'README.md').write_text(f'# {task} 第四階段結果\n\n{data["reason"]}。'
                                               '選型、候選、續訓與 SHA 見 handoff.json；下一階段待指定。\n')
        results[task] = data
        rows = [{'epoch': r['epoch'], **vector(task, r['evaluation'])} for r in study['history']]
        with (output / f'{task}-metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot(output)
    removed = []
    for task in ('detect', 'pose'):
        for p in (output / f'{task}-smoke').glob('*.pt'):
            removed.append({'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size})
            p.unlink()
    write_json(output / 'smoke-cleanup.json', {'removed': removed})
    decision = {'ready': True, 'stage': 4, 'tasks': results, 'next_stage_started': False}
    write_json(output / 'comparison.json', decision)
    text = '# 第四階段 MASF 結果\n\n'
    text += '| 任務 | Parent AP | Alpha-on AP | Alpha-off AP | 建議 |\n|---|---:|---:|---:|---|\n'
    for task, d in results.items():
        key = 'AP50_95' if task == 'detect' else 'pose_AP50_95'
        text += f'| {task} | {d["parent"][key]:.9f} | {d["alpha_on"][key]:.9f} | {d["alpha_off"][key]:.9f} | {d["reason"]} |\n'
    text += ('\nDetect 為 COCO val5000 官方 bbox AP；Pose 為 BBAT5 val683 原生 keypoint AP，不能混比。'
             '各輪完整指標見 task-metrics.csv，圖可由 CLI plot 在無 GPU、無 checkpoint 下重畫。'
             '單 seed、無 control 加訓；alpha-on/off 為同 checkpoint 診斷。'
             'CPU 等價與折疊證據在 cpu-diagnostics.json；沒有硬體實測加速結論。'
             '第四階段完成，停止，未啟動第五階段。\n')
    (output / 'RESULTS.md').write_text(text)
    with (ROOT / 'docs/worklogs/2026-10-19-masf-stage4.md').open('a') as stream:
        stream.write('\n\n## 背景流程完成\n\n' + text.split('\n\n', 1)[1])
    with (PROJECT / 'README.md').open('a') as stream:
        stream.write(f'\n\n第四階段背景流程已完成：[結果]({output.relative_to(PROJECT)}/RESULTS.md)。下一階段待指定。\n')
    return decision
