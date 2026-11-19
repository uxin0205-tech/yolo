"""續訓比較保留真正已訓最佳候選，與推薦採用分開。"""

import csv
import shutil
from pathlib import Path

from .common import ARTIFACTS, PROJECT, ROOT, parent, read, sha256, write_json
from .extension_training import review
from .training import score, vector


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task in zip(axes, ('detect', 'pose')):
        for arm in ('L1', 'L2'):
            hist = read(output / f'{task}-{arm}-training/result.json')['history']
            ax.plot([5 + r['epoch'] for r in hist], [score(task, r['evaluation']) for r in hist], marker='.', label=arm)
        baseline = read(Path(read(ARTIFACTS / 'extension-reference.json')['original_run']) / f'{task}-parent/metrics.json')
        ax.axhline(score(task, baseline), color='gray', linestyle='--', label='parent')
        ax.set(title=task, xlabel='cumulative epochs', ylabel='primary AP50-95')
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'extension-curves.svg')
    plt.close(fig)


def summarize(output):
    output = Path(output)
    data = review()
    tasks, rows = {}, []
    for task in ('detect', 'pose'):
        baseline = read(Path(data['original_run']) / f'{task}-parent/metrics.json')
        options = []
        for arm in ('L1', 'L2'):
            training = output / f'{task}-{arm}-training'
            result = read(training / 'result.json')
            on = read(output / f'{task}-{arm}-verification/metrics.json')
            off = read(output / f'{task}-{arm}-verification/alpha-off.json')
            values, base = vector(task, on), vector(task, baseline)
            safe = all(values[k] >= base[k] - .001 for k in base)
            accepted = score(task, on) > score(task, baseline) and safe and score(task, on) > score(task, off)
            destination = PROJECT / f'result/stage4/{task}/{arm}'
            destination.mkdir(exist_ok=False)
            source = training / result['best']['checkpoint']
            candidate = destination / 'candidate.pt'
            shutil.move(source, candidate)
            shutil.move(source.with_suffix('.json'), candidate.with_suffix('.json'))
            source.symlink_to(candidate)
            source.with_suffix('.json').symlink_to(candidate.with_suffix('.json'))
            record = {'arm': arm, 'adopt_recommendation': accepted, 'class_safety': safe,
                      'candidate': str(candidate), 'sha256': sha256(candidate),
                      'best_cumulative_epoch': 5 + result['best']['epoch'],
                      'completed_cumulative_epochs': 5 + len(result['history']) - 1,
                      'stop_reason': result['stop_reason'], 'parent': base,
                      'alpha_on': values, 'alpha_off': vector(task, off),
                      'primary_gain': score(task, on) - score(task, baseline),
                      'masf_on_off_gain': score(task, on) - score(task, off),
                      'resume': str(training / 'resume.pt'), 'resume_sha256': sha256(training / 'resume.pt')}
            write_json(destination / 'comparison.json', record)
            options.append(record)
            rows.extend({'task': task, 'arm': arm, 'cumulative_epoch': 5 + h['epoch'],
                         'primary_AP': score(task, h['evaluation'])} for h in result['history'])
        eligible = [d for d in options if d['adopt_recommendation']]
        selected = max(eligible, key=lambda d: d['primary_gain']) if eligible else None
        path = Path(selected['candidate']) if selected else parent(task)[0]
        handoff = {'task': task, 'checkpoint': str(path), 'sha256': sha256(path),
                   'adopt_masf_recommendation': selected is not None, 'selected_arm': selected['arm'] if selected else None,
                   'options': options, 'source_run': str(output), 'next_stage_started': False,
                   'interpretation': '最高 AP 候選與正式建議分開；使用者可覆核 class trade-off，沒有隱藏未過門檻的權重'}
        write_json(PROJECT / f'result/stage4/{task}/handoff.json', handoff)
        (PROJECT / f'result/stage4/{task}/README.md').write_text(
            f'# {task} 第四階段交接\n\n延長比較已完成。推薦採用：{handoff["selected_arm"] or "原 parent"}。'
            '完整 class AP、alpha-on/off、候選與 resume 見 handoff.json；第五階段待指定。\n')
        tasks[task] = handoff
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(output)
    write_json(output / 'comparison.json', {'ready': True, 'stage': 4, 'tasks': tasks, 'next_stage_started': False})
    lines = ['# 第四階段延長結果', '', '| 任務／方法 | 總輪數 | 最佳 AP | 與 parent 差 | Alpha-on/off 差 | 建議採用 |',
             '|---|---:|---:|---:|---:|---|']
    for task, handoff in tasks.items():
        key = 'AP50_95' if task == 'detect' else 'pose_AP50_95'
        for row in handoff['options']:
            lines.append(f'| {task}/{row["arm"]} | {row["completed_cumulative_epochs"]} | '
                         f'{row["alpha_on"][key]:.9f} | {row["primary_gain"]:+.9f} | '
                         f'{row["masf_on_off_gain"]:+.9f} | {row["adopt_recommendation"]} |')
    lines += ['', ('每個任務的 L1／L2 都從同一 E5 完整訓練狀態出發；早停導致預算可能不同，不能宣稱嚴格等輪數比較。'
              '單 seed、未新增 control、沒有硬體實測。最高 AP 與類別保護建議分開，完整結果和權重均保留供覆核。'
              '第 4 階段完成，未啟動第 5 階段。')]
    report = '\n'.join(lines) + '\n'
    (output / 'RESULTS.md').write_text(report)
    with (ROOT / 'docs/worklogs/2026-10-19-masf-stage4-extension.md').open('a') as stream:
        stream.write('\n\n## 背景流程完成\n\n' + report)
    removed = []
    for directory in output.glob('*-smoke'):
        for path in directory.glob('*.pt'):
            removed.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size})
            path.unlink()
    write_json(output / 'smoke-cleanup.json', {'removed': removed})
