"""Stage 3 KD：既有 QueueExecutor 執行校準、smoke、五輪、reload AP、選型。"""

import csv
import gc
import json
import shutil
from pathlib import Path

import torch

from .queue_model import JobKind, QueueJob, QueueResult, QueueState
from .queue_policy import SelectionDecision
from .queue_store import QueueStore
from .stage3_checkpoint import ARTIFACTS, PROJECT, ROOT, load, sha256
from .stage3_diagnostics import write_json
from .stage3_kd_training import REVIEW, KDStudy, calibrate, eligible, metric_vector, verify_reference
from .stage3_workflow import emit
from .stage3_workflow import verify_source as verify_code

STAGE = 'integrated-stage3-kd'


def verify_source(output, create=False):
    verify_code(output, create=create)
    target = Path(output)/'kd-reference.json'
    if create:
        write_json(target, {'sha256': sha256(REVIEW)})
    elif json.loads(target.read_text())['sha256'] != sha256(REVIEW):
        raise ValueError('KD reviewed protocol changed')
    verify_reference()


def initialize(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ARTIFACTS):
        raise ValueError('KD needs a fresh directory within artifacts/stage3')
    data = verify_reference()
    steps = [('calibration', JobKind.VALIDATE), ('smoke', JobKind.VALIDATE),
             ('training', JobKind.TRAIN), ('verification', JobKind.EVALUATE), ('selection', JobKind.SELECT)]
    jobs = tuple(QueueJob(id='kd-'+name, run_name=name, stage=STAGE, kind=kind, order=i,
                         parent_job_ids=() if i == 0 else ('kd-'+steps[i-1][0],),
                         parent_checkpoint=data['parent'], requires_gpu=kind is not JobKind.SELECT,
                         evaluation_path=str(output/name)) for i, (name, kind) in enumerate(steps))
    QueueStore(output).initialize(QueueState.initial(jobs, project_root=str(PROJECT)))
    verify_source(output, create=True)
    (output/'README.md').write_text('# Stage 3 KD 恢復\n\nD1 → train-only 校準 → AMP smoke → 五輪 → reload val5000 → 八項 AP 選型。'
                                  '同一 QueueExecutor，600 秒監測，完成即停。小型數據提交；checkpoint 不提交。\n')
    (output/'generated/README.md').write_text('# 固定設定\n\nKD 設定見 reference、calibration 與 training/recipe.json。\n')


def summarize(output):
    data = verify_reference()
    baseline = json.loads(Path(data['parent_metrics']).read_text())
    result = json.loads((output/'training/result.json').read_text())
    report = json.loads((output/'verification/metrics.json').read_text())
    accepted = result['best']['epoch'] > 0 and eligible(report, baseline)
    best_record = next(r for r in result['history'] if r['epoch'] == result['best']['epoch'])
    current, expected = metric_vector(report), metric_vector(best_record['evaluation'])
    if any(abs(current[k]-expected[k]) > 1e-8 for k in current):
        raise ValueError('reloaded KD checkpoint AP differs from selected EMA')
    selected = Path(data['parent'])
    if accepted:
        source = output/'training'/result['best']['checkpoint']
        destination = PROJECT/'result/stage3/kd'
        destination.mkdir(exist_ok=False)
        selected = destination/'evaluated.pt'
        shutil.move(source, selected)
        shutil.move(source.with_suffix('.json'), selected.with_suffix('.json'))
        source.symlink_to(selected)
        source.with_suffix('.json').symlink_to(selected.with_suffix('.json'))
        shutil.copy2(output/'verification/metrics.json', destination/'metrics.json')
        (destination/'README.md').write_text('# Stage 3 KD 選定模型\n\n從 D1 恢復；部署不含教師，完整契約見 evaluated.json。'
                                            f'續訓保存於 `{output}/training/resume.pt`。\n')
    decision = {'ready': True, 'status': 'KD 通過事前八項 AP 門檻' if accepted else 'KD 未通過採用門檻，維持原 D1',
                'accepted_kd': accepted, 'selected': str(selected), 'sha256': sha256(selected),
                'baseline': metric_vector(baseline), 'verified_best': current,
                'best_epoch': result['best']['epoch'], 'epochs_completed': len(result['history'])-1,
                'next_stage_started': False, 'training_result': str(output/'training/result.json')}
    write_json(output/'comparison.json', decision)
    write_json(PROJECT/'result/stage3/kd-selection.json', decision)
    rows = [{'epoch': r['epoch'], **metric_vector(r['evaluation']),
             'eligible': eligible(r['evaluation'], baseline)} for r in result['history']]
    with (output/'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(output)
    summary = (f"# Stage 3 KD 結果\n\n{decision['status']}。完成 {decision['epochs_completed']} 輪；"
               f"選定 epoch {decision['best_epoch']}，重新載入官方 AP50–95 {current['AP50_95']:.9f}，"
               f"D1 為 {decision['baseline']['AP50_95']:.9f}。實際採用模型：`{selected}`。\n\n"
               '八項 AP 與各輪結果見 comparison.csv，契約與教師限制見 docs/stage3-recovery-kd.md。'
               '單 seed，沒有額外 control 加訓，不能分離 KD 與額外訓練的獨立貢獻。無硬體實測。'
               '第 4 階段未開始。\n')
    (output/'RESULTS.md').write_text(summary)
    log = ROOT/'docs/worklogs/2026-10-19-attention-stage3-kd.md'
    with log.open('a') as stream:
        stream.write('\n\n## 背景流程完成\n\n'+summary.split('\n\n', 1)[1])
    # Smoke updates are disposable; preserve diagnostics and full formal resume.
    removed = []
    for path in sorted((output/'smoke').glob('*.pt')):
        removed.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size})
        path.unlink()
    write_json(output/'smoke-cleanup.json', {'reason': '已完成正式訓練，smoke 更新不作 parent', 'removed': removed})
    return decision


def plot(output):
    """只需小型 JSON；無 GPU 或 checkpoint 也可重畫。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads((output/'training/result.json').read_text())
    rows = [metric_vector(r['evaluation']) for r in result['history']]
    epochs = [r['epoch'] for r in result['history']]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, name in zip(axes.flat, rows[0]):
        ax.plot(epochs, [r[name] for r in rows], marker='o')
        ax.axhline(rows[0][name], color='gray', linestyle='--')
        ax.set(title=name, xlabel='epoch')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'eight-ap.svg')
    plt.close(fig)


def execute(job, state):
    from binaryqk.data import dataset
    from binaryqk.evaluate import evaluate
    from binaryqk.run import require_idle_gpu, seed
    from binaryqk.train import run
    folder = Path(job.evaluation_path)
    output = folder.parent
    verify_source(output)
    data = verify_reference()
    callback = lambda event: emit(output, {'job': job.run_name, **event})
    callback({'kind': 'starting-job'})
    if job.kind is JobKind.SELECT:
        decision = summarize(output)
        callback({'kind': 'selection-complete', **decision})
        return SelectionDecision(winners=('kd-verification',), skipped=(), reason=decision['status'])
    require_idle_gpu()
    seed(1)
    try:
        if job.run_name == 'calibration':
            folder.mkdir(exist_ok=False)
            calibrate(folder, callback)
            result = QueueResult(checkpoint_path=data['parent'], metrics_path=str(folder/'calibration.json'))
        elif job.run_name in {'smoke', 'training'}:
            policy = KDStudy(output/'calibration/calibration.json')
            run('D1-KD', folder, ROOT/'yolo_binaryqk/artifacts/stage1/calibration.json',
                phase='kd_recovery', parent=data['parent'], smoke=job.run_name == 'smoke',
                resume=(folder/'resume.pt').exists(), policy=policy)
            result_data = json.loads((folder/'result.json').read_text())
            best = result_data['best']
            result = QueueResult(map50_95=best['AP50_95'], checkpoint_path=str(folder/best['checkpoint']),
                                 metrics_path=str(folder/'result.json'))
            del policy
        elif job.run_name == 'verification':
            folder.mkdir(exist_ok=False)
            best = json.loads((output/'training/result.json').read_text())['best']
            path = output/'training'/best['checkpoint']
            model, manifest = load(path)
            if any('teacher' in n for n in model.state_dict()) or not manifest['normalization']['complete_fixed_normalization']:
                raise ValueError('KD export contains teacher or loses integer path')
            ds, config, _ = dataset('val', 8)
            metrics = evaluate(model.to('cuda:0').eval(), ds, config, folder, progress=callback)
            write_json(folder/'metrics.json', metrics)
            result = QueueResult(map50_95=metrics['official']['AP50_95'], checkpoint_path=str(path),
                                 metrics_path=str(folder/'metrics.json'))
            del model, ds
        else:
            raise ValueError('unknown KD job')
        (folder/'README.md').write_text(f'# KD {job.run_name}\n\n本輪產物；metrics／recipe／result JSON 保存實際證據。'
                                       'checkpoint 不提交，正式 resume 保留；smoke 權重完成後移除。\n')
        callback({'kind': 'job-complete'})
        return result
    finally:
        gc.collect()
        torch.cuda.empty_cache()
