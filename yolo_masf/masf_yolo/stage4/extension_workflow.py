"""已授權的第四階段延長：先補實際已訓開關，再配對續訓。"""

import gc
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed
from yolo_attention.queue_model import JobKind, QueueJob, QueueResult, QueueState
from yolo_attention.queue_policy import SelectionDecision
from yolo_attention.queue_store import QueueStore

from . import checkpoint
from .common import ARTIFACTS, PROJECT, ROOT, emit, read, write_json
from .extension_training import ContinuationStudy, review
from .training import score, vector
from .workflow import verify_source


def initialize(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ARTIFACTS):
        raise ValueError('extension needs a fresh Stage4 run')
    review()
    if not read(ARTIFACTS / 'extension-cpu.json')['passed']:
        raise ValueError('CPU continuation validation required')
    steps = [(f'{task}-diagnostics', JobKind.EVALUATE) for task in ('detect', 'pose')]
    steps += [(f'{task}-{arm}-smoke', JobKind.VALIDATE) for task in ('detect', 'pose') for arm in ('L1', 'L2')]
    steps += [(f'{task}-{arm}-{action}', kind) for task in ('pose', 'detect') for arm in ('L1', 'L2')
              for action, kind in (('training', JobKind.TRAIN), ('verification', JobKind.EVALUATE))]
    steps += [('extension-selection', JobKind.SELECT)]
    jobs = tuple(QueueJob(id=name.lower(), run_name=name, stage='integrated-stage4-extension', kind=kind, order=i,
                         parent_job_ids=() if i == 0 else (steps[i-1][0].lower(),), requires_gpu=kind is not JobKind.SELECT,
                         evaluation_path=str(output / name)) for i, (name, kind) in enumerate(steps))
    QueueStore(output).initialize(QueueState.initial(jobs, project_root=str(PROJECT)))
    write_json(output / 'extension.json', {'schema': 1, 'stage': 4, 'arms': ['L1', 'L2'],
               'minimum_additional_epochs': 5, 'maximum_additional_epochs': 15, 'next_stage_started': False})
    verify_source(output, create=True)
    (output / 'README.md').write_text('# 第四階段延長與 LR 方法比較\n\n'
        '每個任務從同一 E5 resume 獨立分成 L1、L2；保留 live／EMA／optimizer／scaler。'
        '先完整評估已訓 E5 alpha-on/off，再四個 smoke，Pose 短週期研究先行，接著 Detect。'
        '每組增加 5～15 輪，以 primary AP 的 5 輪 patience 早停；最多累計 20 輪。'
        '完成後只整理第四階段，不啟動 HOG。\n')


def evaluate_pair(task, path, folder, callback, *, expected=None):
    from binaryqk.data import dataset
    from binaryqk.evaluate import evaluate

    from .data import pose_dataset
    from .evaluate import evaluate_pose
    model, _ = checkpoint.load(path)
    make_dataset = dataset if task == 'detect' else pose_dataset
    evaluator = evaluate if task == 'detect' else evaluate_pose
    ds, data, _ = make_dataset('val', 8)
    model.to('cuda:0')
    on = evaluator(model, ds, data, folder / 'alpha-on', progress=callback)
    if expected is not None and any(abs(v - vector(task, expected)[k]) > 1e-8
                                    for k, v in vector(task, on).items()):
        raise ValueError('trained checkpoint reload AP differs from EMA')
    alpha = float(model.model[-1].p3_masf.alpha.detach())
    with torch.no_grad():
        model.model[-1].p3_masf.alpha.zero_()
    off = evaluator(model, ds, data, folder / 'alpha-off', progress=callback)
    write_json(folder / 'metrics.json', on)
    write_json(folder / 'alpha-off.json', off)
    write_json(folder / 'alpha.json', {'alpha': alpha, 'checkpoint': str(path),
               'primary_on_minus_off': score(task, on) - score(task, off),
               'interpretation': '真正已訓權重的開關；不以 alpha=0 的 E0 代替'})
    del model, ds
    return on


def execute(job, state):
    from binaryqk.train import run
    folder = Path(job.evaluation_path)
    output = folder.parent
    verify_source(output)
    data = review()
    callback = lambda event: emit(output, {'job': job.id, **event})
    callback({'kind': 'starting-job'})
    if job.kind is JobKind.SELECT:
        from .extension_report import summarize
        summarize(output)
        callback({'kind': 'completed', 'stage': 4, 'next_stage_started': False})
        return SelectionDecision(winners=tuple(f'{t}-{a}-verification' for t in ('detect', 'pose') for a in ('l1', 'l2')),
                                 skipped=(), reason='第四階段延長完成，停止供使用者覆核')
    require_idle_gpu()
    seed(6)
    task, *rest = job.run_name.split('-')
    try:
        if rest == ['diagnostics']:
            folder.mkdir(exist_ok=False)
            expected = read(Path(data['original_run']) / f'{task}-training/metrics-05.json')
            source = Path(data['parents'][task]['inference'])
            metrics = evaluate_pair(task, source, folder, callback, expected=expected)
            result = QueueResult(map50_95=score(task, metrics), checkpoint_path=str(source),
                                 metrics_path=str(folder / 'metrics.json'))
        else:
            arm, action = rest
            if action in ('smoke', 'training'):
                policy = ContinuationStudy(task, arm)
                run(f'{task}-{arm}', folder, ROOT / 'yolo_binaryqk/artifacts/stage1/calibration.json',
                    phase='masf-extension', parent=data['parents'][task]['resume'], smoke=action == 'smoke',
                    resume=(folder / 'resume.pt').exists(), policy=policy)
                best = read(folder / 'result.json')['best']
                result = QueueResult(map50_95=best['AP50_95'], checkpoint_path=str(folder / best['checkpoint']),
                                     metrics_path=str(folder / 'result.json'))
            elif action == 'verification':
                folder.mkdir(exist_ok=False)
                training = output / f'{task}-{arm}-training'
                best = read(training / 'result.json')['best']
                path = training / best['checkpoint']
                metrics = evaluate_pair(task, path, folder, callback,
                                        expected=read(training / f'metrics-{best["epoch"]:02d}.json'))
                result = QueueResult(map50_95=score(task, metrics), checkpoint_path=str(path),
                                     metrics_path=str(folder / 'metrics.json'))
            else:
                raise ValueError(action)
        callback({'kind': 'job-complete'})
        return result
    finally:
        gc.collect()
        torch.cuda.empty_cache()
