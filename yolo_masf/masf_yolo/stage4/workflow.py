"""以既有 QueueExecutor 完成 parent、smoke、五輪與 alpha-on/off。"""

import gc
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed
from yolo_attention.queue_model import JobKind, QueueJob, QueueResult, QueueState
from yolo_attention.queue_policy import SelectionDecision
from yolo_attention.queue_store import QueueStore
from yolo_attention.stage3_kd_training import metric_vector

from yolo_attention import stage3_checkpoint

from .common import (
    ARTIFACTS,
    KD,
    KD_RUN,
    PROJECT,
    REVIEW,
    ROOT,
    emit,
    parent,
    read,
    sha256,
    verify_reference,
    write_json,
)


def verify_source(output, create=False):
    verify_reference()
    path = Path(output) / 'source.json'
    roots = ['yolo_masf/masf_yolo/stage4', 'yolo_attention/src/yolo_attention', 'yolo_binaryqk/binaryqk',
             'yolo_activation/src/activation_lab', 'yolo_achitechure/achitechure_1/src/achitechure_1',
             'yolo_combine/src/yolo_combine']
    files = [p for r in roots for p in (ROOT / r).rglob('*.py')]
    files += [REVIEW, ARTIFACTS / 'cpu-diagnostics.json']
    if (Path(output) / 'extension.json').exists():
        files += [ARTIFACTS / 'extension-reference.json', ARTIFACTS / 'extension-cpu.json']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    if create:
        if not read(ARTIFACTS / 'cpu-diagnostics.json')['passed']:
            raise ValueError('CPU diagnostics required')
        write_json(path, hashes)
    elif hashes != read(path):
        raise ValueError('Stage4 source/protocol changed after queue initialization')


def initialize(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ARTIFACTS):
        raise ValueError('use fresh artifacts/stage4 run directory')
    steps = [('detect-parent', JobKind.EVALUATE), ('pose-parent', JobKind.EVALUATE),
             ('detect-smoke', JobKind.VALIDATE), ('pose-smoke', JobKind.VALIDATE),
             ('detect-training', JobKind.TRAIN), ('detect-verification', JobKind.EVALUATE),
             ('pose-training', JobKind.TRAIN), ('pose-verification', JobKind.EVALUATE),
             ('selection', JobKind.SELECT)]
    jobs = tuple(QueueJob(id=name, run_name=name, stage='integrated-stage4-masf', kind=kind, order=i,
                         parent_job_ids=() if i == 0 else (steps[i-1][0],),
                         requires_gpu=kind is not JobKind.SELECT, evaluation_path=str(output / name))
                 for i, (name, kind) in enumerate(steps))
    QueueStore(output).initialize(QueueState.initial(jobs, project_root=str(PROJECT)))
    verify_source(output, create=True)
    (output / 'README.md').write_text('# 第四階段 MASF\n\n單一 QueueExecutor；兩個 task-specific P3 分支，各五輪。'
                                    '600 秒監測，完成選型即停止，不啟動 HOG。\n')


def promote_kd(report):
    expected = read(KD_RUN / 'training/metrics-05.json')
    current = metric_vector(report)
    if any(abs(current[k] - metric_vector(expected)[k]) > 1e-8 for k in current):
        raise ValueError('KD epoch5 exported EMA AP differs from training evaluation')
    path, digest = parent('detect')
    export = read(KD / 'export.json')
    export['formal_gpu_verification'] = 'passed: exact eight AP after reload val5000'
    write_json(KD / 'export.json', export)
    write_json(KD / 'metrics.json', report)
    _, manifest = stage3_checkpoint.load(path)
    old = read(KD.parent / 'handoff.json')
    override = {'accepted_kd': True, 'selected_epoch': 5, 'selection_authority': 'user override 2026-10-19',
                'old_automatic_gate_passed': False, 'old_min_gain': .001,
                'reason': '使用者明確接受第五輪小幅提升；不追加訓練',
                'verified_metrics': current, 'previous_checkpoint': old['checkpoint'],
                'previous_sha256': old['sha256'], 'resume': export['source'],
                'resume_sha256': export['source_sha256']}
    new = {**old, 'checkpoint': str(path), 'sha256': digest, 'manifest': manifest,
           'metrics': {**current, 'formal': True, 'checkpoint': str(path)},
           'training': 'KD five epochs, epoch5 EMA; native teacher not exported',
           'user_selection_override': override, 'next_stage_started': True}
    write_json(KD.parent / 'handoff.json', new)
    write_json(KD.parent / 'user-selection.json', override)
    (KD / 'README.md').write_text('# KD 第五輪：第四階段 Detect parent\n\n使用者接受第五輪 EMA。'
                                 f'重新載入完整 COCO val5000，八項 AP 與訓練紀錄一致，AP50–95 {current["AP50_95"]:.9f}。'
                                 'B2／S1／完整 PWL exp＋D1 reciprocal 均保留，部署無教師。'
                                 '原 .001 自動門檻未通過；本次是使用者覆核採用。\n')


class Backend:
    def execute(self, job, state):
        if job.stage == 'integrated-stage4-extension':
            from .extension_workflow import execute
            return execute(job, state)
        from binaryqk.data import dataset
        from binaryqk.evaluate import evaluate
        from binaryqk.train import run

        from . import checkpoint
        from .data import pose_dataset
        from .evaluate import evaluate_pose
        from .training import MASFStudy, score
        output = Path(job.evaluation_path).parent
        folder = Path(job.evaluation_path)
        verify_source(output)
        callback = lambda event: emit(output, {'job': job.id, **event})
        callback({'kind': 'starting-job'})
        if job.kind is JobKind.SELECT:
            from .report import summarize
            summarize(output)
            return SelectionDecision(winners=('detect-verification', 'pose-verification'), skipped=(),
                                     reason='第四階段結果已整理；停止並等待使用者檢視')
        require_idle_gpu()
        seed(1)
        task, action = job.run_name.split('-', 1)
        make_dataset = dataset if task == 'detect' else pose_dataset
        evaluator = evaluate if task == 'detect' else evaluate_pose
        parent_path, _ = parent(task)
        try:
            if action == 'parent':
                folder.mkdir(exist_ok=False)
                if task == 'detect':
                    model, _ = stage3_checkpoint.load(parent_path)
                else:
                    payload = torch.load(parent_path, map_location='cpu', weights_only=False)
                    model = (payload.get('ema') or payload['model']).float().eval()
                ds, data, _ = make_dataset('val', 8)
                metrics = evaluator(model.to('cuda:0'), ds, data, folder, progress=callback)
                write_json(folder / 'metrics.json', metrics)
                if task == 'detect':
                    promote_kd(metrics)
                result = QueueResult(map50_95=score(task, metrics), checkpoint_path=str(parent_path),
                                     metrics_path=str(folder / 'metrics.json'))
                del model, ds
            elif action in {'smoke', 'training'}:
                policy = MASFStudy(task, output / f'{task}-parent/metrics.json')
                run('M1' if task == 'detect' else 'M2', folder,
                    ROOT / 'yolo_binaryqk/artifacts/stage1/calibration.json', phase='masf',
                    parent=parent_path, smoke=action == 'smoke', resume=(folder / 'resume.pt').exists(), policy=policy)
                best = read(folder / 'result.json')['best']
                result = QueueResult(map50_95=best['AP50_95'], checkpoint_path=str(folder / best['checkpoint']),
                                     metrics_path=str(folder / 'result.json'))
            elif action == 'verification':
                folder.mkdir(exist_ok=False)
                training = output / f'{task}-training'
                best = read(training / 'result.json')['best']
                model, _ = checkpoint.load(training / best['checkpoint'])
                ds, data, _ = make_dataset('val', 8)
                metrics = evaluator(model.to('cuda:0'), ds, data, folder / 'alpha-on', progress=callback)
                write_json(folder / 'metrics.json', metrics)
                expected = read(training / f'metrics-{best["epoch"]:02d}.json')
                from .training import vector
                if any(abs(v - vector(task, expected)[k]) > 1e-8 for k, v in vector(task, metrics).items()):
                    raise ValueError('MASF best EMA reload AP mismatch')
                alpha = float(model.model[-1].p3_masf.alpha.detach())
                with torch.no_grad():
                    model.model[-1].p3_masf.alpha.zero_()
                off = evaluator(model, ds, data, folder / 'alpha-off', progress=callback)
                write_json(folder / 'alpha-off.json', off)
                write_json(folder / 'alpha.json', {'selected_epoch': best['epoch'], 'alpha': alpha,
                           'bridge_beta': 1., 'interpretation': '同一 checkpoint 開關診斷，不能取代 control 加訓'})
                result = QueueResult(map50_95=score(task, metrics), checkpoint_path=str(training / best['checkpoint']),
                                     metrics_path=str(folder / 'metrics.json'))
                del model, ds
            else:
                raise ValueError(action)
            callback({'kind': 'job-complete'})
            return result
        finally:
            gc.collect()
            torch.cuda.empty_cache()
