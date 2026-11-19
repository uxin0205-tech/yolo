"""整合研究仍使用 QueueExecutor／ResearchQueueBackend，與歷史 queue 分開保存。"""

import gc
import json
import time
from pathlib import Path

import torch

from .queue_model import JobKind, QueueJob, QueueResult, QueueState
from .queue_policy import SelectionDecision
from .queue_store import QueueStore
from .stage3_checkpoint import ARTIFACTS, PROJECT, ROOT, build, load, review, save, sha256
from .stage3_diagnostics import CANDIDATES, real_data_diagnostics, write_json

STAGE = 'integrated-stage3'


def verify_source(output, create=False):
    paths = [*sorted((PROJECT/'src/yolo_attention').glob('*.py')),
             *sorted((ROOT/'yolo_binaryqk/binaryqk').glob('*.py')),
             ARTIFACTS/'reference-review.json', ARTIFACTS/'cpu-reference.json']
    current = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    target = Path(output)/'source.json'
    if create:
        if target.exists():
            raise FileExistsError(target)
        write_json(target, current)
    elif json.loads(target.read_text()) != current:
        raise ValueError('Stage 3 source/reference changed; review before continuing')


def initialize(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ARTIFACTS) or output.exists():
        raise ValueError('use a fresh output directory inside attention artifacts/stage3')
    policy = review()
    if json.loads((ARTIFACTS/'cpu-reference.json').read_text())['passed'] is not True:
        raise ValueError('CPU numeric reference must pass before initialization')
    jobs = [QueueJob(id='s3-diagnostics', run_name='diagnostics', stage=STAGE,
                     kind=JobKind.VALIDATE, order=0, requires_gpu=True,
                     parent_checkpoint=policy['parent'], evaluation_path=str(output/'diagnostics'))]
    for i, name in enumerate(CANDIDATES, 1):
        jobs.append(QueueJob(id='s3-'+name.lower(), run_name=name, stage=STAGE,
                             kind=JobKind.VALIDATE, order=i, requires_gpu=True,
                             parent_job_ids=('s3-diagnostics',), parent_checkpoint=policy['parent'],
                             evaluation_path=str(output/name)))
    jobs.append(QueueJob(id='s3-selection', run_name='selection', stage=STAGE, kind=JobKind.SELECT,
                         order=len(jobs), parent_job_ids=tuple(job.id for job in jobs[1:]),
                         evaluation_path=str(output/'selection')))
    QueueStore(output).initialize(QueueState.initial(tuple(jobs), project_root=str(PROJECT)))
    verify_source(output, create=True)
    (output/'README.md').write_text('# 第 3 階段正式評估\n\n由同一 S1 parent 比較 N1～N9／D1～D3；'
                                  'queue.json 為單一排程狀態，monitor.jsonl 每 600 秒監測。'
                                  '完成後停止，不啟動第 4 階段。小型數據可提交，checkpoint 不提交。\n')
    (output/'generated/README.md').write_text('# Queue 產生設定的保留入口\n\n本輪配置固定於 manifest；不修改歷史 queue。\n')
    return output


def emit(output, payload):
    if (output/'STOP').exists():
        raise InterruptedError('Stage 3 STOP requested')
    data = {'time_unix': time.time(), **payload}
    write_json(output/'progress.json', data)
    with (output/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(data, ensure_ascii=False)+'\n')


def execute(job, state):
    from binaryqk.data import dataset
    from binaryqk.evaluate import evaluate
    from binaryqk.run import require_idle_gpu, seed
    folder = Path(job.evaluation_path)
    output = folder.parent
    verify_source(output)
    policy = review()
    callback = lambda data: emit(output, {'candidate': job.run_name, **data})
    callback({'kind': 'starting-job'})
    if job.run_name == 'selection':
        from .stage3_report import summarize
        decision = summarize(output)
        callback({'kind': 'selection-complete', **decision})
        return SelectionDecision(winners=('s3-'+decision['selected'].lower(),), skipped=(),
                                 reason=decision['status'])
    require_idle_gpu()
    seed(1)
    folder.mkdir(exist_ok=False)
    (folder/'README.md').write_text(f'# {job.run_name}\n\n第 3 階段共同 S1 parent 的診斷或完整評估；'
                                   '狀態及指標見 JSON，不把診斷排除當成 AP=0。checkpoint 不提交。\n')
    if job.run_name == 'diagnostics':
        real_data_diagnostics(folder, callback)
        torch.cuda.empty_cache()
        return QueueResult(checkpoint_path=policy['parent'], metrics_path=str(folder/'diagnostics.json'))
    gate = json.loads((output/'diagnostics/diagnostics.json').read_text())['gates'][job.run_name]
    if not gate['evaluate_full']:
        result = {'candidate': job.run_name, 'formal': False, 'AP50_95': None,
                  'reason': '超過事前 PV relative L2 診斷門檻，未做完整 AP', **gate}
        write_json(folder/'result.json', result)
        return QueueResult(checkpoint_path=policy['parent'], metrics_path=str(folder/'result.json'))
    model = build(job.run_name)
    checkpoint = folder/'evaluated.pt'
    manifest = save(checkpoint, model)
    # 評估實際重新載入的部署 checkpoint，不只驗證記憶體中剛改的模型。
    del model
    model, reloaded = load(checkpoint)
    if manifest != reloaded:
        raise ValueError('materialized variant did not reload identically')
    model.to('cuda:0').eval()
    ds, data, _ = dataset('val', batch=8)
    torch.cuda.reset_peak_memory_stats()
    metrics = evaluate(model, ds, data, folder, batch=8, progress=callback)
    write_json(folder/'metrics.json', metrics)
    official = metrics['official']
    result = {'candidate': job.run_name, 'formal': True, 'AP50_95': official['AP50_95'],
              'AP50': official['AP50'], 'internal_AP50_95': metrics['internal']['metrics/mAP50-95(B)'],
              'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint),
              'checkpoint_bytes': checkpoint.stat().st_size, 'parameters': sum(p.numel() for p in model.parameters()),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'reason': '完整 COCO val5000；無加訓', 'normalization': manifest['normalization']}
    for name, label in (('person', 'person'), ('ball', 'sports_ball'), ('bat', 'baseball_bat')):
        for metric in ('AP50', 'AP50_95'):
            result[name+'_'+metric] = official['classes'][label][metric]
    integer = manifest['normalization'].get('reciprocal')
    result['reciprocal_table_bytes'] = (integer['table_storage_bits']+7)//8 if integer else None
    profile = {'type': 'analytical-reference-not-hardware-timing', 'input': [640, 640],
               'sites': 2, 'heads_per_site': 4, 'tokens_per_site': 400,
               'rows_per_image': 3200, 'weights_per_image': 1280000,
               'row_max_comparisons': 1276800, 'denominator_additions': 1276800,
               'exp_interpolation_multiplications': 1280000 if job.run_name in {'N1', 'N2', 'N3', 'D1', 'D2', 'D3'} else None,
               'reciprocal_table_bytes': result['reciprocal_table_bytes'],
               'reciprocal_per_row': {'D1': 'one LUT, leading-one and shifts',
                                      'D2': 'two endpoint reads, subtraction, one multiply, shifts/add',
                                      'D3': 'one LUT, two Newton multiplies, subtraction, shifts'}.get(job.run_name),
               'probability_scaling_multiplications': 1280000 if integer else None,
               'exp_table_bytes': 42 if integer or job.run_name == 'N3' else None,
               'qkv_pv_projection': 'unchanged float; not included in normalization proxy'}
    write_json(folder/'profile.json', profile)
    write_json(folder/'result.json', result)
    callback({'kind': 'evaluation-complete', 'official_AP50_95': result['AP50_95']})
    del model, ds
    gc.collect(); torch.cuda.empty_cache()
    return QueueResult(map50_95=result['AP50_95'], map50=result['AP50'],
                       checkpoint_path=str(checkpoint), metrics_path=str(folder/'metrics.json'),
                       profile_path=str(folder/'profile.json'))
