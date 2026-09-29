#!/usr/bin/env python3
"""固定一個 native epoch、共享 live trajectory 的 EMA age 配對診斷。

兩個 EMA 不回饋訓練；分別驗證 live/fresh/continued，沒有自動升格 BEST。
不重設 criterion、不改 LR/BN scope，不加入 HOG，不覆寫舊 run。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from yolo_optimize import runtime
from yolo_optimize.ema_diagnostic import PairedEMAObserver, parent_ema_updates
from yolo_optimize.training import (
    EventSink, TimedEngine, _build, _cleanup, _new_run, _reseed, _state_digest,
    severe_regressions,
)
import torch
from yolo_combine.experiment_log import ExperimentLogger
from yolo_combine.joint_trainer import JointEpochRunner
from yolo_combine.metrics import GATE_METRICS, joint_score
from yolo_combine.resume import TrainingProgress, save_inference_weights, save_training_snapshot
from yolo_combine.validation import JointValidator, ValidationSettings


def _save_boundary(state, observers, root, metadata, report, seeds):
    """先保存相同 epoch-end RNG 邊界，再建立 validation graphs。"""
    saved = {}
    progress = TrainingProgress(stage='ema_age_diagnostic', next_epoch=1,
        global_macro_step=report.next_global_macro_step, joint_epochs_completed=1)
    for label, ema in [('fresh', observers.fresh), ('continued', observers.continued)]:
        result = save_training_snapshot(root / 'checkpoints' / f'{label}-epoch-0001.pt',
            model=state.model, ema=ema, optimizer=state.optimizer,
            scheduler=state.scheduler, scaler=state.scaler, criteria=state.router,
            progress=progress, resolved_config={**metadata, 'ema_variant': label},
            provenance={'parent': metadata['parent'], 'shared_live_trajectory': True,
                        'ema_variant': label, 'diagnostic_only': True},
            loader_state={'snapshot_boundary': 'epoch_end_before_validation', 'seeds': seeds},
            best_state={'automatic_promotion': False})
        saved[label] = asdict(result)
    return saved


def _validate_version(state, model, label, config, data, root, parent, sink):
    validator = JointValidator(state.source, detect_data_yaml=data[0], pose_data_yaml=data[1],
        output_root=root / 'validation' / label, settings=ValidationSettings(
            imgsz=config.imgsz, detect_batch_size=config.detect_val_batch_size,
            pose_batch_size=config.pose_val_batch_size, detect_workers=config.detect_workers,
            pose_workers=config.pose_workers, device=str(state.device),
            plots=config.validation_plots, save_coco_json=config.save_coco_json))
    metrics = {}
    for backend in ('float', 'bittrue'):
        sink({'kind': 'validation_start', 'epoch': 1, 'version': label, 'backend': backend})
        result = validator.validate(model, epoch=1, kind=backend)
        metrics[backend] = dict(result.metrics)
        if any(key not in metrics[backend] for key in GATE_METRICS):
            raise RuntimeError(f'{label}/{backend} 缺少必要 AP')
        sink({'kind': 'validation', 'epoch': 1, 'version': label,
              'backend': backend, 'metrics': metrics[backend]})
        del result
        _cleanup()
    result = {'metrics': metrics,
        'joint_scores': {backend: joint_score(values) for backend, values in metrics.items()},
        'deltas_to_parent': {backend: {key: metrics[backend][key] - parent[backend][key]
            for key in GATE_METRICS} for backend in metrics},
        'regressions_gt_0_005': {backend: severe_regressions(values, parent[backend])
            for backend, values in metrics.items()}}
    runtime.write_json(root / f'{label}-validation.json', result)
    return result


def run(output, physical_batch=32):
    if physical_batch != 32:
        raise ValueError('EMA 單變因對照固定 physical Detect batch 32')
    root = _new_run(output)
    sink = EventSink(root, lambda value: print(json.dumps(value, ensure_ascii=False, default=str), flush=True))
    started = time.monotonic()
    state = observers = None
    summary = {'status': 'preparing', 'epochs': [], 'versions': {},
        'shared_live_trajectory': True, 'independent_replicates': False,
        'automatic_promotion': False, 'hog_enabled': False}
    try:
        config = runtime.load_config(root.parent)
        preflight = config.preflight()
        if not preflight.ready or preflight.baseline is None:
            raise RuntimeError(f'原始 baseline preflight 未通過：{preflight.blockers}')
        checkpoint = runtime.FINAL_ROOT / 'weights/combined/inference/best_joint.pt'
        paired_path = runtime.FINAL_ROOT / 'weights/combined/full-resume/best_joint.pt'
        parent_validation = json.loads((root.parent / 'best-joint-revalidation/summary.json').read_text())
        checkpoint_sha = runtime.sha256(checkpoint)
        if parent_validation['provenance']['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('原 BEST 與重驗紀錄的 SHA 不一致，禁止換 parent')
        paired = torch.load(paired_path, map_location='cpu', weights_only=True, mmap=True)
        updates = parent_ema_updates(paired)
        del paired
        data = runtime.prepare_data(config, root.parent / 'datasets')
        state = _build(config, checkpoint, paired_path, data, 32, 'native', 'cuda:0')
        state.training_mode()
        live_names = tuple(name for name, _ in state.model.named_parameters())
        observers = PairedEMAObserver(state.model, updates)
        state.ema = observers.fresh
        state.engine.ema = observers
        if tuple(name for name, _ in state.model.named_parameters()) != live_names:
            raise AssertionError('EMA observer 不可註冊進 live parameter tree')
        initial_live_digest = _state_digest(state.model)
        if any(_state_digest(ema.ema) != initial_live_digest
               for ema in (observers.fresh, observers.continued)):
            raise AssertionError('兩個 EMA 必須從同一完整 live state 起點複製')
        metadata = {**state.metadata, 'experiment': 'paired_ema_age', 'max_epochs': 1,
            'parent_sha256': checkpoint_sha, 'parent_ema_updates': updates,
            'ema_observers_at_start': observers.metadata(), 'shared_live_trajectory': True,
            'independent_replicates': False, 'evaluation_versions': ['live', 'fresh', 'continued'],
            'only_variable': 'ema_updates_at_start', 'mid_epoch_metric_stopping': False,
            'calibration': None, 'data': [str(path) for path in data]}
        runtime.write_json(root / 'resolved-config.json', metadata)
        summary.update(status='training', metadata=metadata)
        runtime.write_json(root / 'summary.json', summary)
        seeds = _reseed(state, 0)
        timed = TimedEngine(state, sink, mu=0.0)
        with ExperimentLogger(root / 'logs', tensorboard='off') as logger:
            runner = JointEpochRunner(engine=timed, detect_loader=state.detect.loader,
                pose_loader=state.pose.loader, scheduler=state.scheduler, logger=logger,
                apply_training_mode=state.training_mode,
                assert_hardware_contract=lambda: state.guard.assert_unchanged(state.model.base),
                detect_batches_per_macro=8,
                gradient_statistics_interval=config.gradient_statistics_interval)
            sink({'kind': 'epoch_start', 'epoch': 1, 'seeds': seeds,
                  'ema_initial_updates': {'fresh': 0, 'continued': updates}})
            report = runner.run_epoch(epoch=1, global_macro_step=0, stage='ema_age_diagnostic')
        count = report.next_global_macro_step
        if count != state.macros or observers.fresh.updates != count or observers.continued.updates != updates + count:
            raise AssertionError('成功 macro 與兩個 EMA 更新計數不一致')
        live = state.model.state_dict()
        for ema in (observers.fresh, observers.continued):
            state.guard.assert_unchanged(ema.ema.base)
            averaged = ema.ema.state_dict()
            if any(not torch.equal(averaged[name], live[name]) for name in ema.fixed_state_names):
                raise AssertionError('固定 state 在 EMA 內不再 bit-exact')
        snapshots = _save_boundary(state, observers, root, metadata, report, seeds)
        summary.update(status='validating', epochs=[{'epoch': 1, 'training': asdict(report)}],
                       snapshots=snapshots, ema_observers_at_end=observers.metadata())
        runtime.write_json(root / 'summary.json', summary)
        sink({'kind': 'epoch_trained', 'epoch': 1, 'training': asdict(report)})
        versions = [('live', state.model.base, None),
                    ('fresh', observers.fresh.ema.base, observers.fresh),
                    ('continued', observers.continued.ema.base, observers.continued)]
        for label, model, ema in versions:
            digest = _state_digest(model)
            # 先留存推論 state：validation 即使失敗也不必重訓。
            saved = save_inference_weights(root / 'inference' / f'{label}-epoch-0001.pt',
                model=state.model.base,
                ema=SimpleNamespace(ema=ema.ema.base) if ema is not None else None,
                use_ema=ema is not None,
                metadata={'epoch': 1, 'diagnostic_only': True, 'version': label,
                          'parent_sha256': checkpoint_sha, 'shared_live_trajectory': True,
                          'metrics_file': f'{label}-validation.json'})
            result = _validate_version(state, model, label, config, data, root,
                parent_validation['metrics'], sink)
            if _state_digest(model) != digest:
                raise AssertionError(f'{label} validation 修改了來源 state')
            result.update(inference_path=str(saved), state_sha256=digest,
                          source='live' if ema is None else 'ema')
            summary['versions'][label] = result
            runtime.write_json(root / 'summary.json', summary)
        summary.update(status='complete', elapsed_seconds=time.monotonic() - started,
            comparison={backend: {
                'continued_minus_fresh_joint': summary['versions']['continued']['joint_scores'][backend]
                    - summary['versions']['fresh']['joint_scores'][backend],
                'continued_minus_parent_joint': summary['versions']['continued']['joint_scores'][backend]
                    - joint_score(parent_validation['metrics'][backend]),
            } for backend in ('float', 'bittrue')},
            interpretation_limit='EMA age 只改同一 trajectory 的平滑結果；不代表 live 學習改善或獨立 seed 重複。')
        runtime.write_json(root / 'summary.json', summary)
        sink({'kind': 'complete', 'epochs_completed': 1, 'comparison': summary['comparison']})
        return summary
    except BaseException as error:
        summary.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
        runtime.write_json(root / 'summary.json', summary)
        sink({'kind': 'failed', 'error': repr(error)})
        raise
    finally:
        state = observers = None
        _cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-batch', type=int, choices=(32,), default=32)
    args = parser.parse_args()
    run(args.output, args.physical_batch)


if __name__ == '__main__':
    main()
