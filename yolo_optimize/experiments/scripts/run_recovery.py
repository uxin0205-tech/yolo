#!/usr/bin/env python3
"""隔離 final 的第一輪優化入口。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from yolo_optimize.runtime import FINAL_ROOT, load_config, load_model, output_path, prepare_data, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate')
    validate.add_argument('--candidate', choices=('best_joint', 'best_pose', 'j2_best_joint', 'masf_off', 'qk_fp10', 'qk_fp22'), required=True)
    validate.add_argument('--output', type=Path, required=True)
    validate.add_argument('--backend', choices=('both', 'bittrue'), default='both')
    benchmark = sub.add_parser('benchmark')
    benchmark.add_argument('--output', type=Path, required=True)
    benchmark.add_argument('--warmup-macros', type=int, default=2)
    benchmark.add_argument('--timed-macros', type=int, default=5)
    train = sub.add_parser('train')
    train.add_argument('--variant', choices=('native', 'hog', 'rep17', 'masf_off', 'heads'), required=True)
    train.add_argument('--physical-batch', choices=(32, 64, 128), type=int, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--ema-age-mode', choices=('fresh', 'parent'), default='fresh')
    train.add_argument('--validate-live', action='store_true', help='每 epoch 額外記錄 live BitTrue AP，不混入 EMA selector')
    train.add_argument('--adopt-ema-prefix', type=Path,
                       help='明示採用已完成 EMA age 診斷的 continued E1 完整快照；僅限 native/parent/32/live')
    train.add_argument('--base-lr-scale', type=float, choices=(1.0, 0.25), default=1.0)
    train.add_argument('--pause-on-live-regression', action='store_true')
    args = parser.parse_args()
    output = output_path(args.output)
    if args.command != 'validate':
        from yolo_optimize.training import run_benchmark, run_training
        common = dict(config=load_config(output.parent),
            checkpoint=FINAL_ROOT / 'weights/combined/inference/best_joint.pt',
            paired_snapshot=FINAL_ROOT / 'weights/combined/full-resume/best_joint.pt',
            run_dir=output, device='cuda:0',
            callback=lambda event: print(json.dumps(event, ensure_ascii=False, default=str), flush=True))
        if args.command == 'benchmark':
            run_benchmark(**common, warmup_macros=args.warmup_macros, timed_macros=args.timed_macros)
        else:
            run_training(**common, variant=args.variant, physical_batch=args.physical_batch,
                         ema_age_mode=args.ema_age_mode, validate_live=args.validate_live,
                         adopt_ema_prefix_dir=args.adopt_ema_prefix, base_lr_scale=args.base_lr_scale,
                         pause_on_live_regression=args.pause_on_live_regression)
        return
    if output.exists():
        raise FileExistsError(f'輸出已存在，禁止覆寫：{output}')
    output.mkdir(parents=True)
    started = time.time()
    try:
        import torch
        from yolo_combine.metrics import GATE_METRICS, joint_score
        from yolo_combine.validation import JointValidator, ValidationSettings

        config = load_config(output)
        checkpoint_name = 'best_pose' if args.candidate == 'best_pose' else 'best_joint'
        checkpoint = FINAL_ROOT / 'weights/combined/inference' / f'{checkpoint_name}.pt'
        if args.candidate == 'j2_best_joint':
            checkpoint = FINAL_ROOT / 'weights/rollback/j2/inference/best_joint.pt'
        provenance = {'candidate': args.candidate, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'phase': 'preflight'}
        write_json(output / 'status.json', provenance)
        report = config.preflight()
        write_json(output / 'preflight.json', asdict(report))
        if not report.ready:
            raise RuntimeError(f'前置檢查未通過：{report.blockers}')
        detect_yaml, pose_yaml = prepare_data(config, output.parent / 'datasets')
        source, model, factory, loaded = load_model(config, checkpoint, torch.device('cuda:0'))
        if args.candidate == 'masf_off':
            from yolo_optimize.masf_bridge import disable_shared_masf
            provenance['intervention'] = disable_shared_masf(model)
        qk_source = None
        if args.candidate in ('qk_fp10','qk_fp22'):
            from yolo_optimize.qk_diagnostic import ScoreDiagnosticSource
            qk_source = ScoreDiagnosticSource(source, int(args.candidate.removeprefix('qk_fp')))
            source = qk_source
            provenance['intervention'] = qk_source.report()
        write_json(output / 'model.json', {'loaded': asdict(loaded), 'factory': factory.as_dict()})
        validator = JointValidator(source, detect_data_yaml=detect_yaml, pose_data_yaml=pose_yaml,
            output_root=output / 'validation', settings=ValidationSettings(
                imgsz=640, detect_batch_size=32, pose_batch_size=16,
                detect_workers=4, pose_workers=8, device='cuda:0', plots=True, save_coco_json=False))
        metrics = {}
        for kind in (('float', 'bittrue') if args.backend == 'both' else ('bittrue',)):
            write_json(output / 'status.json', {**provenance, 'phase': f'validate-{kind}', 'elapsed_seconds': time.time() - started})
            result = validator.validate(model, epoch=0, kind=kind)
            metrics[kind] = dict(result.metrics)
            if qk_source is not None:
                if any(item['calls'] <= 0 for item in qk_source.installations):
                    raise RuntimeError('QK diagnostic 沒有作用於全部預期 task 圖')
                provenance['intervention'] = qk_source.report()
            del result
            gc.collect()
            torch.cuda.empty_cache()
            write_json(output / 'summary.json', {'provenance': provenance, 'loaded': asdict(loaded), 'metrics': metrics,
                'joint_scores': {key: joint_score(value) for key, value in metrics.items()},
                'gate_metrics': list(GATE_METRICS), 'elapsed_seconds': time.time() - started})
        write_json(output / 'status.json', {**provenance, 'phase': 'complete', 'elapsed_seconds': time.time() - started})
        print(json.dumps({'candidate': args.candidate, 'joint_scores': {k: joint_score(v) for k, v in metrics.items()}}, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(output / 'status.json', {'phase': 'failed', 'error': repr(error), 'elapsed_seconds': time.time() - started})
        raise


if __name__ == '__main__':
    main()
