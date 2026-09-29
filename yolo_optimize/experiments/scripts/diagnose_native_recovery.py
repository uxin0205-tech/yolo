#!/usr/bin/env python3
"""以完整 BBAT5 val 分離權重與 head BN 統計的影響；不執行訓練。"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from yolo_optimize.runtime import FINAL_ROOT, load_config, load_model, output_path, prepare_data, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=('trained', 'trained_parent_bn', 'parent_trained_bn', 'pose_params_only', 'neck_params_only'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--live-snapshot', type=Path,
                        help='明示以 native full snapshot 的 live base state 做 trained／trained_parent_bn 對照')
    args = parser.parse_args()
    if args.live_snapshot is not None and args.case not in ('trained', 'trained_parent_bn'):
        parser.error('--live-snapshot 只支援 trained 或 trained_parent_bn')
    output = output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(output.parent)
    import torch
    from ultralytics.models.yolo.pose import PoseValidator
    from yolo_combine.graph_materialize import build_graph_validation_models
    from yolo_combine.validation import extract_pose_metrics

    root = output.parents[1]
    parent = FINAL_ROOT / 'weights/combined/inference/best_joint.pt'
    trained = root / 'native-control/inference/epoch-0001.pt'
    original_metrics = json.loads((root / 'best-joint-revalidation/summary.json').read_text())['metrics']['bittrue']
    parameters_case = args.case in ('pose_params_only', 'neck_params_only')
    selected = parent if args.live_snapshot is not None or args.case == 'parent_trained_bn' or parameters_case else trained
    donor = trained if args.case == 'parent_trained_bn' or parameters_case else parent
    source, model, factory, loaded = load_model(config, selected, torch.device('cuda:0'))
    live_provenance = None
    if args.live_snapshot is not None:
        snapshot = output_path(args.live_snapshot)
        payload = torch.load(snapshot, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('schema_version') != 2 or payload.get('checkpoint_kind') != 'full_resume':
            raise ValueError('live source 必須是 schema2 full-resume')
        if payload['contract'].get('base') != model.contract() or payload['contract'].get('auxiliary') is not None:
            raise ValueError('live snapshot 架構不符或不是 native')
        if Path(payload['resolved_config']['parent']).resolve() != parent.resolve():
            raise ValueError('live snapshot parent 不符')
        if any(not key.startswith('base.') for key in payload['model_state']):
            raise ValueError('live snapshot 有非 base state')
        model.load_state_dict({key.removeprefix('base.'): value for key, value in payload['model_state'].items()}, strict=True)
        live_provenance = {'path': str(snapshot), 'sha256': sha256(snapshot),
                           'source': 'model_state.base_live', 'progress': payload['progress']}
        del payload

    def parameter_hash():
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    before_parameters = parameter_hash() if args.live_snapshot is not None else None
    replaced = []
    replaced_parameters = []
    if parameters_case:
        donor_state = torch.load(donor, map_location='cpu', weights_only=True, mmap=True)['state_dict']
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                index = int(name.split('.')[2])
                is_neck = 11 <= index <= 22 and '.p3_masf.' not in name and '.attn.' not in name
                chosen = '.pose_head.' in name if args.case == 'pose_params_only' else is_neck
                if chosen:
                    parameter.copy_(donor_state[name].to(parameter.device))
                    replaced_parameters.append(name)
        if not replaced_parameters:
            raise RuntimeError('參數介入為空')
        del donor_state
    elif args.case != 'trained':
        donor_state = torch.load(donor, map_location='cpu', weights_only=True, mmap=True)['state_dict']
        state = model.state_dict()
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            if '.detect_head.' not in module_name and '.pose_head.' not in module_name:
                continue
            for suffix in ('running_mean', 'running_var', 'num_batches_tracked'):
                name = f'{module_name}.{suffix}'
                state[name].copy_(donor_state[name].to(state[name].device))
                replaced.append(name)
        if not replaced:
            raise RuntimeError('沒有找到 head BN 統計，禁止空介入')
        del donor_state
    if before_parameters is not None and parameter_hash() != before_parameters:
        raise RuntimeError('BN-only 診斷不得改動任何參數')
    _, pose_yaml = prepare_data(config, root / 'datasets')
    materialized = build_graph_validation_models(model, source, kind='bittrue')
    validator = PoseValidator(save_dir=output / 'pose', args={
        'task': 'pose', 'data': str(pose_yaml), 'imgsz': 640, 'batch': 16,
        'workers': 8, 'device': 'cuda:0', 'plots': False, 'save_json': False,
        'compile': False, 'rect': True, 'split': 'val', 'mode': 'val',
    })
    started = time.time()
    validator(model=materialized.pose)
    metrics = extract_pose_metrics(validator.metrics, names=materialized.pose.names)
    deltas = {key: value - original_metrics[key] for key, value in metrics.items() if key.endswith('/map50_95')}
    failures = {key: delta for key, delta in deltas.items() if delta < -0.005}
    report = {'case': args.case, 'weights_source': str(selected),
        'bn_source': str(donor) if replaced else str(selected), 'replaced_buffers': replaced,
        'parameter_donor': str(donor) if replaced_parameters else None,
        'replaced_parameters': replaced_parameters,
        'loaded': asdict(loaded), 'metrics': metrics, 'ap_deltas_to_parent': deltas,
        'criterion': '每個 BBAT AP 必須 >= parent - 0.005', 'passed': not failures,
        'failed_metrics': failures, 'validation_seconds': time.time() - started,
        'scope': '完整 canonical BBAT5 val，無訓練、無資料或原權重修改'}
    if live_provenance is not None:
        report.update({'weights_source': live_provenance['path'], 'live_snapshot': live_provenance,
                       'parameter_sha256_before_after': before_parameters,
                       'initial_parent_load': report.pop('loaded')})
    write_json(output / 'result.json', report)
    print(json.dumps({key: report[key] for key in ('case', 'ap_deltas_to_parent', 'passed', 'failed_metrics')}, ensure_ascii=False), flush=True)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
