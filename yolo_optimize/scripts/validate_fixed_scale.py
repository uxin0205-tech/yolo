#!/usr/bin/env python3
"""固定係數早退的完整資料集驗證，不產生新訓練權重。"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from yolo_optimize import runtime
from yolo_optimize.fixed_scale import install_fixed_early_return
from yolo_optimize.qk_diagnostic import SITES
from yolo_combine.validation import JointValidator, ValidationSettings
import torch


class FixedSource:
    def __init__(self, source):
        self.source = source
        self.installed = 0

    def __getattr__(self, name):
        return getattr(self.source, name)

    def build_task_models(self, kind):
        models = self.source.build_task_models(kind)
        for graph in (models.detect, models.pose):
            modules = dict(graph.named_modules())
            for path in SITES.values():
                install_fixed_early_return(modules[path])
                self.installed += 1
        return models


def main():
    output = ROOT / 'artifacts/direction1-20260908/fixed-scale-full-validation'
    output.mkdir(exist_ok=False)
    config = runtime.load_config(output.parent)
    detect, pose = runtime.prepare_data(config, output.parent / 'datasets')
    checkpoint = runtime.FINAL_ROOT / 'weights/combined/inference/best_joint.pt'
    source, model, _, _ = runtime.load_model(config, checkpoint, torch.device('cuda:0'))
    source = FixedSource(source)
    validator = JointValidator(source, detect_data_yaml=detect, pose_data_yaml=pose,
        output_root=output / 'validation', settings=ValidationSettings(imgsz=640,
        detect_batch_size=32, pose_batch_size=16, detect_workers=4, pose_workers=8,
        device='cuda:0', plots=False, save_coco_json=False))
    metrics = dict(validator.validate(model, epoch=0, kind='bittrue').metrics)
    import json
    baseline = json.loads((output.parent / 'best-joint-revalidation/summary.json').read_text())['metrics']['bittrue']
    deltas = {key: metrics[key] - value for key, value in baseline.items()
              if key.endswith('map50_95')}
    passed = source.installed == 4 and all(abs(value) <= 1e-12 for value in deltas.values())
    runtime.write_json(output / 'summary.json', {'status': 'passed' if passed else 'failed',
        'metrics': metrics, 'ap_deltas': deltas, 'installed_sites': source.installed,
        'checkpoint_sha256': runtime.sha256(checkpoint), 'training_performed': False,
        'hardware_latency_measured': False})
    print('JOB_DONE' if passed else 'ERROR: AP 不相等', flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
