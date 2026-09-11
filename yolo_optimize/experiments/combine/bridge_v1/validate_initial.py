"""完整新 baseline 與同一 checkpoint alpha 開／關；不替換正式 Pose 比較基準。"""
import json
from safe_source import HERE, COMBINE, BridgeSource, initialize
from adapted_source import AdaptedBridgeSource
import torch
from common import prepare_coco
from yolo_combine.factory import FusionModelFactory
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import JointValidator, ValidationSettings
from yolo_combine.metrics import GATE_METRICS
import yolo_combine.validation as validation
from validate import InternalValidator


def main():
    initialize()
    torch.set_num_threads(4)
    assert json.loads((HERE / 'artifacts/preflight-v1.json').read_text())['status'] == 'passed'
    out = HERE / 'artifacts/initial-v1'
    out.mkdir(parents=True, exist_ok=False)
    source = AdaptedBridgeSource()
    coco = prepare_coco()
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',
                            HERE / 'artifacts/datasets/bbat5-v1-runtime')
    built = FusionModelFactory(source, detect_data_yaml=coco, pose_data_yaml=view.yaml).build(
        pose_head_checkpoint=source.record['pose_checkpoint'], checkpoint_kind='float')
    assert built.report.complete
    validation.DetectionValidator = InternalValidator
    validator = JointValidator(source, detect_data_yaml=coco, pose_data_yaml=view.yaml,
        output_root=out / 'validation', settings=ValidationSettings(imgsz=640, detect_batch_size=32,
            pose_batch_size=16, detect_workers=4, pose_workers=4, device='0', plots=False, save_coco_json=False))
    results = {}
    for name, kind in [('alpha-on-float', 'float'), ('alpha-on-bittrue', 'bittrue'), ('alpha-zero-bittrue', 'bittrue')]:
        if name.startswith('alpha-zero'):
            with torch.no_grad():
                built.model.detect_head.p3_masf.alpha.zero_()
        result = validator.validate(built.model.eval(), epoch=len(results), kind=kind)
        assert len(InternalValidator.last_instance.dataloader.dataset) == 5000
        results[name] = dict(result.metrics)
        del result
        InternalValidator.last_instance = None
    for k, expected in source.record['detect_expected_metrics'].items():
        assert abs(results['alpha-on-bittrue'][k] - expected) <= 1e-8
    for k in GATE_METRICS:
        if k.startswith('bbat/'):
            assert abs(results['alpha-on-bittrue'][k] - results['alpha-zero-bittrue'][k]) <= 1e-8
    old = json.loads((COMBINE / 'artifacts/baseline-v1/summary.json').read_text())
    baseline = {k: (results['alpha-on-bittrue'][k] if k.startswith('coco/')
                  else old['backends']['bittrue']['baseline'][k]) for k in GATE_METRICS}
    with (out / 'baseline.json').open('x') as handle:
        json.dump({'metrics': baseline, 'source': source.provenance('bittrue'),
            'pose_baseline_reused': str(COMBINE / 'artifacts/baseline-v1/summary.json')}, handle, indent=2)
    with (out / 'summary.json').open('x') as handle:
        json.dump({'status': 'passed', 'results': results, 'baseline': baseline,
            'alpha_zero_delta': {k: results['alpha-zero-bittrue'][k]-results['alpha-on-bittrue'][k] for k in GATE_METRICS},
            'pose_initial_is_not_original_pose_baseline': True}, handle, indent=2)
    print('JOB_DONE: Bridge＋已訓練 J0 Pose 完整 baseline／alpha 開關已驗證，準備 J1。', flush=True)


if __name__ == '__main__':
    main()
