"""融合前的新 Detect／原獨立 Pose baseline 與新共享模型初始 AP。"""
import json
from pathlib import Path
from local_source import HERE, SELECTION, LocalSource, initialize
import torch
from common import prepare_coco
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.factory import FusionModelFactory
from yolo_combine.xnor import XNORExecutionConfig
import yolo_combine.validation as validation
from yolo_combine.validation import JointValidator, ValidationSettings, extract_pose_metrics
from yolo_combine.metrics import GATE_METRICS
from ultralytics.models.yolo.pose import PoseValidator
from validate import InternalValidator
from pwl_contract import verify_pwl


def main():
    output = HERE / 'artifacts/baseline-v1'
    output.mkdir(parents=True, exist_ok=False)
    initialize()
    torch.set_num_threads(4)
    data = prepare_coco()
    registry = Path('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml')
    view = prepare_bbt5_view(registry, HERE / 'artifacts/datasets/bbat5-v1-runtime')
    source = LocalSource()
    built = FusionModelFactory(source, detect_data_yaml=data, pose_data_yaml=view.yaml,
        xnor=XNORExecutionConfig(backend='bool_tiled', token_tile=32)).build(
            pose_head_checkpoint=source.record['pose_checkpoint'], checkpoint_kind='float')
    assert built.report.complete
    validation.DetectionValidator = InternalValidator
    validator = JointValidator(source, detect_data_yaml=data, pose_data_yaml=view.yaml,
        output_root=output / 'shared-initial', settings=ValidationSettings(
            imgsz=640, detect_batch_size=32, pose_batch_size=16, detect_workers=4,
            pose_workers=4, device='0', plots=False, save_coco_json=False))
    reports = {}
    for kind in ('float', 'bittrue'):
        shared = validator.validate(built.model.eval(), epoch=0, kind=kind)
        assert len(InternalValidator.last_instance.dataloader.dataset) == 5000
        verify_pwl(shared.materialized.detect)
        verify_pwl(shared.materialized.pose)
        # 獨立 Pose 保留原 trunk，用相同 canonical validation 重驗，不能用共享後初始 AP 取代。
        original = source.original_pose(kind)
        pose_validator = PoseValidator(save_dir=output / f'original-pose-{kind}', args={
            'task': 'pose', 'data': str(view.yaml), 'imgsz': 640, 'batch': 16, 'workers': 4,
            'device': '0', 'plots': False, 'save_json': False, 'compile': False,
            'rect': True, 'split': 'val', 'mode': 'val', 'half': False})
        pose_validator(model=original)
        assert len(pose_validator.dataloader.dataset) == 683
        pose_metrics = extract_pose_metrics(pose_validator.metrics, names=original.names)
        baseline = {**{k: v for k, v in shared.metrics.items() if k.startswith('coco/')}, **pose_metrics}
        assert all(k in baseline for k in GATE_METRICS)
        if kind == 'bittrue':
            expected = source.record['detect_expected_metrics']
            for key in ('coco/box/map50_95', 'coco/person/box/map50_95'):
                assert abs(shared.metrics[key] - expected[key]) <= 1e-8, (key, shared.metrics[key], expected[key])
        reports[kind] = {'baseline': baseline, 'shared_initial': shared.metrics,
                         'delta': {k: shared.metrics[k] - baseline[k] for k in GATE_METRICS}}
        del shared, original, pose_validator
        InternalValidator.last_instance = None
    with (output / 'summary.json').open('x') as handle:
        json.dump({'status': 'passed', 'source_selection': str(SELECTION),
                   'detect_val_images': 5000, 'pose_val_images': 683,
                   'dataset_version': 'bbat5-v1', 'backends': reports,
                   'detect_initial_full_ap_exact': True, 'training_started': False,
                   'factory': built.report.as_dict()}, handle, ensure_ascii=False, indent=2)
    with (output / 'independent-baseline.json').open('x') as handle:
        json.dump({'metrics': {k: reports['bittrue']['baseline'][k] for k in GATE_METRICS},
                   'backend': 'bittrue', 'source': str(output / 'summary.json')}, handle, indent=2)
    print('ALL_DONE: 新COCO baseline／原Pose baseline／共享初始值完整重驗通過', flush=True)


if __name__ == '__main__':
    main()
