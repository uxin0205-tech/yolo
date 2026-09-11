#!/usr/bin/env python3
"""完整 canonical val 的固定門檻錯誤對照；不以門檻搜尋替代 AP。"""
import sys
import argparse
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from yolo_optimize import runtime
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.validation import PoseValidator, extract_pose_metrics, JointValidator, ValidationSettings
import torch


class ErrorValidator(PoseValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for index, pred in enumerate(preds):
            target = self._prepare_batch(index, batch)
            full_match = self._process_batch(pred, target)
            record = {'image': str(Path(target['im_file']).resolve()),
                'runtime_image': str(target['im_file']), 'thresholds': {},
                'ranking': {'conf': pred['conf'].cpu().tolist(),
                    'pred_cls': pred['cls'].cpu().tolist(),
                    'target_cls': target['cls'].cpu().tolist(),
                    'tp': full_match['tp'].tolist(), 'tp_p': full_match['tp_p'].tolist()}}
            for threshold in (.001, .25):
                keep = pred['conf'] >= threshold
                selected = {key: value[keep] for key, value in pred.items()}
                matched = self._process_batch(selected, target)
                per_class = {}
                for cls in (0, 1):
                    pmask = selected['cls'].cpu().numpy() == cls
                    count = int((target['cls'] == cls).sum())
                    detections = int(pmask.sum())
                    row = {'gt': count, 'pred': detections}
                    for label, key in (('box', 'tp'), ('pose', 'tp_p')):
                        for level, column in (('50', 0), ('75', 5)):
                            tp = int(matched[key][pmask, column].sum())
                            row[f'{label}{level}'] = {'tp': tp, 'fp': detections-tp, 'fn': count-tp}
                    per_class[str(cls)] = row
                record['thresholds'][str(threshold)] = per_class
            # Original image coordinates for later native SVG overlays; no source image edits.
            selected = {key: value[pred['conf'] >= .25] for key, value in pred.items()}
            scaled = self.scale_preds(selected, target)
            gt = self.scale_preds({'bboxes': target['bboxes'], 'keypoints': target['keypoints']}, target)
            record['predictions'] = {key: value.detach().cpu().tolist() for key, value in scaled.items()}
            record['targets'] = {key: value.detach().cpu().tolist() for key, value in gt.items()}
            record['targets']['cls'] = target['cls'].cpu().tolist()
            self.records.append(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/direction1-20260909/pose-error-audit')
    root = parser.parse_args().output.resolve()
    if not root.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('輸出只允許 workspace artifacts')
    root.mkdir(parents=True, exist_ok=False)
    data_root = ROOT / 'artifacts/direction1-20260908'
    config = runtime.load_config(data_root)
    detect, pose = runtime.prepare_data(config, data_root / 'datasets')
    paths = {'parent': runtime.FINAL_ROOT / 'weights/combined/inference/best_joint.pt',
             'native_e5': data_root / 'native-parent-ema-control-adopted/inference/epoch-0005.pt'}
    results = {}
    for label, checkpoint in paths.items():
        source, base, _, _ = runtime.load_model(config, checkpoint, torch.device('cuda:0'))
        models = build_graph_validation_models(base, source, kind='bittrue')
        settings = ValidationSettings(device='cuda:0', plots=False)
        builder = JointValidator(source, detect_data_yaml=detect, pose_data_yaml=pose,
            output_root=root / label, settings=settings)
        validator = ErrorValidator(save_dir=root / label,
            args=builder._args(task='pose', data=Path(pose), batch=16, workers=8))
        validator(model=models.pose)
        assert len(validator.records) == 683
        assert len({r['image'] for r in validator.records}) == 683
        metrics = extract_pose_metrics(validator.metrics, names=models.pose.names)
        runtime.write_json(root / f'{label}.json', {'checkpoint': str(checkpoint),
            'sha256': runtime.sha256(checkpoint), 'metrics': metrics, 'records': validator.records})
        results[label] = metrics
        del validator, models, base
        torch.cuda.empty_cache()
    delta = results['native_e5']['bbat/ball/box/map50_95'] - results['parent']['bbat/ball/box/map50_95']
    runtime.write_json(root / 'summary.json', {'status': 'complete', 'metrics': results,
        'ball_box_delta': delta, 'regression_reproduced': delta < -.005,
        'images_each': 683, 'display_conf': .25, 'diagnostic_low_conf': .001,
        'not_independent_test': True})
    print('JOB_DONE: 完整錯誤對照完成', flush=True)


if __name__ == '__main__':
    main()
