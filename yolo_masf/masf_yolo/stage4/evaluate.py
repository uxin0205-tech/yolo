"""Pose 使用原生 matching，固定 FP32 E2E，禁止隱式 fusion。"""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.pose.val import PoseValidator


@torch.inference_mode()
def evaluate_pose(model, ds, data, output, batch=8, device='cuda:0', limit_batches=None, progress=None):
    cfg = get_cfg(overrides={'task': 'pose', 'imgsz': 640, 'batch': batch, 'workers': 2, 'plots': False,
                             'conf': .001, 'iou': .7, 'max_det': 300, 'end2end': True, 'save_json': False})
    val = PoseValidator(args=vars(cfg), save_dir=Path(output))
    val.training = False
    val.device, val.data = torch.device(device), data
    val.iouv = val.iouv.to(device)
    model.eval()
    if not model.end2end:
        raise ValueError('Pose evaluator requires E2E')
    val.init_metrics(model)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, collate_fn=ds.collate_fn,
                        pin_memory=device.startswith('cuda'))
    started, paths = time.time(), []
    for index, raw in enumerate(loader):
        item = val.preprocess(raw)
        val.update_metrics(val.postprocess(model(item['img'])), item)
        paths.extend(raw['im_file'])
        if progress and (index + 1) % 25 == 0:
            progress({'kind': 'evaluation', 'images': len(paths), 'seconds': time.time() - started})
        if limit_batches is not None and index + 1 >= limit_batches:
            break
    stats = {k: float(v) for k, v in val.get_stats().items()}
    if not all(math.isfinite(v) for v in stats.values()):
        raise ValueError('nonfinite Pose metrics')
    if limit_batches is None and (len(paths) != 683 or len(set(paths)) != 683):
        raise ValueError('Pose evaluation must cover val683 exactly once')
    vector = {'box_AP50': float(val.metrics.box.map50), 'box_AP50_95': float(val.metrics.box.map),
              'pose_AP50': float(val.metrics.pose.map50), 'pose_AP50_95': float(val.metrics.pose.map)}
    for kind in ('box', 'pose'):
        metric = getattr(val.metrics, kind)
        indices = list(metric.ap_class_index)
        for cls, name in ((0, 'ball'), (1, 'bat')):
            if cls in indices:
                i = indices.index(cls)
                vector.update({f'{name}_{kind}_AP50': float(metric.ap50[i]),
                               f'{name}_{kind}_AP50_95': float(metric.ap[i])})
    return {'protocol': 'native Pose matching; FP32 E2E conf=.001 iou=.7 max_det=300; no fusion',
            'images': len(paths), 'formal': limit_batches is None, 'internal': stats,
            'metrics': vector, 'seconds': time.time() - started}
