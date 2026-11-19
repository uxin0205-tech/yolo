"""沿用 COCO 與 BBAT5 不可變資料；cache 僅寫 runtime View。"""

from pathlib import Path

from binaryqk.data_safety import install_readonly_data_guard
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from .common import PROJECT, ROOT


def pose_dataset(mode, batch=8, *, epochs=5):
    from yolo_combine.data import prepare_bbt5_view
    install_readonly_data_guard(cache_workspace=PROJECT / 'artifacts/datasets')
    view = prepare_bbt5_view(ROOT / 'configs/datasets/bbat5-v1.yaml',
                             PROJECT / 'artifacts/datasets/bbat5-v1/pose', task='pose')
    data = check_det_dataset(str(view.yaml), autodownload=False)
    cfg = get_cfg(overrides={
        'task': 'pose', 'data': str(view.yaml), 'imgsz': 640, 'batch': batch, 'epochs': epochs,
        'workers': 2, 'seed': 1, 'cache': False, 'fraction': 1., 'plots': False,
        'mosaic': 0., 'mixup': 0., 'cutmix': 0., 'copy_paste': 0., 'close_mosaic': 0, 'fliplr': 0.,
        'amp': False, 'conf': .001, 'iou': .7, 'max_det': 300, 'end2end': True,
        'box': 7.5, 'cls': .5, 'dfl': 1.5, 'pose': 12., 'kobj': 1., 'rle': 1.,
    })
    ds = build_yolo_dataset(cfg, data[mode], batch, data, mode=mode, rect=mode == 'val', stride=32)
    expected = 5964 if mode == 'train' else 683
    source = Path(data[mode])
    names = {p.name for p in source.iterdir() if p.is_file()}
    if len(ds) != expected or {Path(p).name for p in ds.im_files} != names:
        raise ValueError('BBAT5 assignment changed by loader')
    return ds, data, cfg
