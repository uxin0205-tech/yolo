"""完整 canonical dataset；不重切或更改 labels。"""

from pathlib import Path

from binaryqk.data import dataset as coco_dataset
from binaryqk.data_safety import install_readonly_data_guard
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from yolo_combine.data import prepare_bbt5_view

from .common import ARTIFACTS, ROOT, review


def dataset(task, mode, batch=16, epochs=20):
    if task == "detect":
        return coco_dataset(mode, batch, epochs=epochs)
    if task != "pose":
        raise ValueError(task)
    workspace = ARTIFACTS / "datasets"
    install_readonly_data_guard(cache_workspace=workspace)
    prepared = prepare_bbt5_view(
        ROOT / "configs/datasets/bbat5-v1.yaml", workspace / "bbat5-v1", task="pose"
    )
    data = check_det_dataset(str(prepared.yaml), autodownload=False)
    cfg = get_cfg(
        overrides={
            "task": "pose",
            "data": str(prepared.yaml),
            "imgsz": 640,
            "batch": batch,
            "epochs": epochs,
            "workers": 2,
            "seed": review()["protocol"]["seed"],
            "cache": False,
            "fraction": 1.0,
            "plots": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 0,
            "fliplr": 0.0,
            "amp": False,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "end2end": True,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "pose": 12.0,
            "kobj": 1.0,
            "rle": 1.0,
        }
    )
    ds = build_yolo_dataset(
        cfg, data[mode], batch, data, mode=mode, rect=mode == "val", stride=32
    )
    expected = 5964 if mode == "train" else 683
    if len(ds) != expected or {Path(p).name for p in ds.im_files} != {
        p.name for p in Path(data[mode]).iterdir() if p.is_file()
    }:
        raise ValueError("BBAT dataset membership changed")
    return ds, data, cfg
