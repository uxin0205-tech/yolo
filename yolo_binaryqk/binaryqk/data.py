"""完整 COCO runtime View；清單逐位相同，快取不寫入來源。"""

from pathlib import Path

import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from .calibration import COCO, SPLITS, verify_splits
from .data_safety import install_readonly_data_guard
from .model import ROOT

RUNTIME = ROOT / "yolo_binaryqk/artifacts/datasets/coco2017-runtime"


def prepare():
    verify_splits()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for split in ("train2017", "val2017"):
        for kind in ("images", "labels"):
            link, target = RUNTIME / kind / split, COCO / kind / split
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != target.resolve():
                    raise ValueError(f"runtime link mismatch: {link}")
            else:
                link.symlink_to(target, target_is_directory=True)
        contents = (COCO / f"{split}.txt").read_bytes()
        if any(not row.startswith(f"./images/{split}/") or ".." in Path(row).parts
               for row in contents.decode().splitlines()):
            raise ValueError("invalid source list path")
        path = RUNTIME / f"{split}.txt"
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("runtime list mismatch")
        if not path.exists():
            path.write_bytes(contents)
    payload = yaml.safe_load((ROOT / "coco2017.yaml").read_text())
    payload = {"path": str(RUNTIME), "train": "train2017.txt", "val": "val2017.txt", "names": payload["names"]}
    rendered = yaml.safe_dump(payload, sort_keys=False)
    path = RUNTIME / "coco2017.yaml"
    if path.exists() and path.read_text() != rendered:
        raise ValueError("runtime YAML mismatch")
    if not path.exists():
        path.write_text(rendered)
    return path


def config(batch=8):
    return get_cfg(overrides={
        "task": "detect", "imgsz": 640, "batch": batch, "workers": 2,
        "epochs": 10, "seed": 0, "cache": False, "fraction": 1.0,
        "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "close_mosaic": 0, "fliplr": 0.5, "amp": False, "plots": False,
        "conf": 0.001, "iou": 0.7, "max_det": 300, "end2end": True,
    })


def dataset(mode, batch=8, *, epochs=10):
    install_readonly_data_guard()
    path = prepare()
    data = check_det_dataset(str(path), autodownload=False)
    cfg = config(batch)
    cfg.epochs = epochs
    cfg.data = str(path)
    result = build_yolo_dataset(cfg, data[mode], batch, data, mode=mode, rect=mode == "val", stride=32)
    split = "train2017" if mode == "train" else "val2017"
    expected = {(RUNTIME / row).as_posix() for row in (RUNTIME / f"{split}.txt").read_text().splitlines()}
    if len(result) != SPLITS[f"{split}.txt"][0] or {str(Path(p)) for p in result.im_files} != expected:
        raise ValueError("native loader changed canonical image membership")
    return result, data, cfg
