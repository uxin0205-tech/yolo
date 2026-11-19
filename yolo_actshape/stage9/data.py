"""固定 canvas；完整沿用 canonical membership／labels，只改執行時幾何。"""

import copy
import math

import cv2
import numpy as np
from torch.utils.data import Dataset
from ultralytics.data.augment import Compose, Format, LetterBox, RandomHSV, RandomFlip
from stage6.data import dataset as upstream_dataset


def geometry(original, target):
    h0, w0 = map(int, original)
    h, w = map(int, target)
    if min(h0, w0, h, w) <= 0 or h % 32 or w % 32 or max(h, w) > 1024:
        raise ValueError("canvas 必須為正的 stride32 倍數且單軸不超過1024")
    ratio = min(h / h0, w / w0)
    resized = max(1, round(h0 * ratio)), max(1, round(w0 * ratio))
    top, left = (h - resized[0]) // 2, (w - resized[1]) // 2
    return {"resized": resized, "gain": (ratio, ratio),
            "padding": (left, top), "padding_fraction": 1 - resized[0] * resized[1] / (h * w)}


class CanvasDataset(Dataset):
    """保持原 membership；避免把 (H,W) 傳進會取 max 的 imgsz 介面。"""

    def __init__(self, source, shape, training=False):
        self.source = source
        self.shape = tuple(shape)
        geometry((1, 1), self.shape)
        self.im_files = source.im_files
        self.labels = source.labels
        self.collate_fn = source.collate_fn
        self.transforms = [LetterBox(self.shape, auto=False, scaleup=True, stride=32)]
        # Isolate shape recovery from random crop/scale: HSV + inherited legal flip only.
        if training:
            self.transforms.append(RandomHSV(hgain=.015, sgain=.7, vgain=.4))
            if not source.use_keypoints:
                self.transforms.append(RandomFlip(p=.5, direction="horizontal"))
        self.transforms.append(Format(bbox_format="xywh", normalize=True,
                                      return_keypoint=source.use_keypoints, batch_idx=True))
        self.transforms = Compose(self.transforms)

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        label = copy.deepcopy(self.labels[index])
        label.pop("shape", None)
        image = cv2.imread(self.im_files[index])
        if image is None:
            raise ValueError(f"無法讀取正式影像：{self.im_files[index]}")
        original = image.shape[:2]
        g = geometry(original, self.shape)
        # LetterBox scales annotations by the same scalar used by scale_boxes/coords.
        # Do not pre-resize with independent ceil gains: that would skew the inverse.
        label.update(img=image, ori_shape=original, resized_shape=original, ratio_pad=g["gain"])
        label = self.source.update_labels_info(label)
        result = self.transforms(label)
        if tuple(result["img"].shape[-2:]) != self.shape:
            raise ValueError("前處理 canvas 改變")
        return result


def dataset(task, mode, shape, batch=8, epochs=5):
    source, data, cfg = upstream_dataset(task, mode, batch, epochs)
    if shape is None:
        if mode != "val":
            raise ValueError("不新增原形狀控制加訓")
        return source, data, cfg
    result = CanvasDataset(source, shape, training=mode == "train")
    if result.im_files is not source.im_files or len(result) != len(source):
        raise ValueError("正式資料 membership 改變")
    return result, data, cfg
