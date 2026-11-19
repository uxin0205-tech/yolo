"""驗證幾何反算、資料不變性及 queue 的停止／恢復邊界。"""

import copy
import signal
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import scale_boxes, scale_coords, xywh2xyxy

from .data import CanvasDataset, geometry
from .policy import choose_training
from .runner import retry_allowed


class Source:
    use_keypoints = True
    collate_fn = None

    def __init__(self, path):
        self.im_files = [str(path)]
        self.labels = [{"im_file": str(path), "shape": (1, 1), "cls": np.array([[0.]], np.float32),
                        "bboxes": np.array([[.5, .45, .4, .3]], np.float32),
                        "keypoints": np.array([[[.35, .3, 2.], [.7, .6, 2.]]], np.float32)}]

    @staticmethod
    def update_labels_info(label):
        label["instances"] = Instances(label.pop("bboxes"), np.zeros((0, 1000, 2), np.float32),
                                       label.pop("keypoints"), bbox_format="xywh", normalized=True)
        return label


class GeometryTests(unittest.TestCase):
    def test_actual_native_transform_and_inverse(self):
        for original in ((333, 997), (997, 333), (513, 1025), (1080, 1920)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "sample.png"
                image = np.full((*original, 3), (20, 80, 160), np.uint8)
                self.assertTrue(cv2.imwrite(str(path), image))
                src = Source(path)
                original_label = copy.deepcopy(src.labels[0])
                for shape in ((384, 768), (512, 512), (512, 1024), (1024, 1024)):
                    ds = CanvasDataset(src, shape)
                    item = ds[0]
                    self.assertEqual(tuple(item["img"].shape), (3, *shape))
                    self.assertIs(ds.im_files, src.im_files)
                    box = xywh2xyxy(item["bboxes"] * torch.tensor([shape[1], shape[0]] * 2))
                    restored = scale_boxes(shape, box, original, ratio_pad=item["ratio_pad"])
                    expected = xywh2xyxy(torch.tensor(original_label["bboxes"]) * torch.tensor([original[1], original[0]] * 2))
                    torch.testing.assert_close(restored, expected, atol=2e-4, rtol=1e-6)
                    kp = item["keypoints"].clone()
                    kp[..., :2] *= torch.tensor([shape[1], shape[0]])
                    restored_kp = scale_coords(shape, kp[..., :2], original, ratio_pad=item["ratio_pad"])
                    expected_kp = torch.tensor(original_label["keypoints"])[..., :2] * torch.tensor([original[1], original[0]])
                    torch.testing.assert_close(restored_kp, expected_kp, atol=2e-4, rtol=1e-6)
                    np.testing.assert_array_equal(src.labels[0]["bboxes"], original_label["bboxes"])
                    np.testing.assert_array_equal(src.labels[0]["keypoints"], original_label["keypoints"])
                    self.assertTrue(torch.equal(item["keypoints"][..., 2], torch.full((1, 2), 2.)))

    def test_invalid_geometry(self):
        for shape in ((0, 512), (511, 512), (1056, 512), (512, -32)):
            with self.assertRaises(ValueError):
                geometry((480, 640), shape)


class QueueTests(unittest.TestCase):
    def test_at_most_one_per_group_and_no_control_training(self):
        base = {"a": .6, "b": .7}
        specs = {"reference": {"train": False, "kind": "quality"},
                 "a": {"train": True, "kind": "efficient"},
                 "b": {"train": True, "kind": "efficient"},
                 "c": {"train": True, "kind": "quality"},
                 "bad": {"train": True, "kind": "quality"}}
        results = {"reference": {"metrics": {"a": .9, "b": .9}}, "a": {"metrics": base},
                   "b": {"metrics": {"a": .61, "b": .71}}, "c": {"metrics": base},
                   "bad": {"metrics": {"a": .1, "b": .99}}}
        selected, reasons = choose_training(specs, results, base,
                    {"screen_max_any_drop": .05, "screen_min_score_delta": -.02, "max_train_candidates": 2},
                    lambda v: sum(v.values()) / len(v))
        self.assertEqual(selected, ["b", "c"])
        self.assertFalse(reasons["bad"]["train"])

    def test_bounded_complete_epoch_sigsegv_retry(self):
        self.assertTrue(retry_allowed(-signal.SIGSEGV, 0, 1, True))
        self.assertFalse(retry_allowed(-signal.SIGSEGV, 1, 1, True))
        self.assertFalse(retry_allowed(-signal.SIGSEGV, 0, 1, False))
        self.assertFalse(retry_allowed(1, 0, 1, True))
        self.assertFalse(retry_allowed(-signal.SIGTERM, 0, 1, True))
