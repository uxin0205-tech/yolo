"""新方向的實際圖、標註反算與隔離 queue 邊界；不啟動 GPU。"""

import copy
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from . import common, runner
from .common import ROOT, HERE, RUN, NAME, protocol, read, write, digest, guard


class QueueBoundaries(unittest.TestCase):
    def test_isolated_queue_and_engine_restore_after_exception(self):
        from stage9_extension import runner as previous
        from stage9 import training
        saved_queue = previous.RUN, previous.HERE, previous.protocol, previous.guard, previous.state
        saved_training = training.RUN, training.guard, training.protocol
        with self.assertRaisesRegex(ValueError, "驗證還原"):
            with common.queue_engine(runner.state) as engine, common.training_run() as train:
                spec = engine.make_spec("evaluate", NAME, protocol()["parent"])
                self.assertEqual(spec["shape"], [768, 512])
                self.assertTrue(Path(spec["output"]).is_relative_to(RUN))
                self.assertEqual(train.RUN, RUN)
                self.assertEqual(train.protocol()["training"], common.previous.protocol()["training"])
                raise ValueError("驗證還原")
        self.assertEqual(saved_queue, (previous.RUN, previous.HERE, previous.protocol, previous.guard, previous.state))
        self.assertEqual(saved_training, (training.RUN, training.guard, training.protocol))

    def test_recovery_preserves_targets_and_stops_after_one(self):
        with common.training_run(calibration_recovery=True) as engine:
            cfg = engine.protocol()["calibration"]
            self.assertEqual((cfg["macros"], cfg["confirm"], cfg["attempts"]), (16, 8, 2))
            for name in ("target_ratio", "band", "bounds", "step"):
                self.assertEqual(cfg[name], protocol()["calibration"][name])
        from stage9 import training
        with tempfile.TemporaryDirectory(prefix="portrait-calibration-") as tmp:
            parent = protocol()["parent"]
            spec = {"parent": parent, "output": tmp}
            failed = dict(status="needs_review", parent_restored=True, validation_used=False,
                          parent_sha256=parent["sha256"])
            def fail(_spec):
                write(Path(tmp) / "result.json", failed)
                raise ValueError("任務梯度比例未通過確認")
            write(Path(tmp) / "bounded-recovery/attempt.json", {"already_tried": True})
            with patch.object(training, "calibrate", side_effect=fail) as call:
                with self.assertRaisesRegex(ValueError, "執行過一次"):
                    runner.recovered_calibration(spec)
                self.assertEqual(call.call_count, 1)

    def test_unrelated_calibration_failure_is_not_retried(self):
        from stage9 import training
        with tempfile.TemporaryDirectory(prefix="portrait-failure-") as tmp:
            with patch.object(training, "calibrate", side_effect=ValueError("資料錯誤")) as call:
                with self.assertRaisesRegex(ValueError, "資料錯誤"):
                    runner.recovered_calibration({"parent": protocol()["parent"], "output": tmp})
                self.assertEqual(call.call_count, 1)
            self.assertFalse((Path(tmp) / "bounded-recovery").exists())


def run():
    from stage9.model import load, verify
    from stage9.diagnostics import trace
    from stage9.portable import records
    from stage9.data import dataset, geometry
    torch.set_num_threads(2)
    guard(create=True)
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QueueBoundaries))
    if not tests.wasSuccessful():
        raise ValueError("queue 邊界驗證失敗")
    model = load().cpu().eval()
    traced = trace(model, [768, 512])
    expected = {"graph.model.16": [96, 64], "graph.model.19": [48, 32], "graph.model.22": [24, 16]}
    for row in traced["shapes"]:
        if row["module"] in expected:
            assert row["output"][-2:] == expected[row["module"]], row
    assert len(traced["attention"]) == 2
    assert all((x["height"], x["width"], x["tokens"]) == (24, 16, 384) for x in traced["attention"])
    with torch.inference_mode():
        sample = torch.rand(1, 3, 192, 128)
        before = records(model(sample, task="both"))
        model(torch.zeros(1, 3, 128, 192), task="both")
        assert before == records(model(sample, task="both")), "形狀 cache 未還原"
        norm = model.get_submodule("graph.model.10.m.0.attn.normalizer")
        probability = norm(torch.zeros(1, 4, 2, 384))
        row_error = float((probability.sum(-1) - 1).abs().max())
        assert row_error <= .003 and probability.min() >= 0 and probability.max() <= 1
    verify(model)
    data = {}
    for task, expected_val, expected_train in (("detect", 5000, 118287), ("pose", 683, 5964)):
        ds, _, _ = dataset(task, "val", [768, 512])
        assert len(ds) == expected_val and ds.im_files is ds.source.im_files and ds.labels is ds.source.labels
        dims = []
        for path in ds.im_files:
            with Image.open(path) as im:
                dims.append((im.height, im.width))
        conditions = {f"{shape[0]}x{shape[1]}": {
            "mean_padding_fraction": statistics.mean(geometry(hw, shape)["padding_fraction"] for hw in dims),
            "mean_linear_gain": statistics.mean(geometry(hw, shape)["gain"][0] for hw in dims)}
            for shape in ([512, 768], [768, 512])}
        errors = []
        # 真實標註經 LetterBox/Format 後反算，確認 H/W 沒有互換，也沒有旋轉影像。
        for index in (0, len(ds) // 2, len(ds) - 1):
            item = ds[index]
            h0, w0 = dims[index]
            gain = geometry((h0, w0), [768, 512])["gain"][0]
            left, top = geometry((h0, w0), [768, 512])["padding"]
            boxes = item["bboxes"].numpy().copy()
            boxes *= np.array([512, 768, 512, 768])
            boxes[:, :2] -= [left, top]
            boxes /= gain
            original = ds.labels[index]["bboxes"].copy()
            if ds.labels[index].get("normalized", True):
                original *= np.array([w0, h0, w0, h0])
            assert boxes.shape == original.shape
            error = float(np.max(np.abs(boxes - original))) if boxes.size else 0.
            assert error < .003, (task, index, error)
            if task == "pose":
                actual = item["keypoints"].numpy()[..., :2] * np.array([512, 768])
                actual = (actual - [left, top]) / gain
                original_kp = ds.labels[index]["keypoints"][..., :2] * np.array([w0, h0])
                visible = ds.labels[index]["keypoints"][..., 2] > 0
                kp_error = float(np.max(np.abs(actual[visible] - original_kp[visible]))) if visible.any() else 0.
                assert kp_error < .003, kp_error
                error = max(error, kp_error)
            errors.append(error)
        train, _, _ = dataset(task, "train", [768, 512], epochs=10)
        assert len(train) == expected_train and train.im_files is train.source.im_files
        data[task] = {"val": len(ds), "train": len(train), "membership_preserved": True,
                      "coordinate_roundtrip_max_pixel_error": max(errors), "conditions": conditions}
    if torch.cuda.is_initialized():
        raise ValueError("CPU preflight 不應使用 GPU")
    guard()
    result = {"status": "passed", "source_sha256": digest(RUN / "source.json"), "tests": tests.testsRun,
              "parent_sha256": protocol()["parent"]["sha256"], "trace": traced, "data": data,
              "shape_cache_bitwise_equal": True, "D1_384_row_sum_error": row_error,
              "operators_verified": True, "GPU_used": False}
    write(RUN / "preflight.json", result)
    print("PASS：768×512 圖形、標註、完整資料與 queue 隔離契約", flush=True)
