"""CPU 合約與排程邊界測試；不佔用正在訓練的 GPU。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import torch

from stage9 import training
from stage9.model import load
from . import common, runner
from .layout import tile_roundtrip, pitch_roundtrip, tiled_conv, tiled_normalization, run
from .report import choose


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        cls.model = load().cpu().eval()

    def test_tile_and_pitch_preserve_rectangular_tail(self):
        for h, w in ((7, 13), (40, 40), (24, 32)):
            x = torch.arange(2 * 3 * h * w, dtype=torch.float32).reshape(2, 3, h, w)
            for tile in (8, 16, 32):
                with self.subTest(h=h, w=w, tile=tile):
                    restored, info = tile_roundtrip(x, tile)
                    pitched, _ = pitch_roundtrip(x, tile)
                    self.assertTrue(torch.equal(restored, x))
                    self.assertTrue(torch.equal(pitched, x))
                    self.assertLessEqual(info["valid_cell_utilization"], 1)

    def test_conv_halo_stride_dilation_groups_and_qsilu(self):
        activation = self.model.graph.model[17].act
        self.assertEqual(type(activation).__name__, "ShiftPiecewiseQuadraticSiLU")
        x = torch.randn(2, 4, 17, 29)
        for stride, dilation, groups in ((1, 1, 1), (2, 1, 1), (2, 2, 2)):
            conv = torch.nn.Conv2d(4, 8, 3, stride=stride, padding=dilation, dilation=dilation, groups=groups)
            with torch.inference_mode():
                expected = activation(conv(x))
                for tile in (4, 8, 16):
                    torch.testing.assert_close(tiled_conv(x, conv, tile, activation), expected, atol=1e-5, rtol=1e-5)

    def test_D1_keeps_full_denominator_for_nonpower_tail(self):
        norm = self.model.get_submodule("graph.model.10.m.0.attn.normalizer")
        self.assertNotIsInstance(norm, torch.nn.Softmax)
        with torch.inference_mode():
            for keys in (128, 192, 384, 400):
                scores = torch.randn(1, 4, 37, keys) * 3
                values = torch.randn(1, 4, 64, keys)
                expected_p = norm(scores)
                expected_v = values @ expected_p.transpose(-2, -1)
                for tile in (16, 32, 64):
                    p, v = tiled_normalization(norm, scores, values, tile)
                    self.assertTrue(torch.equal(p, expected_p))
                    torch.testing.assert_close(v, expected_v, atol=1e-5, rtol=1e-5)

    def test_real_detect_pose_hooks_on_CPU(self):
        with tempfile.TemporaryDirectory(prefix="actshape-layout-") as tmp:
            result = run({"parent": common.protocol()["parent"], "candidate": "cpu-integration",
                          "shape": [64, 96], "output": tmp}, device="cpu")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["rows"]), 42)
            self.assertTrue(all(r["passed"] for r in result["rows"]))
            self.assertIsNone(result["whole_model_both_timing"])
            common.write(common.RUN / "cpu-layout-integration.json", result)
        self.assertFalse(torch.cuda.is_initialized())


class QueueTests(unittest.TestCase):
    def test_engine_adapter_restores_even_on_failure(self):
        previous = training.RUN, training.guard
        with self.assertRaisesRegex(ValueError, "test-only"):
            with common.training_run() as engine:
                self.assertEqual(engine.RUN, common.RUN)
                self.assertIs(engine.guard, common.guard)
                raise ValueError("test-only")
        self.assertEqual((training.RUN, training.guard), previous)

    def test_initial_complete_status_still_requires_exclusive_lock(self):
        with patch.object(runner, "stopped", return_value=False), \
             patch.object(runner, "read", return_value={"status": "complete"}), \
             patch.object(runner.fcntl, "flock", side_effect=[BlockingIOError, None]) as flock, \
             patch.object(runner, "state", return_value={}), patch.object(runner, "monitor"), \
             patch.object(runner.time, "sleep") as sleep:
            runner.acquire_initial(Mock())
            self.assertEqual(flock.call_count, 2)
            sleep.assert_called_once()

    def test_orphan_initial_is_not_skipped(self):
        with patch.object(runner, "stopped", return_value=False), \
             patch.object(runner, "read", return_value={"status": "running"}), \
             patch.object(runner.fcntl, "flock"):
            with self.assertRaisesRegex(RuntimeError, "未完成"):
                runner.acquire_initial(Mock())

    def test_resume_requires_matching_full_epoch_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "resume.pt").write_bytes(b"test-only")
            common.write(output / "config.json", {"test": 1})
            meta = {"next_epoch": 2, "sha256": common.digest(output / "resume.pt"),
                    "config_sha256": common.digest(output / "config.json"),
                    "source_sha256": common.digest(common.RUN / "source.json")}
            common.write(output / "resume.json", meta)
            self.assertTrue(runner.resume_valid(output))
            common.write(output / "resume.json", {**meta, "next_epoch": 1})
            self.assertFalse(runner.resume_valid(output))
            common.write(output / "resume.json", {**meta, "source_sha256": "wrong"})
            self.assertFalse(runner.resume_valid(output))

    def test_quality_guard_not_hidden_by_aggregate_improvement(self):
        p = common.protocol()
        original = {**p["parent"], "candidate": "reference_rect640", "shape": None}
        bad = {**original, "candidate": "rect256_512", "shape": [256, 512],
               "metrics": {k: v + .02 for k, v in original["metrics"].items()}}
        bad["metrics"]["person_AP"] = original["metrics"]["person_AP"] - .02
        best, _, promoted = choose([original, bad], {"rect256_512": 15e9}, p)
        self.assertFalse(promoted)
        self.assertEqual(best["candidate"], "reference_rect640")
        good = copy.deepcopy(bad)
        good["metrics"]["person_AP"] = original["metrics"]["person_AP"] + .002
        best, _, promoted = choose([original, good], {"rect256_512": 15e9}, p)
        self.assertTrue(promoted)
        self.assertEqual(best["candidate"], "rect256_512")

    def test_expansion_budget_and_initial_source_unchanged(self):
        p = common.protocol()
        self.assertEqual(len(p["candidates"]) * (p["training"]["epochs"] + p["training"]["low_lr_epochs"]), 60)
        self.assertEqual(p["selection"]["total_with_initial_max_epochs"], 100)
        self.assertEqual(common.digest(common.BASE_RUN / "source.json"), p["dependency"]["source_sha256"])
        common.base.guard()
        for name, c in p["candidates"].items():
            self.assertEqual(runner.make_spec("evaluate", name, p["parent"])["shape"], c["shape"])
