"""CPU 回歸：真實 native macro/EMA 搭配可控 overflow，禁止啟動 GPU。

YOLO_OPTIMIZE_TEST_NATIVE_BASELINE=1 可將受測類別換回原生實作，
重現 retry BN/RNG 累計與 frozen EMA 舍入的原始失敗。
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_optimize import runtime
from ultralytics.utils.torch_utils import ModelEMA
from yolo_combine.contracts import Task
from yolo_combine.joint_loss import MacroStepEngine, TaskLossResult

if os.environ.get("YOLO_OPTIMIZE_TEST_NATIVE_BASELINE") == "1":
    RetrySafeMacroStepEngine = MacroStepEngine
    FixedStateEMA = ModelEMA
else:
    from yolo_optimize.training_safety import FixedStateEMA, RetrySafeMacroStepEngine


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.actual_steps = 0

    def step(self, closure=None):
        self.actual_steps += 1
        return super().step(closure)


class InjectedOverflowScaler:
    """保留 scale/unscale/skip-step 行為，只在指定次數的 unscale 注入 inf。"""

    def __init__(self, *, overflows: int, initial_scale: float):
        self.remaining = overflows
        self.value = initial_scale
        self.overflow = False
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, outputs):
        return outputs * self.value

    def unscale_(self, optimizer):
        parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        for parameter in parameters:
            parameter.grad.div_(self.value)
        self.overflow = self.remaining > 0
        if self.overflow:
            parameters[0].grad.flatten()[0] = float("inf")

    def step(self, optimizer):
        self.step_calls += 1
        if not self.overflow:
            return optimizer.step()
        return None

    def update(self):
        self.update_calls += 1
        if self.overflow:
            self.remaining -= 1
            self.value /= 2

    def get_scale(self):
        return self.value

    def is_enabled(self):
        return True


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2, momentum=0.1)
        self.dropout = nn.Dropout(0.3)
        self.projection = nn.Conv2d(2, 1, 1)

    def forward(self, images):
        return self.projection(self.dropout(self.bn(images)))


class ToyLosses:
    def __init__(self, model):
        self.model = model
        self.outputs = []

    def loss_for(self, task, batch):
        output = self.model(batch["img"])
        self.outputs.append(output.detach().clone())
        per_image = output.square().flatten(1).mean(1)
        return TaskLossResult(task=Task(task), raw_total=per_image.sum(),
                              components=per_image.mean().detach().reshape(1),
                              actual_batch_size=len(per_image))


def batches():
    image = torch.arange(64, dtype=torch.float32).reshape(2, 2, 4, 4) / 64
    return {"detect_batches": [{"img": image}, {"img": image + 0.2}],
            "pose_batches": [{"img": image - 0.3}]}


def make_engine(engine_type, initial, *, overflows=0, retries=2, scale=4.0):
    model = deepcopy(initial).train()
    losses = ToyLosses(model)
    optimizer = CountingSGD(model.parameters())
    scaler = InjectedOverflowScaler(overflows=overflows, initial_scale=scale)
    engine = engine_type(model=model, losses=losses, optimizer=optimizer, scaler=scaler,
                         reference_batch_size=4, max_grad_norm=10.0, max_amp_retries=retries)
    return engine, model, losses, optimizer, scaler


class RetrySafetyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.initial = ToyModel()

    def test_retry_matches_one_successful_macro_not_failed_bn_and_dropout_attempts(self):
        clean = make_engine(MacroStepEngine, self.initial)
        retry = make_engine(RetrySafeMacroStepEngine, self.initial, overflows=1, scale=8.0)
        torch.manual_seed(130)
        clean_report = clean[0].run(**batches())
        clean_rng = torch.get_rng_state()
        torch.manual_seed(130)
        with patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("CPU test must not read CUDA RNG")):
            retry_report = retry[0].run(**batches())
        self.assertEqual(int(clean[1].bn.num_batches_tracked), 3)
        self.assertEqual(int(retry[1].bn.num_batches_tracked), 3)
        self.assertTrue(torch.equal(torch.get_rng_state(), clean_rng))
        self.assertEqual(clean[3].actual_steps, 1)
        self.assertEqual(retry[3].actual_steps, 1)
        self.assertEqual(retry[4].step_calls, 2)  # 一次 skip、一次真正 step
        self.assertEqual(retry[4].update_calls, 2)
        self.assertEqual(retry_report.amp_overflow_retries, 1)
        self.assertAlmostEqual(clean_report.joint_mean_loss, retry_report.joint_mean_loss, places=7)
        for name, value in clean[1].state_dict().items():
            torch.testing.assert_close(value, retry[1].state_dict()[name], rtol=0, atol=0)
        for expected, actual in zip(clean[2].outputs, retry[2].outputs[-3:], strict=True):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_multiple_retries_restore_the_original_macro_snapshot(self):
        clean = make_engine(MacroStepEngine, self.initial, scale=2.0)
        retry = make_engine(RetrySafeMacroStepEngine, self.initial, overflows=2, scale=8.0)
        torch.manual_seed(140)
        clean[0].run(**batches())
        torch.manual_seed(140)
        report = retry[0].run(**batches())
        self.assertEqual(report.amp_overflow_retries, 2)
        self.assertEqual(retry[3].actual_steps, 1)
        self.assertEqual(int(retry[1].bn.num_batches_tracked), 3)
        for name, value in clean[1].state_dict().items():
            torch.testing.assert_close(value, retry[1].state_dict()[name], rtol=0, atol=0)

    def test_exhausted_retries_restore_bn_rng_and_leave_no_snapshot(self):
        engine, model, _, optimizer, _ = make_engine(
            RetrySafeMacroStepEngine, self.initial, overflows=2, retries=1, scale=8.0)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        torch.manual_seed(150)
        rng = torch.get_rng_state()
        with self.assertRaises(FloatingPointError):
            engine.run(**batches())
        self.assertEqual(optimizer.actual_steps, 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in original.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(hasattr(engine, "_macro_safety_snapshot"))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_no_overflow_stays_identical_to_native(self):
        native = make_engine(MacroStepEngine, self.initial)
        guarded = make_engine(RetrySafeMacroStepEngine, self.initial)
        torch.manual_seed(160)
        expected = native[0].run(**batches())
        torch.manual_seed(160)
        actual = guarded[0].run(**batches())
        self.assertEqual(expected, actual)
        for name, value in native[1].state_dict().items():
            torch.testing.assert_close(value, guarded[1].state_dict()[name], rtol=0, atol=0)
        self.assertFalse(hasattr(guarded[0], "_macro_safety_snapshot"))


class HOGRetryStatisticsTests(unittest.TestCase):
    def test_actual_hog_router_reports_only_the_successful_attempt(self):
        from yolo_optimize.hog import HOGAuxiliary
        from yolo_optimize.training import HOGTaskLossRouter, TrainingModel

        class P3Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.p3_masf = nn.Identity()

            def forward(self, features):
                return self.p3_masf(features)

        class RawP3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.AvgPool2d(8), nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2))
                self.graph = nn.Module()
                self.graph.model = nn.ModuleList([nn.Identity() for _ in range(16)] + [P3Layer()])
                self.dropout = nn.Dropout(0.3)
                self.projection = nn.Conv2d(2, 1, 1)

            def forward(self, images):
                raw_p3 = self.features(images)
                p3 = self.graph.model[16](raw_p3)
                return self.projection(self.dropout(p3))

        torch.manual_seed(170)
        initial = TrainingModel(RawP3Model(), HOGAuxiliary(2))
        image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 16).expand(2, 3, 16, 16) / 15
        batch = {"img": image, "bboxes": torch.tensor([[0.5, 0.5, 1., 1.], [0.5, 0.5, 1., 1.]]),
                 "batch_idx": torch.tensor([0, 1])}
        inputs = {"detect_batches": [batch, batch], "pose_batches": [batch]}

        def run_attempts(overflows, *, stale_statistics=False):
            model = deepcopy(initial).train()
            native = ToyLosses(model.base)
            router = HOGTaskLossRouter(native, model, device="cpu", amp=False)
            router.mu = 0.05
            if stale_statistics:
                router.last_aux = {"previous_macro": {"images": 999}}
            optimizer = CountingSGD(model.parameters())
            scaler = InjectedOverflowScaler(overflows=overflows, initial_scale=4.0 * 2 ** overflows)
            engine = RetrySafeMacroStepEngine(
                model=model, losses=router, optimizer=optimizer, scaler=scaler,
                reference_batch_size=4, max_amp_retries=overflows,
            )
            torch.manual_seed(180)
            report = engine.run(**inputs)
            return model, router, optimizer, scaler, report

        reference = run_attempts(0)
        for overflows in (1, 4):
            with self.subTest(overflows=overflows):
                actual = run_attempts(overflows, stale_statistics=True)
                self.assertEqual(actual[2].actual_steps, 1)
                self.assertEqual(actual[3].step_calls, overflows + 1)
                self.assertEqual(actual[4].amp_overflow_retries, overflows)
                for task, expected_images, expected_cells, expected_sizes in (
                    ("detect", 4, 16, [2, 2]), ("pose", 2, 8, [2]),
                ):
                    expected = reference[1].last_aux[task]
                    observed = actual[1].last_aux[task]
                    self.assertEqual(expected["images"], expected_images)
                    self.assertEqual(expected["valid_cells"], expected_cells)
                    self.assertEqual(expected["physical_batch_sizes"], expected_sizes)
                    for field in ("images", "valid_cells", "images_with_valid_cells", "physical_batch_sizes"):
                        self.assertEqual(observed[field], expected[field], f"{task}.{field}")
                    self.assertAlmostEqual(observed["loss_per_image_mean"], expected["loss_per_image_mean"], places=7)
                    self.assertEqual(observed["mu"], 0.05)
                self.assertNotIn("previous_macro", actual[1].last_aux)
                self.assertAlmostEqual(actual[4].joint_mean_loss, reference[4].joint_mean_loss, places=7)
                for name, value in reference[0].state_dict().items():
                    torch.testing.assert_close(value, actual[0].state_dict()[name], rtol=0, atol=0)


class EMAFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed = nn.Parameter(torch.tensor([1.234567, 0.001]), requires_grad=False)
        self.aux = nn.Parameter(torch.tensor([1.234567, 0.001]))
        self.frozen_bn = nn.BatchNorm1d(2).eval()
        self.train_bn = nn.BatchNorm1d(2)
        self.branch = nn.Module()
        self.branch.attn = nn.Module()
        self.branch.attn.score = nn.Module()
        self.branch.attn.score.register_buffer("fixed_coefficients", self.fixed.detach().clone())
        self.branch.attn.score.register_buffer("calibration_scale", self.fixed.detach().clone())
        self.branch.attn.normalize = nn.Module()
        self.branch.attn.normalize.register_buffer("knots", self.fixed.detach().clone())
        self.branch.attn.normalize.register_buffer("values", self.fixed.detach().clone())
        self.frozen_bn.running_mean.copy_(self.fixed)
        self.frozen_bn.running_var.copy_(self.fixed)
        self.frozen_bn.num_batches_tracked.fill_(9)


class FixedEMATests(unittest.TestCase):
    def test_immutable_floats_remain_bit_exact(self):
        model = EMAFixture()
        ema = FixedStateEMA(model)
        ema.update(model)
        for name in ("fixed", "branch.attn.score.fixed_coefficients", "branch.attn.score.calibration_scale",
                     "branch.attn.normalize.knots", "branch.attn.normalize.values",
                     "frozen_bn.running_mean", "frozen_bn.running_var", "frozen_bn.num_batches_tracked"):
            self.assertTrue(torch.equal(model.state_dict()[name], ema.ema.state_dict()[name]), name)

    def test_trainables_and_train_bn_follow_native_decay_and_update_age(self):
        model = EMAFixture()
        guarded = FixedStateEMA(model, decay=0.91, tau=21, updates=4)
        native = ModelEMA(model, decay=0.91, tau=21, updates=4)
        with torch.no_grad():
            model.aux.add_(2)
            model.train_bn.running_mean.add_(0.3)
            model.train_bn.running_var.add_(0.4)
        guarded.update(model)
        native.update(model)
        self.assertEqual(guarded.updates, native.updates)
        self.assertEqual(guarded.updates, 5)
        self.assertEqual(guarded.decay(5), native.decay(5))
        for name in ("aux", "train_bn.running_mean", "train_bn.running_var"):
            torch.testing.assert_close(guarded.ema.state_dict()[name], native.ema.state_dict()[name], rtol=0, atol=0)
            self.assertFalse(torch.equal(guarded.ema.state_dict()[name], model.state_dict()[name]))

    def test_metadata_and_disabled_ema_preserve_native_semantics(self):
        model = EMAFixture()
        ema = FixedStateEMA(model, updates=3)
        self.assertIn("fixed", ema.fixed_state_names)
        self.assertIn("frozen_bn.num_batches_tracked", ema.fixed_state_names)
        self.assertNotIn("aux", ema.fixed_state_names)
        self.assertNotIn("train_bn.running_mean", ema.fixed_state_names)
        self.assertEqual(ema.fixed_state_names, tuple(sorted(ema.fixed_state_names)))
        before = deepcopy(ema.ema.state_dict())
        ema.enabled = False
        with torch.no_grad():
            model.fixed.add_(1)
            model.aux.add_(1)
        ema.update(model)
        self.assertEqual(ema.updates, 3)
        for name, value in before.items():
            torch.testing.assert_close(value, ema.ema.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
