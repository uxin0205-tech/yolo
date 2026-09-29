"""真實 CPU macro 軌跡對照；不初始化 CUDA，不讀取正式模型或資料。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_optimize.ema_diagnostic import PairedEMAObserver, parent_ema_updates
from yolo_optimize.training_safety import FixedStateEMA, RetrySafeMacroStepEngine

# 沿用已驗證的真實 BN/dropout 與 scaler 故障注入 fixture，不 mock macro engine。
from test_training_safety import CountingSGD, InjectedOverflowScaler, ToyLosses, ToyModel, batches


class PairedEMATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(220)
        self.initial = ToyModel().train()
        self.initial.fixed = nn.Parameter(torch.tensor([1.234567, 0.001]), requires_grad=False)
        self.parent_updates = parent_ema_updates({"ema_updates": 26597})

    def run_trajectory(self, mode, *, steps=3, overflows=0):
        model = deepcopy(self.initial)
        if mode == "paired":
            observer = PairedEMAObserver(model, self.parent_updates)
        else:
            observer = FixedStateEMA(model, updates=0 if mode == "fresh" else self.parent_updates)
        optimizer = CountingSGD(model.parameters())
        losses = ToyLosses(model)
        scaler = InjectedOverflowScaler(overflows=overflows, initial_scale=4.0 * 2 ** overflows)
        engine = RetrySafeMacroStepEngine(
            model=model, losses=losses, optimizer=optimizer, scaler=scaler, ema=observer,
            reference_batch_size=4, max_amp_retries=overflows,
        )
        torch.manual_seed(230)
        reports = []
        for _ in range(steps):
            reports.append(engine.run(**batches()))
        return model, observer, optimizer, reports, torch.get_rng_state()

    def assert_same_state(self, actual, expected):
        self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)

    def test_shared_trajectory_matches_two_independent_single_ema_references(self):
        paired = self.run_trajectory("paired")
        fresh = self.run_trajectory("fresh")
        continued = self.run_trajectory("continued")
        for reference in (fresh, continued):
            self.assert_same_state(paired[0], reference[0])
            self.assertEqual(paired[2].actual_steps, reference[2].actual_steps)
            self.assertEqual(paired[3], reference[3])
            self.assertTrue(torch.equal(paired[4], reference[4]))
        self.assert_same_state(paired[1].fresh.ema, fresh[1].ema)
        self.assert_same_state(paired[1].continued.ema, continued[1].ema)
        self.assertEqual(paired[2].actual_steps, 3)
        self.assertEqual(paired[1].fresh.updates, 3)
        self.assertEqual(paired[1].continued.updates, 26600)
        self.assertFalse(torch.equal(paired[1].fresh.ema.projection.weight,
                                    paired[1].continued.ema.projection.weight))
        for observer in (paired[1].fresh, paired[1].continued):
            self.assertTrue(torch.equal(observer.ema.fixed, paired[0].fixed))

    def test_one_overflow_updates_both_observers_only_once(self):
        reference = self.run_trajectory("paired", steps=1)
        with patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("CPU test cannot read CUDA RNG")):
            retry = self.run_trajectory("paired", steps=1, overflows=1)
        self.assertEqual(retry[2].actual_steps, 1)
        self.assertEqual(retry[3][0].amp_overflow_retries, 1)
        self.assertEqual(retry[1].observations, 1)
        self.assertEqual(retry[1].fresh.updates, 1)
        self.assertEqual(retry[1].continued.updates, 26598)
        self.assert_same_state(retry[0], reference[0])
        self.assert_same_state(retry[1].fresh.ema, reference[1].fresh.ema)
        self.assert_same_state(retry[1].continued.ema, reference[1].continued.ema)
        self.assertTrue(torch.equal(retry[4], reference[4]))

    def test_observer_is_external_and_starts_from_identical_parent_state(self):
        model = deepcopy(self.initial)
        original_modules = tuple(model.named_modules())
        original_parameters = tuple(model.parameters())
        rng = torch.get_rng_state()
        observer = PairedEMAObserver(model, self.parent_updates)
        self.assertNotIsInstance(observer, nn.Module)
        self.assertEqual(tuple(model.named_modules()), original_modules)
        self.assertEqual(tuple(model.parameters()), original_parameters)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assert_same_state(observer.fresh.ema, model)
        self.assert_same_state(observer.continued.ema, model)
        self.assertTrue(all(not parameter.requires_grad for parameter in observer.fresh.ema.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in observer.continued.ema.parameters()))
        self.assertEqual(observer.fresh.fixed_state_names, observer.continued.fixed_state_names)
        metadata = observer.metadata()
        self.assertEqual(metadata["initial_updates"], {"fresh": 0, "continued": 26597})
        self.assertEqual(metadata["decay"], 0.9999)
        self.assertEqual(metadata["tau"], 2000)
        json.dumps(metadata, allow_nan=False)

    def test_top_level_parent_age_cannot_be_guessed_or_coerced(self):
        for payload in ({}, {"progress": {"ema_updates": 26597}},
                        *({"ema_updates": value} for value in (-1, True, False, 26597.0, "26597", None))):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parent_ema_updates(payload)
        for value in (-1, True, 26597.0):
            with self.assertRaises(ValueError):
                PairedEMAObserver(self.initial, value)
        self.assertEqual(parent_ema_updates({"ema_updates": 0}), 0)


if __name__ == "__main__":
    unittest.main()
