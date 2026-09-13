"""fixed_scale early-return 的 deepcopy 狀態隔離回歸。"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import MethodType
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from yolo_optimize.fixed_scale import install_fixed_early_return


class DummyScore:
    def __init__(self, value=1.0, mode='dynamic', calibration=False):
        self.value = value
        self.scale_mode = mode
        self.calibration_enabled = calibration
        self.calls = 0
        self.fixed_calls = 0

    def _coefficient(self, index, q, k, attention_scale):
        self.calls += 1
        return self.value

    def _fixed(self, index):
        self.fixed_calls += 1
        return 40.0 + index


class FixedScaleCopyTests(unittest.TestCase):
    def test_dynamic_fallback_uses_clone(self):
        source = DummyScore(value=1.0)
        install_fixed_early_return(source)
        clone = copy.deepcopy(source)
        clone.value = 9.0
        self.assertEqual(clone._coefficient(0, None, None, None), 9.0)
        self.assertEqual(source.calls, 0)
        self.assertEqual(clone.calls, 1)

    def test_calibration_fallback_uses_clone(self):
        source = DummyScore(value=2.0, mode='power_of_two', calibration=True)
        install_fixed_early_return(source)
        clone = copy.deepcopy(source)
        clone.value = 7.0
        self.assertEqual(clone._coefficient(0, None, None, None), 7.0)
        self.assertEqual(source.calls, 0)
        self.assertEqual(clone.calls, 1)

    def test_fixed_path_uses_clone_fixed_method(self):
        source = DummyScore(mode='power_of_two', calibration=False)
        install_fixed_early_return(source)
        clone = copy.deepcopy(source)
        self.assertEqual(clone._coefficient(3, None, None, None), 43.0)
        self.assertEqual(source.fixed_calls, 0)
        self.assertEqual(clone.fixed_calls, 1)
        self.assertEqual(clone.calls, 0)

    def test_duplicate_install_is_rejected(self):
        score = DummyScore()
        install_fixed_early_return(score)
        with self.assertRaises(ValueError):
            install_fixed_early_return(score)

    def test_foreign_bound_method_is_rejected_without_mutation(self):
        score = DummyScore()
        other = DummyScore(value=5.0)
        score._coefficient = MethodType(DummyScore._coefficient, other)
        with self.assertRaises(TypeError):
            install_fixed_early_return(score)
        self.assertFalse(hasattr(score, '_fixed_early_return_installed'))


if __name__ == '__main__':
    unittest.main()
