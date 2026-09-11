"""真實 HOG 校準比例與不可行條件的 CPU 回歸檢查。"""
import math
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from yolo_optimize.training import calibration_coefficient

def test_recorded_tasks_share_feasible_coefficient():
    tasks = [(0.2648051360923013, 0.0022811430367223806),
             (0.4657386665699108, 0.04422562934602963)]
    n, h = map(sum, zip(*tasks))
    mu = calibration_coefficient(n, h, task_squared=tasks)
    assert mu == pytest.approx(0.21548, rel=0.001)
    assert all(0.02 <= mu * math.sqrt(b/a) <= 0.10 for a,b in tasks)
    assert mu * math.sqrt(h/n) == pytest.approx(0.054369, rel=0.001)

def test_feasible_global_target_is_unchanged():
    assert calibration_coefficient(400, 4, task_squared=[(200,2),(200,2)]) == 0.5

@pytest.mark.parametrize('tasks', [[(1,1),(1,100)], [(0,1)], [(1,float('nan'))]])
def test_infeasible_or_invalid_tasks_fail_closed(tasks):
    with pytest.raises(FloatingPointError):
        calibration_coefficient(2,2,task_squared=tasks)
