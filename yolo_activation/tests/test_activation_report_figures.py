from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/generate_activation_report_figures.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "activation_report_figures", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_accuracy_summary_uses_four_metrics_without_relu() -> None:
    generator = _load_generator()
    rows = generator._accuracy_rows(generator._load_results())
    summary_rows = [row for row in rows if row["metric"] in generator.SUMMARY_METRICS]

    assert len(summary_rows) == 2 * 4 * 4
    assert {row["activation"] for row in summary_rows} == set(generator.ACTIVATIONS)
    assert {row["metric"] for row in summary_rows} == set(generator.SUMMARY_METRICS)
    assert "relu" not in generator.ACTIVATIONS
    assert generator.CURVE_ACTIVATIONS[0] == "silu"


def test_operation_projection_keeps_silu_and_qsilu_contract() -> None:
    generator = _load_generator()
    rows = {row["activation"]: row for row in generator._operation_rows()}

    assert "silu" in rows
    assert "relu" not in rows
    assert rows["qsilu_pq"]["variable_multiplications"] == 1
    assert rows["qsilu_pq"]["fused_selected_constant_products"] == 2
    assert rows["qsilu_pq"]["detect_variable_multiplications_per_image"] == 71_577_600
    assert rows["poly_shift"]["pose_variable_multiplications_per_image"] == 290_611_200
