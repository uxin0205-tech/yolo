from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_quantize.exact_preparation import (
    ExactPreparationLayout,
    ExactWeightPreparation,
)


def test_exact_preparation_uses_only_the_fair_w4_sd4_matrix() -> None:
    plan = ExactWeightPreparation.analysis_plan()

    assert plan.view_names == ("master", "deployment")
    assert plan.uniform_bits == (4,)
    assert plan.uniform_granularities == ("per_output_channel",)
    assert plan.uniform_scale_methods == (
        "mse_grid_v1",
        "optimal_scaled_codebook",
    )
    assert plan.fixed_sd4_granularities == ("per_output_channel",)
    assert plan.fixed_sd4_scale_methods == (
        "mse_grid_v1",
        "optimal_scaled_codebook",
    )
    assert plan.include_fixed_sd4 is True
    assert plan.include_paper_twn is False


def test_exact_preparation_layout_is_versioned_and_keeps_historical_v2() -> None:
    layout = ExactPreparationLayout.default(Path("/tmp/example-project"))

    assert layout.report_paths == {
        "qsilu_pq": Path(
            "/tmp/example-project/artifacts/reports/"
            "weight-format-analysis-qsilu-pq-a8-exact-w4-sd4-v1.json"
        ),
        "hardswish": Path(
            "/tmp/example-project/artifacts/reports/"
            "weight-format-analysis-hardswish-a8-exact-w4-sd4-v1.json"
        ),
        "poly_shift": Path(
            "/tmp/example-project/artifacts/reports/"
            "weight-format-analysis-poly-shift-a8-exact-w4-sd4-v1.json"
        ),
    }
    assert layout.routing_path.name == "fixed-sd4-routing-candidates-v3.yaml"
    assert layout.boundary_path.name == (
        "full35-integer-boundary-contract-qsilu-pq-a8-v1.json"
    )
    assert layout.delivery_path.name == "exact-w4-sd4-cpu-delivery-v1.yaml"


def test_exact_preparation_refuses_to_overwrite_an_artifact(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    output.write_text('{"historical": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ExactWeightPreparation.write_new_json(output, {"replacement": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"historical": True}
