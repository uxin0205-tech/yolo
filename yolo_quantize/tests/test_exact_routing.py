from __future__ import annotations

import copy

import pytest

from yolo_quantize.exact_routing import ExactRoutingSource, ExactWeightRouting

_PARENTS = {
    "qsilu_pq": "1" * 64,
    "hardswish": "2" * 64,
    "poly_shift": "3" * 64,
}
_ARTIFACT_SHA = {"qsilu_pq": "4" * 64, "hardswish": "5" * 64, "poly_shift": "6" * 64}
_FORMATS = (
    "uniform-w4-per_output_channel-mse_grid_v1",
    "uniform-w4-per_output_channel-optimal_scaled_codebook",
    "fixed-sd4-per_output_channel-mse_grid_v1",
    "fixed-sd4-per_output_channel-optimal_scaled_codebook",
)


def _measurement(
    *,
    view: str,
    path: str,
    region: str,
    format_id: str,
    mse: float,
) -> dict[str, object]:
    family = "fixed_sd4" if format_id.startswith("fixed-sd4") else "uniform"
    method = format_id.rsplit("-", 1)[-1]
    return {
        "view": view,
        "path": path,
        "region": region,
        "family": family,
        "format_id": format_id,
        "bits": 4,
        "granularity": "per_output_channel",
        "scale_method": method,
        "elements": 10,
        "scale_count": 2,
        "code_bytes": 5,
        "metadata_bytes": 8,
        "distribution": {
            "mean": 0.0,
            "standard_deviation": 1.0,
        },
        "numeric": {
            "mse": mse,
            "normalized_rmse": mse**0.5,
        },
    }


def _source(parent: str) -> ExactRoutingSource:
    measurements: list[dict[str, object]] = []
    for view in ("master", "deployment"):
        for path, region in (("layer.a", "backbone_early"), ("layer.b", "neck")):
            exact_control = 0.10 if path == "layer.a" else 0.20
            exact_candidate = 0.08 if path == "layer.a" else 0.19
            if parent == "poly_shift" and view == "deployment" and path == "layer.b":
                exact_candidate = 0.21
            values = {
                _FORMATS[0]: exact_control + 0.02,
                _FORMATS[1]: exact_control,
                _FORMATS[2]: exact_candidate + 0.03,
                _FORMATS[3]: exact_candidate,
            }
            measurements.extend(
                _measurement(
                    view=view,
                    path=path,
                    region=region,
                    format_id=format_id,
                    mse=mse,
                )
                for format_id, mse in values.items()
            )
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "full35_exact_w4_sd4_static_analysis",
        "profile": "exact_w4_sd4_v1",
        "gpu_used": False,
        "formal_training": False,
        "parent": {
            "name": parent,
            "policy_id": f"{parent}--lsq-plus-a8",
            "checkpoint_sha256": _PARENTS[parent],
        },
        "analysis_plan": {
            "views": ["master", "deployment"],
            "formats": list(_FORMATS),
        },
        "measurements": measurements,
    }
    return ExactRoutingSource(
        parent_name=parent,
        checkpoint_sha256=_PARENTS[parent],
        artifact_path=f"artifacts/{parent}.json",
        artifact_sha256=_ARTIFACT_SHA[parent],
        payload=payload,
    )


def test_exact_routing_requires_cross_parent_and_cross_view_wins() -> None:
    manifest = ExactWeightRouting().build(tuple(_source(name) for name in _PARENTS))

    assert manifest.stable_candidates == ("layer.a",)
    assert manifest.counts["both_view_cross_parent_winners"] == 1
    assert manifest.counts["regions"] == {"backbone_early": 1}
    assert manifest.comparison == {
        "candidate": "fixed-sd4-per_output_channel-optimal_scaled_codebook",
        "control": "uniform-w4-per_output_channel-optimal_scaled_codebook",
        "equal_code_bits": 4,
        "equal_scale_granularity": "per_output_channel",
        "selection_metric": "per_layer_weight_reconstruction_mse",
        "required_parents": ["qsilu_pq", "hardswish", "poly_shift"],
        "required_views": ["master", "deployment"],
        "rule": (
            "candidate_mse_strictly_less_than_control_in_every_active_parent_and_view"
        ),
    }
    assert all(
        value["hybrid_nrmse"] < value["all_w4_nrmse"]
        for value in manifest.static_proxy_effect.values()
    )
    assert manifest.exact_solver_audit["uniform_exact_never_worse_than_grid"] is True
    assert manifest.exact_solver_audit["fixed_sd4_exact_never_worse_than_grid"] is True
    assert manifest.execution_authorized is False
    assert manifest.gpu_used is False


def test_exact_routing_fails_closed_when_one_required_cell_is_missing() -> None:
    sources = [_source(name) for name in _PARENTS]
    payload = copy.deepcopy(sources[-1].payload)
    payload["measurements"].pop()  # type: ignore[union-attr]
    sources[-1] = ExactRoutingSource(
        parent_name=sources[-1].parent_name,
        checkpoint_sha256=sources[-1].checkpoint_sha256,
        artifact_path=sources[-1].artifact_path,
        artifact_sha256=sources[-1].artifact_sha256,
        payload=payload,
    )

    with pytest.raises(ValueError, match="incomplete exact routing matrix"):
        ExactWeightRouting().build(tuple(sources))


def test_exact_routing_rejects_a_grid_result_better_than_claimed_exact() -> None:
    sources = [_source(name) for name in _PARENTS]
    payload = copy.deepcopy(sources[0].payload)
    measurements = payload["measurements"]
    assert isinstance(measurements, list)
    exact = next(
        item
        for item in measurements
        if item["format_id"] == "fixed-sd4-per_output_channel-optimal_scaled_codebook"
    )
    grid = next(
        item
        for item in measurements
        if item["format_id"] == "fixed-sd4-per_output_channel-mse_grid_v1"
    )
    exact["numeric"]["mse"] = grid["numeric"]["mse"] + 1.0  # type: ignore[index]
    sources[0] = ExactRoutingSource(
        parent_name=sources[0].parent_name,
        checkpoint_sha256=sources[0].checkpoint_sha256,
        artifact_path=sources[0].artifact_path,
        artifact_sha256=sources[0].artifact_sha256,
        payload=payload,
    )

    with pytest.raises(ValueError, match="exact solver is worse than grid"):
        ExactWeightRouting().build(tuple(sources))
