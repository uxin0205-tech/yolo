from __future__ import annotations

import copy

import pytest

from yolo_quantize.ternary_routing import (
    PaperTWNRouting,
    TernaryRoutingSource,
)

_PARENTS = {
    "qsilu_pq": "1" * 64,
    "hardswish": "2" * 64,
    "poly_shift": "3" * 64,
}


def _measurement(
    *, view: str, path: str, nrmse: float, cosine: float
) -> dict[str, object]:
    return {
        "view": view,
        "path": path,
        "region": "pose_one2one_predictor",
        "family": "paper_twn",
        "format_id": "paper-twn-layerwise",
        "bits": 2,
        "granularity": "per_tensor",
        "scale_method": "paper_delta_0.7_mean_abs",
        "elements": 512,
        "scale_count": 1,
        "code_bytes": 128,
        "metadata_bytes": 4,
        "numeric": {
            "normalized_rmse": nrmse,
            "cosine": cosine,
            "zero_ratio": 0.6,
        },
    }


def _source(parent: str) -> TernaryRoutingSource:
    measurements = []
    for view in ("master", "deployment"):
        measurements.extend(
            (
                _measurement(view=view, path="safe", nrmse=0.34, cosine=0.94),
                _measurement(view=view, path="balanced", nrmse=0.39, cosine=0.92),
                _measurement(view=view, path="reject", nrmse=0.45, cosine=0.88),
            )
        )
    return TernaryRoutingSource(
        parent_name=parent,
        checkpoint_sha256=_PARENTS[parent],
        artifact_path=f"artifacts/{parent}.json",
        artifact_sha256="4" * 64,
        payload={
            "kind": "full35_static_weight_format_analysis",
            "profile": "main",
            "gpu_used": False,
            "formal_training": False,
            "parent": {
                "activation": parent,
                "checkpoint_sha256": _PARENTS[parent],
            },
            "measurements": measurements,
        },
    )


def test_paper_twn_routing_requires_all_parents_and_both_weight_views() -> None:
    manifest = PaperTWNRouting().build(tuple(_source(name) for name in _PARENTS))

    assert manifest.safe_candidates == ("safe",)
    assert manifest.balanced_candidates == ("balanced", "safe")
    assert manifest.counts["safe"] == 1
    assert manifest.counts["balanced"] == 2
    assert manifest.thresholds == {
        "safe": {"maximum_normalized_rmse": 0.35, "minimum_cosine": 0.93},
        "balanced": {"maximum_normalized_rmse": 0.4, "minimum_cosine": 0.91},
    }
    assert manifest.execution_authorized is False
    assert manifest.map_validation_run is False


def test_paper_twn_routing_fails_on_incomplete_cross_view_matrix() -> None:
    sources = [_source(name) for name in _PARENTS]
    payload = copy.deepcopy(sources[-1].payload)
    payload["measurements"].pop()  # type: ignore[union-attr]
    sources[-1] = TernaryRoutingSource(
        parent_name=sources[-1].parent_name,
        checkpoint_sha256=sources[-1].checkpoint_sha256,
        artifact_path=sources[-1].artifact_path,
        artifact_sha256=sources[-1].artifact_sha256,
        payload=payload,
    )

    with pytest.raises(ValueError, match="incomplete Paper-TWN matrix"):
        PaperTWNRouting().build(tuple(sources))
