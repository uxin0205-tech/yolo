from __future__ import annotations

import torch
from torch import nn

from yolo_quantize import (
    Full35ActivationAdapter,
    Full35ActivationPolicy,
    Full35DeploymentValidationSource,
    Full35WeightRegionCatalog,
    Full35WeightViewAdapter,
    UniformWeightSpec,
    WeightQuantizationAdapter,
)


def test_deployment_validation_source_preserves_calibrated_graph_shape() -> None:
    built = Full35ActivationAdapter().build(
        Full35ActivationPolicy(activation="qsilu_pq", bits=8),
        checkpoint=(
            "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
            "short-recovery-v2-lr01-uniform-qsilu-pq-seed1/"
            "inference/best_joint.pt"
        ),
        checkpoint_sha256=(
            "7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e"
        ),
    )
    views = Full35WeightViewAdapter().build(built.model)
    applied = built.applied.rebind(views.deployment)
    source = Full35DeploymentValidationSource(built.source, applied)

    from yolo_combine.graph_materialize import build_graph_validation_models

    materialized = build_graph_validation_models(
        views.deployment,
        source,
        kind="bittrue",
    )

    assert materialized.detect_report.complete
    assert materialized.pose_report.complete
    assert (
        sum(
            type(module).__name__ == "_ObservedQuantizedActivation"
            for module in materialized.detect.modules()
        )
        > 0
    )
    assert (
        sum(
            type(module).__name__ == "_ObservedQuantizedActivation"
            for module in materialized.pose.modules()
        )
        > 0
    )
    assert not any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for model in (materialized.detect, materialized.pose)
        for module in model.modules()
    )

    catalog = Full35WeightRegionCatalog.inspect(views.deployment)
    site = next(
        item for item in catalog.deployment_sites if item.region == "backbone_early"
    )
    source_module = views.deployment.get_submodule(site.path)
    original = source_module.weight.detach().clone()
    with WeightQuantizationAdapter().quantized(
        views.deployment,
        catalog=catalog,
        region="backbone_early",
        spec=UniformWeightSpec(bits=8, scale_method="mse_grid_v1"),
    ):
        quantized = source_module.weight.detach().clone()
        candidate = build_graph_validation_models(
            views.deployment,
            source,
            kind="bittrue",
        )
        target = candidate.detect.get_submodule(site.path.removeprefix("graph."))
        assert not torch.equal(original, quantized)
        assert torch.equal(target.weight, quantized)
    assert torch.equal(source_module.weight, original)
