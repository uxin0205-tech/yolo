from __future__ import annotations

import torch
from torch import nn

from yolo_quantize import (
    Full35ActivationAdapter,
    Full35ActivationPolicy,
    Full35WeightViewAdapter,
)


def test_full35_weight_views_preserve_master_and_fold_deployment_graph() -> None:
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
    source = built.model

    views = Full35WeightViewAdapter().build(source)
    rebound = built.applied.rebind(views.deployment)

    assert {
        "distinct_objects": len({id(source), id(views.master), id(views.deployment)}),
        "source_unchanged": views.manifest.source_unchanged,
        "master_matches_source": views.manifest.master_matches_source,
        "source_bn": views.manifest.source_batch_norm_modules,
        "master_bn": views.manifest.master_batch_norm_modules,
        "deployment_bn": views.manifest.deployment_batch_norm_modules,
        "deployment_paths_match": views.manifest.deployment_path_parity,
        "inference_contract_matches": views.manifest.inference_contract_parity,
        "forward_parity_evaluated": views.manifest.forward_parity_evaluated,
        "forward_same_structure": views.manifest.forward_same_structure,
        "forward_all_finite": views.manifest.forward_all_finite,
        "forward_parity_passed": views.manifest.forward_parity_passed,
        "master_deployment_modules": (
            views.manifest.master_catalog["totals"]["deployment_modules"]
        ),
        "deployment_modules": (
            views.manifest.deployment_catalog["totals"]["deployment_modules"]
        ),
        "deployment_training_only_modules": (
            views.manifest.deployment_catalog["totals"]["training_only_modules"]
        ),
        "master_is_fp32": all(
            tensor.dtype.is_floating_point is False or tensor.dtype == torch.float32
            for tensor in views.master.state_dict().values()
        ),
        "deployment_is_fp32": all(
            tensor.dtype.is_floating_point is False or tensor.dtype == torch.float32
            for tensor in views.deployment.state_dict().values()
        ),
        "master_bn_runtime": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in views.master.modules()
        ),
        "deployment_bn_runtime": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in views.deployment.modules()
        ),
        "rebound_model": rebound.model is views.deployment,
        "rebound_quantizers": rebound.quantizer_count,
        "rebound_mode": rebound.mode,
    } == {
        "distinct_objects": 3,
        "source_unchanged": True,
        "master_matches_source": True,
        "source_bn": 179,
        "master_bn": 179,
        "deployment_bn": 0,
        "deployment_paths_match": True,
        "inference_contract_matches": True,
        "forward_parity_evaluated": True,
        "forward_same_structure": True,
        "forward_all_finite": True,
        "forward_parity_passed": True,
        "master_deployment_modules": 148,
        "deployment_modules": 148,
        "deployment_training_only_modules": 0,
        "master_is_fp32": True,
        "deployment_is_fp32": True,
        "master_bn_runtime": 179,
        "deployment_bn_runtime": 0,
        "rebound_model": True,
        "rebound_quantizers": 124,
        "rebound_mode": "observe",
    }
