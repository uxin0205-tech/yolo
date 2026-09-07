from __future__ import annotations

import copy
import importlib
from pathlib import Path

import torch
from torch import nn
from ultralytics.nn.modules import Conv

from yolo_quantize.full35_adapter import (
    Full35ActivationAdapter,
    Full35ActivationPolicy,
)
from yolo_quantize.qat_graph import (
    BNFoldedHardwareContractGuard,
    fold_batch_norm_only,
    prepare_folded_qat_training_graph,
)
from yolo_quantize.qat_runtime import _save_qat_deployment_inference_weights
from yolo_quantize.qat_validation import build_qat_deployment_view
from yolo_quantize.qat_weights import FoldedQATWeightAdapter, QATConv2d
from yolo_quantize.validation_source import Full35DeploymentValidationSource
from yolo_quantize.weight_quantization import (
    Full35WeightRegionCatalog,
    UniformWeightSpec,
    WeightRegionAssignment,
)


class _FuseTrapHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.one2many_cv2 = nn.ModuleList([nn.Conv2d(4, 4, 1)])
        self.fuse_called = False

    def fuse(self) -> None:
        self.fuse_called = True
        del self.one2many_cv2


class _ToyTrainingGraph(nn.Module):
    def __init__(self, *, with_head: bool = True) -> None:
        super().__init__()
        self.graph = nn.Module()
        layers: list[nn.Module] = [Conv(3, 4, 3)]
        if with_head:
            layers.append(_FuseTrapHead())
        self.graph.model = nn.ModuleList(layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.graph.model[0](value)


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Module()
        self.qkv.q = nn.Linear(2, 2)
        self.qkv.k = nn.Linear(2, 2)
        self.score = nn.Module()
        self.score.gamma = nn.Parameter(torch.tensor(1.0))
        self.score.register_buffer("fixed_coefficients", torch.ones(2))
        self.normalize = nn.Module()
        self.normalize.register_buffer("knots", torch.arange(2.0))
        self.normalize.register_buffer("values", torch.arange(2.0))


class _ToyHardwareGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = nn.Module()
        self.graph.model = nn.ModuleList([nn.Module(), nn.Module()])
        self.graph.model[0].attn = _ToyAttention()
        self.graph.model[1].attn = _ToyAttention()


def test_bn_only_fold_preserves_training_head_and_forward() -> None:
    torch.manual_seed(11)
    model = _ToyTrainingGraph().eval()
    reference = copy.deepcopy(model)
    sample = torch.randn(2, 3, 16, 16)
    expected = reference(sample)
    head = model.graph.model[1]
    assert isinstance(head, _FuseTrapHead)
    one2many_weight = head.one2many_cv2[0].weight.detach().clone()

    report = fold_batch_norm_only(model)

    assert report.initial_batch_norm_modules == 1
    assert report.final_batch_norm_modules == 0
    assert report.folded_conv_batch_norm_modules == 1
    assert not head.fuse_called
    assert hasattr(head, "one2many_cv2")
    assert torch.equal(head.one2many_cv2[0].weight, one2many_weight)
    assert torch.allclose(model(sample), expected, atol=1e-5, rtol=1e-5)


def test_prepare_training_graph_keeps_catalog_and_applies_qat_after_fold() -> None:
    model = _ToyTrainingGraph(with_head=False).eval()
    before = Full35WeightRegionCatalog.inspect(model)
    assignments = (
        WeightRegionAssignment(
            region="backbone_early",
            spec=UniformWeightSpec(bits=8),
        ),
    )

    prepared = prepare_folded_qat_training_graph(
        model,
        expected_catalog=before.summary(),
        assignments=assignments,
    )

    assert prepared.fold_report.final_batch_norm_modules == 0
    assert prepared.catalog.summary() == before.summary()
    assert prepared.policy.quantized_modules == 1
    assert isinstance(model.get_submodule("graph.model.0.conv"), QATConv2d)
    assert prepared.policy.blend_ratio == 0.0


def test_bn_only_fold_refuses_unknown_remaining_batch_norm() -> None:
    model = _ToyTrainingGraph(with_head=False)
    model.extra_bn = nn.BatchNorm2d(4)

    try:
        fold_batch_norm_only(model)
    except RuntimeError as error:
        assert "retains 1 BatchNorm" in str(error)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("unknown BatchNorm must fail closed")


def test_bn_folded_hardware_guard_freezes_and_checks_exact_state() -> None:
    model = _ToyHardwareGraph()
    guard = BNFoldedHardwareContractGuard.capture(model)

    assert len(guard.paths) == 16
    assert not model.graph.model[0].attn.qkv.q.weight.requires_grad
    assert not model.graph.model[1].attn.score.gamma.requires_grad
    guard.assert_unchanged(model)

    model.graph.model[0].attn.normalize.knots.add_(1.0)
    try:
        guard.assert_unchanged(model)
    except AssertionError as error:
        assert "hardware-contract state changed" in str(error)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("immutable PWL state drift must fail")


def test_real_full35_bn_only_fold_preserves_all_training_only_weights(
    tmp_path: Path,
) -> None:
    built = Full35ActivationAdapter().build(
        Full35ActivationPolicy(activation="poly_shift", bits=8),
        checkpoint=(
            "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
            "short-recovery-v2-lr01-uniform-poly-shift-seed1/"
            "inference/best_joint.pt"
        ),
        checkpoint_sha256=(
            "8783248a513329ae80e3f659e6f3ce617e4bcca9fca9530b1f95448de5e19713"
        ),
    )
    model = built.model.eval()
    before = Full35WeightRegionCatalog.inspect(model)
    training_paths = tuple(site.path for site in before.training_only_sites)

    report = fold_batch_norm_only(model)
    after = Full35WeightRegionCatalog.inspect(model)

    assert report.initial_batch_norm_modules == 179
    assert report.final_batch_norm_modules == 0
    assert before.summary() == after.summary()
    assert len(training_paths) == 99
    assert tuple(site.path for site in after.training_only_sites) == training_paths
    assert hasattr(model.detect_head, "cv2")
    assert hasattr(model.detect_head, "one2one_cv2")
    assert hasattr(model.pose_head, "cv2")
    assert hasattr(model.pose_head, "one2one_cv2")

    policy = FoldedQATWeightAdapter().apply(
        model,
        catalog=after,
        assignments=(
            WeightRegionAssignment(
                region="neck",
                spec=UniformWeightSpec(bits=8),
            ),
        ),
    )
    view = build_qat_deployment_view(
        shared_ema=model,
        source=built.source,
        activation_policy=built.applied,
        weight_policy=policy,
    )
    deployment = view.model
    deployment_catalog = Full35WeightRegionCatalog.inspect(deployment)

    assert len(deployment_catalog.deployment_sites) == 148
    assert len(deployment_catalog.training_only_sites) == 0
    assert not any(isinstance(module, QATConv2d) for module in deployment.modules())
    assert isinstance(view.source, Full35DeploymentValidationSource)
    assert view.activation_policy.quantizer_count == built.applied.quantizer_count
    assert deployment.detect_head.cv2 is None
    assert deployment.pose_head.cv2 is None

    destination = tmp_path / "full35-deployment.pt"
    _save_qat_deployment_inference_weights(
        importlib.import_module("yolo_combine.resume"),
        destination,
        deployment_model=deployment,
        contract_source=model,
        metadata={"test": "real-full35-fused-head"},
    )
    payload = torch.load(destination, map_location="cpu", weights_only=True)

    assert payload["contract"] == model.contract()
    assert payload["state_dict"].keys() == deployment.state_dict().keys()
    assert not any("weight_quantizer" in name for name in payload["state_dict"])
