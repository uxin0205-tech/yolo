"""BN-only folding seam for QAT that preserves Full35 training branches."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from ultralytics.nn.modules import (
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    RepConv,
    RepVGGDW,
)
from ultralytics.utils.torch_utils import fuse_conv_and_bn, fuse_deconv_and_bn

from .qat_weights import (
    AppliedFoldedQATWeightPolicy,
    FoldedQATWeightAdapter,
    QATConv2d,
    QATLinear,
)
from .weight_quantization import (
    Full35WeightRegionCatalog,
    WeightRegionAssignment,
)

_IMMUTABLE_PARAMETER_PARTS = (
    ".attn.qkv.q.",
    ".attn.qkv.k.",
    ".attn.score.gamma",
)
_IMMUTABLE_BUFFER_PARTS = (
    ".attn.score.fixed_coefficients",
    ".attn.score.calibration_",
    ".attn.normalize.knots",
    ".attn.normalize.values",
)
_REQUIRED_IMMUTABLE_SUFFIXES = (
    "score.fixed_coefficients",
    "normalize.knots",
    "normalize.values",
    "score.gamma",
)


@dataclass(frozen=True)
class BNOnlyFoldReport:
    """Evidence that only normalization/reparameterization sites were folded."""

    initial_batch_norm_modules: int
    final_batch_norm_modules: int
    folded_conv_batch_norm_modules: int
    folded_deconv_batch_norm_modules: int
    reparameterized_repconv_modules: int
    reparameterized_repvggdw_modules: int

    @property
    def folded_modules(self) -> int:
        return (
            self.folded_conv_batch_norm_modules
            + self.folded_deconv_batch_norm_modules
            + self.reparameterized_repconv_modules
            + self.reparameterized_repvggdw_modules
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "initial_batch_norm_modules": self.initial_batch_norm_modules,
            "final_batch_norm_modules": self.final_batch_norm_modules,
            "folded_conv_batch_norm_modules": self.folded_conv_batch_norm_modules,
            "folded_deconv_batch_norm_modules": self.folded_deconv_batch_norm_modules,
            "reparameterized_repconv_modules": self.reparameterized_repconv_modules,
            "reparameterized_repvggdw_modules": self.reparameterized_repvggdw_modules,
            "folded_modules": self.folded_modules,
        }


def _batch_norm_count(model: nn.Module) -> int:
    return sum(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )


def _immutable_hardware_state(model: nn.Module) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS):
            selected[name] = parameter
    for name, buffer in model.named_buffers():
        if any(part in name for part in _IMMUTABLE_BUFFER_PARTS):
            selected[name] = buffer
    return selected


@dataclass(frozen=True)
class BNFoldedHardwareContractGuard:
    """Protect Binary Q/K and PWL state when the training graph has no BN."""

    state: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: nn.Module) -> BNFoldedHardwareContractGuard:
        retained_bn = _batch_norm_count(model)
        if retained_bn:
            raise ValueError(
                f"BN-folded hardware guard found {retained_bn} BatchNorm modules"
            )
        selected = _immutable_hardware_state(model)
        for suffix in _REQUIRED_IMMUTABLE_SUFFIXES:
            matches = tuple(name for name in selected if name.endswith(suffix))
            if len(matches) != 2:
                raise ValueError(
                    f"expected two immutable states ending {suffix}, got {matches}"
                )
        for name, parameter in model.named_parameters():
            if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS):
                parameter.requires_grad_(False)
        return cls(
            state={
                name: value.detach().cpu().clone()
                for name, value in selected.items()
            }
        )

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(self.state)

    def assert_unchanged(self, model: nn.Module) -> None:
        retained_bn = _batch_norm_count(model)
        if retained_bn:
            raise AssertionError(
                f"BN-folded QAT graph regained {retained_bn} BatchNorm modules"
            )
        selected = _immutable_hardware_state(model)
        if set(selected) != set(self.state):
            missing = sorted(set(self.state) - set(selected))
            unexpected = sorted(set(selected) - set(self.state))
            raise AssertionError(
                f"hardware state keys changed; missing={missing}, "
                f"unexpected={unexpected}"
            )
        changed = tuple(
            name
            for name, value in selected.items()
            if not torch.equal(value.detach().cpu(), self.state[name])
        )
        if changed:
            raise AssertionError(f"hardware-contract state changed: {changed[:20]}")
        unfrozen = tuple(
            name
            for name, parameter in model.named_parameters()
            if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS)
            and parameter.requires_grad
        )
        if unfrozen:
            raise AssertionError(
                f"hardware-contract parameters became trainable: {unfrozen}"
            )


def fold_batch_norm_only(model: nn.Module) -> BNOnlyFoldReport:
    """Fold Ultralytics normalization blocks without calling Detect/Pose fuse()."""

    initial_bn = _batch_norm_count(model)
    conv_bn = 0
    deconv_bn = 0
    repconv = 0
    repvggdw = 0
    for module in tuple(model.modules()):
        if isinstance(module, (Conv, Conv2, DWConv)) and hasattr(module, "bn"):
            if isinstance(module, Conv2):
                module.fuse_convs()
            module.conv = fuse_conv_and_bn(module.conv, module.bn)
            delattr(module, "bn")
            module.forward = module.forward_fuse
            conv_bn += 1
        elif isinstance(module, ConvTranspose) and hasattr(module, "bn"):
            module.conv_transpose = fuse_deconv_and_bn(
                module.conv_transpose,
                module.bn,
            )
            delattr(module, "bn")
            module.forward = module.forward_fuse
            deconv_bn += 1
        elif isinstance(module, RepConv) and hasattr(module, "conv1"):
            module.fuse_convs()
            module.forward = module.forward_fuse
            repconv += 1
        elif isinstance(module, RepVGGDW) and hasattr(module, "conv"):
            module.fuse()
            module.forward = module.forward_fuse
            repvggdw += 1

    final_bn = _batch_norm_count(model)
    if final_bn:
        raise RuntimeError(
            f"BN-only QAT fold retains {final_bn} BatchNorm modules; "
            "an unsupported normalization seam exists"
        )
    if initial_bn and conv_bn + deconv_bn + repconv + repvggdw == 0:
        raise RuntimeError(
            "BN-only QAT fold found BatchNorm but folded no known modules"
        )
    return BNOnlyFoldReport(
        initial_batch_norm_modules=initial_bn,
        final_batch_norm_modules=final_bn,
        folded_conv_batch_norm_modules=conv_bn,
        folded_deconv_batch_norm_modules=deconv_bn,
        reparameterized_repconv_modules=repconv,
        reparameterized_repvggdw_modules=repvggdw,
    )


@dataclass(frozen=True)
class PreparedFoldedQATTrainingGraph:
    """One trainable BN-folded graph with one reviewed weight-QAT policy."""

    model: nn.Module
    catalog: Full35WeightRegionCatalog
    fold_report: BNOnlyFoldReport
    policy: AppliedFoldedQATWeightPolicy


def prepare_folded_qat_training_graph(
    model: nn.Module,
    *,
    expected_catalog: Mapping[str, object],
    assignments: tuple[WeightRegionAssignment, ...],
) -> PreparedFoldedQATTrainingGraph:
    """Fold BN only, prove catalog stability, then install persistent QAT wrappers."""

    before = Full35WeightRegionCatalog.inspect(model)
    if before.summary() != expected_catalog:
        raise RuntimeError(
            "QAT training graph differs from the reviewed master catalog"
        )
    fold_report = fold_batch_norm_only(model)
    after = Full35WeightRegionCatalog.inspect(model)
    if after.summary() != expected_catalog:
        raise RuntimeError("BN-only folding changed the reviewed Full35 weight catalog")
    policy = FoldedQATWeightAdapter().apply(
        model,
        catalog=after,
        assignments=assignments,
    )
    return PreparedFoldedQATTrainingGraph(
        model=model,
        catalog=after,
        fold_report=fold_report,
        policy=policy,
    )


def materialize_qat_deployment_graph(
    policy: AppliedFoldedQATWeightPolicy,
    *,
    expected_catalog: Mapping[str, object] | None = None,
    blend_ratio: float = 1.0,
) -> nn.Module:
    """Project a full-strength QAT clone and remove only training-time head paths."""

    working = copy.deepcopy(policy.model)
    rebound = policy.rebind(working)
    rebound.set_blend_ratio(blend_ratio)
    target = rebound.materialize(
        clone_model=False,
        expected_blend_ratio=blend_ratio,
    )
    for name in ("detect_head", "pose_head"):
        head = getattr(target, name, None)
        fuse = getattr(head, "fuse", None)
        if not callable(fuse) or not bool(getattr(head, "end2end", False)):
            raise TypeError(f"QAT deployment model has no end-to-end {name}")
        fuse()
    target.eval()
    retained_bn = _batch_norm_count(target)
    if retained_bn:
        raise RuntimeError(
            f"QAT deployment materialization retains {retained_bn} BatchNorm modules"
        )
    if any(isinstance(module, (QATConv2d, QATLinear)) for module in target.modules()):
        raise RuntimeError("QAT deployment materialization retains fake-quant modules")
    catalog = Full35WeightRegionCatalog.inspect(target)
    if catalog.training_only_sites:
        raise RuntimeError("QAT deployment materialization retains training-only weights")
    if expected_catalog is not None and catalog.summary() != expected_catalog:
        raise RuntimeError("QAT deployment materialization catalog differs")
    return target


__all__ = (
    "BNFoldedHardwareContractGuard",
    "BNOnlyFoldReport",
    "PreparedFoldedQATTrainingGraph",
    "fold_batch_norm_only",
    "materialize_qat_deployment_graph",
    "prepare_folded_qat_training_graph",
)
