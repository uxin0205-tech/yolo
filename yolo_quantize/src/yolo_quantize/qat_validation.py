"""EMA-to-deployment projection for official QAT validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from .activation_adapter import AppliedActivationQuantization
from .qat_graph import materialize_qat_deployment_graph
from .qat_weights import AppliedFoldedQATWeightPolicy
from .validation_source import Full35DeploymentValidationSource


@dataclass(frozen=True)
class QATDeploymentView:
    """Plain full-strength weight graph plus activation-aware task templates."""

    model: nn.Module
    source: Full35DeploymentValidationSource
    activation_policy: AppliedActivationQuantization
    quantized_weight_modules: int


class QATJointValidatorAdapter:
    """Materialize a full-strength EMA clone before each official backend."""

    def __init__(
        self,
        source: Any,
        *,
        activation_policy: Any,
        weight_policy: Any,
        expected_catalog: Mapping[str, object],
        weight_blend_ratio: float = 1.0,
        upstream_validator_cls: type,
        view_builder: Callable[..., Any] | None = None,
        **validator_kwargs: Any,
    ) -> None:
        self.source = source
        self.activation_policy = activation_policy
        self.weight_policy = weight_policy
        self.expected_catalog = expected_catalog
        self.weight_blend_ratio = float(weight_blend_ratio)
        self.upstream_validator_cls = upstream_validator_cls
        self.view_builder = build_qat_deployment_view if view_builder is None else view_builder
        self.validator_kwargs = dict(validator_kwargs)

    def validate(self, shared_ema: nn.Module, *, epoch: int, kind: str) -> Any:
        view = self.view_builder(
            shared_ema=shared_ema,
            source=self.source,
            activation_policy=self.activation_policy,
            weight_policy=self.weight_policy,
            expected_catalog=self.expected_catalog,
            weight_blend_ratio=self.weight_blend_ratio,
        )
        validator = self.upstream_validator_cls(
            view.source,
            **self.validator_kwargs,
        )
        return validator.validate(view.model, epoch=epoch, kind=kind)

    def validate_backends(
        self,
        shared_ema: nn.Module,
        *,
        epoch: int,
        kinds: Sequence[str] = ("bittrue",),
    ) -> dict[str, Any]:
        if not kinds or len(set(kinds)) != len(kinds):
            raise ValueError("QAT validation backends must be non-empty and unique")
        return {
            kind: self.validate(shared_ema, epoch=epoch, kind=kind) for kind in kinds
        }


def build_qat_deployment_view(
    *,
    shared_ema: nn.Module,
    source: Any,
    activation_policy: AppliedActivationQuantization,
    weight_policy: AppliedFoldedQATWeightPolicy,
    expected_catalog: Mapping[str, object] | None = None,
    weight_blend_ratio: float = 1.0,
) -> QATDeploymentView:
    """Materialize EMA at ratio=1 without mutating the live/EMA training graph."""

    rebound_weight = weight_policy.rebind(shared_ema)
    deployment = materialize_qat_deployment_graph(
        rebound_weight,
        expected_catalog=expected_catalog,
        blend_ratio=weight_blend_ratio,
    )
    rebound_activation = activation_policy.rebind(deployment)
    return QATDeploymentView(
        model=deployment,
        source=Full35DeploymentValidationSource(source, rebound_activation),
        activation_policy=rebound_activation,
        quantized_weight_modules=rebound_weight.quantized_modules,
    )


__all__ = (
    "QATDeploymentView",
    "QATJointValidatorAdapter",
    "build_qat_deployment_view",
)
