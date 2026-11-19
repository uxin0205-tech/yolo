"""Small external interface for the deep SIPA-BCSP training module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .domain import (
    ActivationManifest,
    DeliveryAudit,
    PolicyObservation,
    RegionRule,
    StaticPolicy,
    TrainingConfig,
    TrainingPlan,
)
from .io import audit_delivery_mapping, load_training_config
from .model import AppliedPolicy, apply_static_policy, inspect_silu_sites
from .search import _validate_policy, compile_next_plan


class TrainingArchitecture:
    """Inspect, replace, and plan through one manifest-driven seam."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str) -> TrainingArchitecture:
        return cls(load_training_config(path))

    def inspect(
        self,
        model: nn.Module,
        *,
        model_id: str,
        region_rules: tuple[RegionRule, ...],
    ) -> ActivationManifest:
        return inspect_silu_sites(model, model_id=model_id, region_rules=region_rules)

    def apply(
        self,
        model: nn.Module,
        manifest: ActivationManifest,
        policy: StaticPolicy,
        *,
        clone_model: bool = True,
    ) -> AppliedPolicy:
        _validate_policy(manifest, policy, self.config.search)
        return apply_static_policy(
            model,
            manifest,
            policy,
            clone_model=clone_model,
        )

    def plan(
        self,
        manifest: ActivationManifest,
        observations: tuple[PolicyObservation, ...] = (),
    ) -> TrainingPlan:
        return compile_next_plan(manifest, self.config, observations)

    def audit_delivery(
        self,
        delivery: Mapping[str, Any],
        *,
        check_files: bool = True,
    ) -> DeliveryAudit:
        return audit_delivery_mapping(delivery, self.config, check_files=check_files)
