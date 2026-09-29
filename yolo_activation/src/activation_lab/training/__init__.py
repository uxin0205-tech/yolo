"""Public interface for manifest-driven SIPA-BCSP training architecture."""

from .architecture import TrainingArchitecture
from .domain import (
    ActivationManifest,
    ActivationSite,
    DatasetContract,
    DeliveryAudit,
    ExperimentSpec,
    PolicyCost,
    PolicyObservation,
    RegionRule,
    SearchConfig,
    StaticPolicy,
    TrainingConfig,
    TrainingPlan,
)
from .full35 import (
    Full35ActivationExperiment,
    Full35ExperimentConfig,
    Full35LoadedPolicy,
    Full35Phase,
    Full35PreflightReport,
    load_full35_manifest,
    uniform_full35_policy,
)
from .io import (
    load_manifest,
    load_observations,
    load_region_rules,
    load_training_config,
    load_yaml_mapping,
)
from .model import AppliedPolicy

__all__ = [
    "ActivationManifest",
    "ActivationSite",
    "AppliedPolicy",
    "DatasetContract",
    "DeliveryAudit",
    "ExperimentSpec",
    "Full35ActivationExperiment",
    "Full35ExperimentConfig",
    "Full35LoadedPolicy",
    "Full35Phase",
    "Full35PreflightReport",
    "PolicyCost",
    "PolicyObservation",
    "RegionRule",
    "SearchConfig",
    "StaticPolicy",
    "TrainingArchitecture",
    "TrainingConfig",
    "TrainingPlan",
    "load_full35_manifest",
    "load_manifest",
    "load_observations",
    "load_region_rules",
    "load_training_config",
    "load_yaml_mapping",
    "uniform_full35_policy",
]
