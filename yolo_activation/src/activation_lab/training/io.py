"""Strict YAML loaders and delivery-contract audit for the training module."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from ..activations import ActivationName
from .domain import (
    ActivationManifest,
    ActivationSite,
    DatasetContract,
    DeliveryAudit,
    PolicyObservation,
    RegionRule,
    SearchConfig,
    StaticPolicy,
    TrainingConfig,
)


def _is_placeholder(value: Any) -> bool:
    text = str(value)
    return (
        "REPLACE_WITH" in text
        or text.startswith("/absolute/path/")
        or bool(re.fullmatch(r"0+", text))
    )


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    actual_path = Path(path)
    payload = yaml.safe_load(actual_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{actual_path} must contain a YAML mapping")
    return payload


def _schema_version(payload: Mapping[str, Any], source: str) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"{source} requires schema_version: 1")


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = load_yaml_mapping(path)
    _schema_version(payload, "training config")
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list):
        raise TypeError("training config datasets must be a list")
    datasets = tuple(
        DatasetContract(
            dataset_id=str(item["dataset_id"]),
            source_kind=str(item["source_kind"]),
            yaml_path=str(item["yaml_path"]),
            task=str(item["task"]),
            num_classes=int(item["num_classes"]),
            immutable=bool(item.get("immutable", True)),
        )
        for item in raw_datasets
    )
    raw_activations = payload.get("activations", {})
    raw_search = payload.get("search", {})
    if not isinstance(raw_activations, dict) or not isinstance(raw_search, dict):
        raise TypeError("activations and search must be mappings")
    return TrainingConfig(
        architecture_id=str(payload["architecture_id"]),
        datasets=datasets,
        seed=int(payload.get("seed", 1)),
        optional_finalist_seed=(
            None
            if payload.get("optional_finalist_seed") is None
            else int(payload["optional_finalist_seed"])
        ),
        accuracy_reference=cast(
            ActivationName, raw_activations.get("accuracy_reference", "silu")
        ),
        hardware_neighbor_baseline=cast(
            ActivationName,
            raw_activations.get("hardware_neighbor_baseline", "hardswish"),
        ),
        cheap_control=cast(
            ActivationName, raw_activations.get("cheap_control", "relu")
        ),
        proposed_candidates=tuple(
            cast(list[ActivationName], raw_activations.get("proposed_candidates", []))
        ),
        region_search_candidates=tuple(
            cast(
                list[ActivationName],
                raw_activations.get("region_search_candidates", []),
            )
        ),
        search=SearchConfig(
            max_deployment_kernels=int(raw_search.get("max_deployment_kernels", 3)),
            beam_width=int(raw_search.get("beam_width", 4)),
            max_changed_regions=int(raw_search.get("max_changed_regions", 3)),
            max_finalists=int(raw_search.get("max_finalists", 2)),
            max_map_loss=(
                None
                if raw_search.get("max_map_loss") is None
                else float(raw_search["max_map_loss"])
            ),
            max_ap_s_loss=(
                None
                if raw_search.get("max_ap_s_loss") is None
                else float(raw_search["max_ap_s_loss"])
            ),
        ),
    )


def load_region_rules(path: str | Path) -> tuple[RegionRule, ...]:
    payload = load_yaml_mapping(path)
    _schema_version(payload, "region rules")
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise TypeError("region rules file must contain a rules list")
    return tuple(
        RegionRule(
            pattern=str(rule["pattern"]),
            region=str(rule["region"]),
            eligible=bool(rule.get("eligible", True)),
        )
        for rule in rules
    )


def _policy_from_mapping(payload: Mapping[str, Any]) -> StaticPolicy:
    raw_regions = payload.get("region_assignments", {})
    raw_sites = payload.get("site_assignments", {})
    if not isinstance(raw_regions, dict) or not isinstance(raw_sites, dict):
        raise TypeError("policy assignments must be mappings")
    return StaticPolicy(
        policy_id=str(payload["policy_id"]),
        default_activation=cast(
            ActivationName, payload.get("default_activation", "silu")
        ),
        region_assignments=tuple(
            sorted(
                (str(region), cast(ActivationName, activation))
                for region, activation in raw_regions.items()
            )
        ),
        site_assignments=tuple(
            sorted(
                (str(site), cast(ActivationName, activation))
                for site, activation in raw_sites.items()
            )
        ),
    )


def load_manifest(path: str | Path) -> ActivationManifest:
    payload = load_yaml_mapping(path)
    _schema_version(payload, "activation manifest")
    raw_sites = payload.get("sites")
    if not isinstance(raw_sites, list):
        raise TypeError("activation manifest sites must be a list")
    sites = tuple(
        ActivationSite(
            module_path=str(site["module_path"]),
            region=str(site["region"]),
            eligible=bool(site.get("eligible", True)),
            cost_weight=float(site.get("cost_weight", 1.0)),
            original_activation=str(site.get("original_activation", "silu")),
        )
        for site in raw_sites
    )
    return ActivationManifest(
        model_id=str(payload["model_id"]),
        sites=sites,
        reviewed=bool(payload.get("reviewed", False)),
        model_source_sha256=(
            None
            if payload.get("model_source_sha256") is None
            else str(payload["model_source_sha256"])
        ),
    )


def load_observations(path: str | Path) -> tuple[PolicyObservation, ...]:
    payload = load_yaml_mapping(path)
    _schema_version(payload, "observations")
    raw_observations = payload.get("observations")
    if not isinstance(raw_observations, list):
        raise TypeError("observations must be a list")
    observations = []
    for item in raw_observations:
        if not isinstance(item, dict) or not isinstance(item.get("policy"), dict):
            raise TypeError("each observation requires a nested policy mapping")
        observations.append(
            PolicyObservation(
                dataset_id=str(item["dataset_id"]),
                stage=str(item["stage"]),
                policy=_policy_from_mapping(item["policy"]),
                map_loss=(
                    None if item.get("map_loss") is None else float(item["map_loss"])
                ),
                ap_s_loss=(
                    None if item.get("ap_s_loss") is None else float(item["ap_s_loss"])
                ),
                latency_ms=(
                    None
                    if item.get("latency_ms") is None
                    else float(item["latency_ms"])
                ),
                failed=bool(item.get("failed", False)),
            )
        )
    return tuple(observations)


def audit_delivery_mapping(
    payload: Mapping[str, Any],
    config: TrainingConfig,
    *,
    check_files: bool = True,
) -> DeliveryAudit:
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("delivery contract requires schema_version: 1")
    delivery_id = payload.get("delivery_id")
    if not isinstance(delivery_id, str) or not delivery_id:
        errors.append("delivery_id is required")
        delivery_id = None
    elif _is_placeholder(delivery_id):
        errors.append("delivery_id still contains a placeholder")

    model = payload.get("model")
    if not isinstance(model, dict):
        errors.append("model mapping is required")
        model = {}
    required_model_fields = (
        "source_id",
        "config_path",
        "checkpoint_path",
        "checkpoint_sha256",
        "framework_commit",
        "python_version",
        "torch_version",
        "framework_version",
    )
    for field in required_model_fields:
        if not model.get(field):
            errors.append(f"model.{field} is required")
        elif _is_placeholder(model[field]):
            errors.append(f"model.{field} still contains a placeholder")
    checkpoint_hash = model.get("checkpoint_sha256")
    if checkpoint_hash and (
        _is_placeholder(checkpoint_hash)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", str(checkpoint_hash))
    ):
        errors.append("model.checkpoint_sha256 must be a 64-character SHA-256")

    manifest_path = payload.get("activation_manifest_path")
    if not manifest_path:
        errors.append("activation_manifest_path is required")
    elif _is_placeholder(manifest_path):
        errors.append("activation_manifest_path still contains a placeholder")

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list):
        errors.append("datasets must be a list")
        raw_datasets = []
    by_id = {
        str(item.get("dataset_id")): item
        for item in raw_datasets
        if isinstance(item, dict) and item.get("dataset_id")
    }
    if len(by_id) != len(raw_datasets):
        errors.append("delivery datasets must have unique dataset_id values")
    for expected in config.datasets:
        item = by_id.get(expected.dataset_id)
        if item is None:
            errors.append(f"missing dataset delivery: {expected.dataset_id}")
            continue
        if item.get("yaml_path") != expected.yaml_path:
            errors.append(
                f"{expected.dataset_id} must use canonical YAML {expected.yaml_path}"
            )
        for field in (
            "dataset_fingerprint",
            "baseline_metrics",
            "recipe_path",
            "baseline_checkpoint_path",
            "baseline_checkpoint_sha256",
        ):
            if not item.get(field):
                errors.append(f"{expected.dataset_id}.{field} is required")
            elif _is_placeholder(item[field]):
                errors.append(
                    f"{expected.dataset_id}.{field} still contains a placeholder"
                )
        metrics = item.get("baseline_metrics")
        if not isinstance(metrics, dict) or not isinstance(
            metrics.get("map50_95"), (int, float)
        ):
            errors.append(
                f"{expected.dataset_id}.baseline_metrics.map50_95 must be numeric"
            )
        dataset_hash = item.get("baseline_checkpoint_sha256")
        if dataset_hash and (
            _is_placeholder(dataset_hash)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(dataset_hash))
        ):
            errors.append(
                f"{expected.dataset_id}.baseline_checkpoint_sha256 must be SHA-256"
            )
    extras = sorted(set(by_id) - {dataset.dataset_id for dataset in config.datasets})
    if extras:
        errors.append(f"delivery contains undeclared datasets: {extras}")

    if check_files:
        paths = [model.get("config_path"), model.get("checkpoint_path"), manifest_path]
        for item in by_id.values():
            paths.extend(
                [
                    item.get("yaml_path"),
                    item.get("recipe_path"),
                    item.get("baseline_checkpoint_path"),
                ]
            )
        for path in (Path(str(value)) for value in paths if value):
            if not path.exists():
                errors.append(f"required delivery path does not exist: {path}")
    hardware_target = payload.get("hardware_target")
    if hardware_target is None:
        warnings.append(
            "hardware_target 尚未提供；只能產生結構 proxy，不能宣稱 latency/area/power"
        )
    elif not isinstance(hardware_target, dict):
        errors.append("hardware_target must be a mapping when provided")

    return DeliveryAudit(
        delivery_id=cast(str | None, delivery_id),
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized=dict(payload),
    )
