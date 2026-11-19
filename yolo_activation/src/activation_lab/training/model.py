"""Manifest inspection and transactional activation replacement."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from torch import nn

from ..activations import build_activation
from .domain import ActivationManifest, ActivationSite, RegionRule, StaticPolicy


@dataclass(frozen=True)
class AppliedPolicy:
    model: nn.Module
    policy: StaticPolicy
    changed_paths: tuple[str, ...]
    unchanged_paths: tuple[str, ...]


def inspect_silu_sites(
    model: nn.Module,
    *,
    model_id: str,
    region_rules: tuple[RegionRule, ...],
) -> ActivationManifest:
    sites: list[ActivationSite] = []
    compiled = tuple((re.compile(rule.pattern), rule) for rule in region_rules)
    # Ultralytics registers one stateless Conv.default_act object under many
    # Conv paths. The default remove_duplicate=True would therefore expose
    # only the first path and silently leave the other references unchanged.
    # A manifest is path-based, so every registered reference must be listed.
    for module_path, module in model.named_modules(remove_duplicate=False):
        if not module_path or not isinstance(module, nn.SiLU):
            continue
        matches = [rule for pattern, rule in compiled if pattern.search(module_path)]
        if len(matches) > 1:
            regions = ", ".join(rule.region for rule in matches)
            raise ValueError(
                f"activation site {module_path!r} matches multiple regions: {regions}"
            )
        if matches:
            rule = matches[0]
            sites.append(
                ActivationSite(
                    module_path=module_path,
                    region=rule.region,
                    eligible=rule.eligible,
                )
            )
        else:
            sites.append(
                ActivationSite(
                    module_path=module_path,
                    region="unmapped",
                    eligible=False,
                )
            )
    return ActivationManifest(model_id=model_id, sites=tuple(sites), reviewed=False)


def _set_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = replacement
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = replacement
    else:
        setattr(parent, child_name, replacement)


def apply_static_policy(
    model: nn.Module,
    manifest: ActivationManifest,
    policy: StaticPolicy,
    *,
    clone_model: bool = True,
) -> AppliedPolicy:
    errors = manifest.audit(require_review=True)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))

    missing_paths: list[str] = []
    stale_paths: list[str] = []
    for site in manifest.sites:
        try:
            current = model.get_submodule(site.module_path)
        except AttributeError:
            missing_paths.append(site.module_path)
            continue
        if not isinstance(current, nn.SiLU):
            stale_paths.append(site.module_path)
    if missing_paths or stale_paths:
        details = []
        if missing_paths:
            details.append("missing paths: " + ", ".join(missing_paths))
        if stale_paths:
            details.append("non-SiLU source paths: " + ", ".join(stale_paths))
        raise ValueError(
            "policy preflight failed before mutation: " + "; ".join(details)
        )

    actual_model = copy.deepcopy(model) if clone_model else model
    changed: list[str] = []
    unchanged: list[str] = []
    for site in manifest.sites:
        activation_name = policy.resolve(site)
        if activation_name == "silu":
            unchanged.append(site.module_path)
            continue
        previous = actual_model.get_submodule(site.module_path)
        replacement = build_activation(activation_name)
        replacement.train(previous.training)
        _set_submodule(actual_model, site.module_path, replacement)
        changed.append(site.module_path)

    return AppliedPolicy(
        model=actual_model,
        policy=policy,
        changed_paths=tuple(changed),
        unchanged_paths=tuple(unchanged),
    )
