"""BCSP: staged, budget-constrained Pareto search over static profiles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..hardware import hardware_proxy
from .domain import (
    ActivationManifest,
    ExperimentSpec,
    PolicyCost,
    PolicyObservation,
    SearchConfig,
    StaticPolicy,
    TrainingConfig,
    TrainingPlan,
)


@dataclass(frozen=True)
class _FrontierEntry:
    observation: PolicyObservation
    cost: PolicyCost


def baseline_policy() -> StaticPolicy:
    return StaticPolicy(policy_id="baseline--silu", default_activation="silu")


def uniform_policy(activation: str) -> StaticPolicy:
    return StaticPolicy(
        policy_id=f"uniform--{activation}",
        default_activation=activation,  # type: ignore[arg-type]
    )


def region_policy(region: str, activation: str) -> StaticPolicy:
    return StaticPolicy(
        policy_id=f"static--{region}={activation}",
        region_assignments=((region, activation),),  # type: ignore[arg-type]
    )


def _policy_id(assignments: tuple[tuple[str, str], ...]) -> str:
    body = "__".join(f"{region}={activation}" for region, activation in assignments)
    return f"static--{body}"


def policy_cost(manifest: ActivationManifest, policy: StaticPolicy) -> PolicyCost:
    variable = 0.0
    constant = 0.0
    range_operations = 0.0
    transcendental = 0.0
    used = set()
    for site in manifest.sites:
        activation = policy.resolve(site)
        used.add(activation)
        proxy = hardware_proxy(activation)
        variable += site.cost_weight * proxy.variable_multiplications
        constant += site.cost_weight * proxy.constant_multiplications
        range_operations += site.cost_weight * proxy.range_operations
        transcendental += site.cost_weight * proxy.transcendental_operations
    coefficient_count = sum(hardware_proxy(name).coefficient_count for name in used)
    return PolicyCost(
        variable_multiplications=variable,
        constant_multiplications=constant,
        range_operations=range_operations,
        transcendental_operations=transcendental,
        coefficient_count=coefficient_count,
        kernel_count=len(used),
    )


def _validate_policy(
    manifest: ActivationManifest, policy: StaticPolicy, search: SearchConfig
) -> None:
    regions = set(manifest.regions)
    paths = {site.module_path for site in manifest.sites}
    unknown_regions = sorted(set(dict(policy.region_assignments)) - regions)
    unknown_paths = sorted(set(dict(policy.site_assignments)) - paths)
    ineligible_paths = sorted(
        path
        for path in dict(policy.site_assignments)
        if path in paths
        and not next(
            site.eligible for site in manifest.sites if site.module_path == path
        )
    )
    if unknown_regions:
        raise ValueError(
            f"policy {policy.policy_id} uses unknown regions: {unknown_regions}"
        )
    if unknown_paths:
        raise ValueError(
            f"policy {policy.policy_id} uses unknown module paths: {unknown_paths}"
        )
    if ineligible_paths:
        raise ValueError(
            f"policy {policy.policy_id} assigns ineligible module paths: {ineligible_paths}"
        )
    kernels = policy_cost(manifest, policy).kernel_count
    if kernels > search.max_deployment_kernels:
        raise ValueError(
            f"policy {policy.policy_id} needs {kernels} activation kernels; "
            f"budget is {search.max_deployment_kernels}"
        )


def _is_feasible(observation: PolicyObservation, search: SearchConfig) -> bool:
    if observation.failed or observation.map_loss is None:
        return False
    if search.max_map_loss is not None and observation.map_loss > search.max_map_loss:
        return False
    if search.max_ap_s_loss is not None:
        if observation.ap_s_loss is None:
            return False
        if observation.ap_s_loss > search.max_ap_s_loss:
            return False
    return True


def _pareto_frontier(
    observations: tuple[PolicyObservation, ...],
    manifest: ActivationManifest,
    search: SearchConfig,
) -> tuple[_FrontierEntry, ...]:
    candidates = [obs for obs in observations if _is_feasible(obs, search)]
    if not candidates:
        return ()
    has_ap_s = [obs.ap_s_loss is not None for obs in candidates]
    has_latency = [obs.latency_ms is not None for obs in candidates]
    if any(has_ap_s) and not all(has_ap_s):
        raise ValueError("Pareto comparison cannot mix present and missing AP_S loss")
    if any(has_latency) and not all(has_latency):
        raise ValueError("Pareto comparison cannot mix measured and missing latency")

    def objectives(entry: _FrontierEntry) -> tuple[float, ...]:
        obs = entry.observation
        values: list[float] = [float(obs.map_loss)]
        if all(has_ap_s):
            values.append(float(obs.ap_s_loss))
        if all(has_latency):
            values.append(float(obs.latency_ms))
        values.extend(entry.cost.objective_tuple())
        return tuple(values)

    entries = tuple(
        _FrontierEntry(observation=obs, cost=policy_cost(manifest, obs.policy))
        for obs in candidates
    )
    frontier: list[_FrontierEntry] = []
    for candidate in entries:
        candidate_values = objectives(candidate)
        dominated = False
        for other in entries:
            if other is candidate:
                continue
            other_values = objectives(other)
            if all(
                left <= right for left, right in zip(other_values, candidate_values)
            ) and any(
                left < right for left, right in zip(other_values, candidate_values)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: item.observation.policy.policy_id))


def _expand_frontier(
    frontier: tuple[_FrontierEntry, ...],
    manifest: ActivationManifest,
    config: TrainingConfig,
) -> tuple[StaticPolicy, ...]:
    ranked = sorted(
        frontier,
        key=lambda entry: (
            float(entry.observation.map_loss),
            entry.cost.objective_tuple(),
            entry.observation.policy.policy_id,
        ),
    )[: config.search.beam_width]
    proposed: dict[str, StaticPolicy] = {}
    for entry in ranked:
        policy = entry.observation.policy
        if policy.default_activation != "silu" or policy.site_assignments:
            continue
        assignments = dict(policy.region_assignments)
        changed = sum(activation != "silu" for activation in assignments.values())
        if changed >= config.search.max_changed_regions:
            continue
        for region in manifest.regions:
            if region in assignments:
                continue
            for activation in config.region_search_candidates:
                expanded = dict(assignments)
                expanded[region] = activation
                normalized = tuple(sorted(expanded.items()))
                candidate = StaticPolicy(
                    policy_id=_policy_id(normalized),
                    region_assignments=normalized,  # type: ignore[arg-type]
                )
                if (
                    policy_cost(manifest, candidate).kernel_count
                    <= config.search.max_deployment_kernels
                ):
                    proposed[candidate.policy_id] = candidate
    ordered = sorted(
        proposed.values(),
        key=lambda policy: (
            policy_cost(manifest, policy).objective_tuple(),
            policy.policy_id,
        ),
    )
    return tuple(ordered[: config.search.beam_width])


def _select_finalists(
    frontier: tuple[_FrontierEntry, ...], search: SearchConfig
) -> tuple[StaticPolicy, ...]:
    if not frontier:
        return ()
    accuracy_best = min(
        frontier,
        key=lambda entry: (
            float(entry.observation.map_loss),
            float(entry.observation.ap_s_loss or 0.0),
            entry.observation.policy.policy_id,
        ),
    )
    selected = [accuracy_best.observation.policy]
    if search.max_finalists > 1:
        hardware_best = min(
            frontier,
            key=lambda entry: (
                float(entry.observation.latency_ms)
                if entry.observation.latency_ms is not None
                else entry.cost.objective_tuple(),
                float(entry.observation.map_loss),
                entry.observation.policy.policy_id,
            ),
        )
        if hardware_best.observation.policy.policy_id != selected[0].policy_id:
            selected.append(hardware_best.observation.policy)
    return tuple(selected)


def _experiment(
    *,
    dataset_id: str,
    dataset_yaml: str,
    stage: str,
    mode: str,
    seed: int,
    policy: StaticPolicy,
    manifest: ActivationManifest,
    depends_on: tuple[str, ...] = (),
) -> ExperimentSpec:
    experiment_id = f"{dataset_id}--{stage}--{policy.policy_id}--seed{seed}"
    return ExperimentSpec(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        dataset_yaml=dataset_yaml,
        stage=stage,
        mode=mode,
        seed=seed,
        policy=policy,
        policy_cost=policy_cost(manifest, policy),
        depends_on=depends_on,
    )


def compile_next_plan(
    manifest: ActivationManifest,
    config: TrainingConfig,
    observations: tuple[PolicyObservation, ...] = (),
) -> TrainingPlan:
    manifest_errors = manifest.audit(require_review=True)
    if manifest_errors:
        return TrainingPlan(
            architecture_id=config.architecture_id,
            manifest_model_id=manifest.model_id,
            next_stage="blocked",
            experiments=(),
            frontier_policy_ids=(),
            blocked_reasons=manifest_errors,
            notes=("先完成 manifest review，任何替換或訓練計畫才可產生。",),
        )

    dataset_ids = {dataset.dataset_id for dataset in config.datasets}
    unknown_datasets = sorted({obs.dataset_id for obs in observations} - dataset_ids)
    if unknown_datasets:
        raise ValueError(f"observations contain unknown datasets: {unknown_datasets}")
    seen_keys: set[tuple[str, str, str]] = set()
    for observation in observations:
        _validate_policy(manifest, observation.policy, config.search)
        key = (observation.dataset_id, observation.stage, observation.policy.policy_id)
        if key in seen_keys:
            raise ValueError(f"duplicate observation: {key}")
        seen_keys.add(key)

    by_dataset: dict[str, list[PolicyObservation]] = defaultdict(list)
    for observation in observations:
        by_dataset[observation.dataset_id].append(observation)

    baseline = baseline_policy()
    uniform_names = tuple(
        dict.fromkeys(
            (
                config.hardware_neighbor_baseline,
                config.cheap_control,
                *config.proposed_candidates,
            )
        )
    )
    uniform = {name: uniform_policy(name) for name in uniform_names}
    recovery_names = (config.hardware_neighbor_baseline, *config.proposed_candidates)

    experiments: list[ExperimentSpec] = []
    blockers: list[str] = []
    final_frontiers: list[str] = []
    stages: set[str] = set()

    for dataset in config.datasets:
        dataset_observations = tuple(by_dataset[dataset.dataset_id])

        def get(
            stage: str,
            policy_id: str,
            current_observations: tuple[PolicyObservation, ...] = dataset_observations,
        ) -> PolicyObservation | None:
            return next(
                (
                    obs
                    for obs in current_observations
                    if obs.stage == stage and obs.policy.policy_id == policy_id
                ),
                None,
            )

        baseline_result = get("baseline_reproduction", baseline.policy_id)
        if baseline_result is None:
            stages.add("baseline_reproduction")
            experiments.append(
                _experiment(
                    dataset_id=dataset.dataset_id,
                    dataset_yaml=dataset.yaml_path,
                    stage="baseline_reproduction",
                    mode="baseline_reproduction",
                    seed=config.seed,
                    policy=baseline,
                    manifest=manifest,
                )
            )
            continue
        if baseline_result.failed:
            blockers.append(f"{dataset.dataset_id}: baseline reproduction failed")
            continue

        missing_zero = [
            policy
            for policy in uniform.values()
            if get("zero_shot", policy.policy_id) is None
        ]
        if missing_zero:
            stages.add("zero_shot")
            for policy in missing_zero:
                experiments.append(
                    _experiment(
                        dataset_id=dataset.dataset_id,
                        dataset_yaml=dataset.yaml_path,
                        stage="zero_shot",
                        mode="zero_shot",
                        seed=config.seed,
                        policy=policy,
                        manifest=manifest,
                    )
                )
            continue

        missing_recovery: list[StaticPolicy] = []
        for name in recovery_names:
            policy = uniform[name]
            zero_result = get("zero_shot", policy.policy_id)
            if (
                zero_result is not None
                and not zero_result.failed
                and get("short_recovery", policy.policy_id) is None
            ):
                missing_recovery.append(policy)
        if missing_recovery:
            stages.add("short_recovery")
            for policy in missing_recovery:
                experiments.append(
                    _experiment(
                        dataset_id=dataset.dataset_id,
                        dataset_yaml=dataset.yaml_path,
                        stage="short_recovery",
                        mode="short_recovery",
                        seed=config.seed,
                        policy=policy,
                        manifest=manifest,
                    )
                )
            continue

        region_activations = [
            activation
            for activation in config.region_search_candidates
            if (result := get("short_recovery", uniform[activation].policy_id))
            is not None
            and not result.failed
        ]
        if not region_activations:
            blockers.append(
                f"{dataset.dataset_id}: no proposed candidate survived recovery"
            )
            continue
        region_policies = tuple(
            region_policy(region, activation)
            for activation in region_activations
            for region in manifest.regions
        )
        missing_regions = [
            policy
            for policy in region_policies
            if get("region_sensitivity", policy.policy_id) is None
        ]
        if missing_regions:
            stages.add("region_sensitivity")
            for policy in missing_regions:
                experiments.append(
                    _experiment(
                        dataset_id=dataset.dataset_id,
                        dataset_yaml=dataset.yaml_path,
                        stage="region_sensitivity",
                        mode="short_recovery",
                        seed=config.seed,
                        policy=policy,
                        manifest=manifest,
                    )
                )
            continue

        search_observations = tuple(
            obs
            for obs in dataset_observations
            if obs.stage in {"region_sensitivity", "policy_search"}
        )
        frontier = _pareto_frontier(search_observations, manifest, config.search)
        if not frontier:
            blockers.append(
                f"{dataset.dataset_id}: no feasible region policy under the frozen budgets"
            )
            continue
        expansions = _expand_frontier(frontier, manifest, config)
        missing_expansions = [
            policy
            for policy in expansions
            if get("policy_search", policy.policy_id) is None
        ]
        if missing_expansions:
            stages.add("policy_search")
            for policy in missing_expansions:
                experiments.append(
                    _experiment(
                        dataset_id=dataset.dataset_id,
                        dataset_yaml=dataset.yaml_path,
                        stage="policy_search",
                        mode="short_recovery",
                        seed=config.seed,
                        policy=policy,
                        manifest=manifest,
                    )
                )
            continue

        comparison_observations = list(search_observations)
        hardswish_result = get(
            "short_recovery", uniform[config.hardware_neighbor_baseline].policy_id
        )
        if hardswish_result is not None and not hardswish_result.failed:
            comparison_observations.append(hardswish_result)
        final_frontier = _pareto_frontier(
            tuple(comparison_observations), manifest, config.search
        )
        finalists = _select_finalists(final_frontier, config.search)
        missing_full_seed1 = [
            policy
            for policy in finalists
            if get("full_recovery_seed1", policy.policy_id) is None
        ]
        if missing_full_seed1:
            stages.add("full_recovery_seed1")
            for policy in missing_full_seed1:
                experiments.append(
                    _experiment(
                        dataset_id=dataset.dataset_id,
                        dataset_yaml=dataset.yaml_path,
                        stage="full_recovery_seed1",
                        mode="full_recovery",
                        seed=config.seed,
                        policy=policy,
                        manifest=manifest,
                    )
                )
            continue

        if config.optional_finalist_seed is not None:
            missing_seed2 = [
                policy
                for policy in finalists
                if get("full_recovery_seed2", policy.policy_id) is None
            ]
            if missing_seed2:
                stages.add("full_recovery_seed2")
                for policy in missing_seed2:
                    experiments.append(
                        _experiment(
                            dataset_id=dataset.dataset_id,
                            dataset_yaml=dataset.yaml_path,
                            stage="full_recovery_seed2",
                            mode="full_recovery",
                            seed=config.optional_finalist_seed,
                            policy=policy,
                            manifest=manifest,
                        )
                    )
                continue
        final_frontiers.extend(
            f"{dataset.dataset_id}:{policy.policy_id}" for policy in finalists
        )

    if experiments:
        next_stage = (
            next(iter(stages)) if len(stages) == 1 else "dataset_stages_diverged"
        )
    elif blockers:
        next_stage = "blocked"
    else:
        next_stage = "complete"
    return TrainingPlan(
        architecture_id=config.architecture_id,
        manifest_model_id=manifest.model_id,
        next_stage=next_stage,
        experiments=tuple(experiments),
        frontier_policy_ids=tuple(sorted(final_frontiers)),
        blocked_reasons=tuple(blockers),
        notes=(
            "SIPA 使用 even-residual symmetry；BCSP 只產生下一個必要 stage。",
            "COCO2017 與 Canonical BBAT5 v1 的 observations、frontier 與結果永不平均。",
            "硬體 cost 是結構 proxy；有 target latency 時才加入實測 latency objective。",
        ),
    )
