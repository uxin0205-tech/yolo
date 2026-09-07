"""Fail-closed cross-parent routing for exact W4 versus Fixed-SD4 profiles."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class ExactRoutingSource:
    """One hash-bound active-parent exact profile artifact."""

    parent_name: str
    checkpoint_sha256: str
    artifact_path: str
    artifact_sha256: str
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if not self.parent_name or not self.artifact_path:
            raise ValueError("exact routing source names and paths must not be empty")
        for label, digest in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("artifact_sha256", self.artifact_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ExactRoutingManifest:
    """Static routing evidence; never an authorization or mAP result."""

    schema_version: int
    manifest_id: str
    status: str
    comparison: dict[str, object]
    source_artifacts: tuple[dict[str, str], ...]
    counts: dict[str, object]
    stable_candidates: tuple[str, ...]
    parent_view_winners: dict[str, dict[str, tuple[str, ...]]]
    static_proxy_effect: dict[str, dict[str, float]]
    exact_solver_audit: dict[str, object]
    limitations: tuple[str, ...]
    execution_authorized: bool = False
    gpu_used: bool = False
    formal_training: bool = False
    map_validation_run: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "status": self.status,
            "execution_authorized": self.execution_authorized,
            "gpu_used": self.gpu_used,
            "formal_training": self.formal_training,
            "map_validation_run": self.map_validation_run,
            "comparison": self.comparison,
            "source_artifacts": list(self.source_artifacts),
            "counts": self.counts,
            "stable_candidates": list(self.stable_candidates),
            "parent_view_winners": {
                parent: {view: list(paths) for view, paths in views.items()}
                for parent, views in self.parent_view_winners.items()
            },
            "static_proxy_effect": self.static_proxy_effect,
            "exact_solver_audit": self.exact_solver_audit,
            "limitations": list(self.limitations),
        }


class ExactWeightRouting:
    """Build one conservative routing decision from complete exact profiles."""

    REQUIRED_PARENTS: ClassVar[tuple[str, ...]] = (
        "qsilu_pq",
        "hardswish",
        "poly_shift",
    )
    REQUIRED_VIEWS: ClassVar[tuple[str, ...]] = ("master", "deployment")
    UNIFORM_GRID: ClassVar[str] = "uniform-w4-per_output_channel-mse_grid_v1"
    UNIFORM_EXACT: ClassVar[str] = (
        "uniform-w4-per_output_channel-optimal_scaled_codebook"
    )
    SD4_GRID: ClassVar[str] = "fixed-sd4-per_output_channel-mse_grid_v1"
    SD4_EXACT: ClassVar[str] = "fixed-sd4-per_output_channel-optimal_scaled_codebook"
    REQUIRED_FORMATS: ClassVar[tuple[str, ...]] = (
        UNIFORM_GRID,
        UNIFORM_EXACT,
        SD4_GRID,
        SD4_EXACT,
    )

    @staticmethod
    def _mapping(value: object, *, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"{label} must be a mapping")
        return value

    @staticmethod
    def _float(value: object, *, label: str) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
        return float(value)

    def _index_source(
        self,
        source: ExactRoutingSource,
    ) -> dict[tuple[str, str, str], dict[str, Any]]:
        payload = source.payload
        if payload.get("kind") != "full35_exact_w4_sd4_static_analysis":
            raise ValueError("unexpected exact routing source kind")
        if payload.get("profile") != "exact_w4_sd4_v1":
            raise ValueError("unexpected exact routing source profile")
        if payload.get("gpu_used") is not False:
            raise ValueError("exact routing source must be CPU-only")
        if payload.get("formal_training") is not False:
            raise ValueError("exact routing source cannot be formal training")
        parent = self._mapping(payload.get("parent"), label="parent")
        if parent.get("name") != source.parent_name:
            raise ValueError("exact routing source parent mismatch")
        if parent.get("checkpoint_sha256") != source.checkpoint_sha256:
            raise ValueError("exact routing source checkpoint mismatch")
        plan = self._mapping(payload.get("analysis_plan"), label="analysis_plan")
        if tuple(plan.get("views", ())) != self.REQUIRED_VIEWS:
            raise ValueError("exact routing source views do not match the contract")
        if tuple(plan.get("formats", ())) != self.REQUIRED_FORMATS:
            raise ValueError("exact routing source formats do not match the contract")
        measurements = payload.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise ValueError("exact routing measurements must be a non-empty list")
        indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
        for raw in measurements:
            item = self._mapping(raw, label="measurement")
            key = (
                str(item.get("view")),
                str(item.get("path")),
                str(item.get("format_id")),
            )
            if key in indexed:
                raise ValueError(f"duplicate exact routing measurement: {key}")
            indexed[key] = item

        paths_by_view = {
            view: {
                path
                for indexed_view, path, format_id in indexed
                if indexed_view == view and format_id in self.REQUIRED_FORMATS
            }
            for view in self.REQUIRED_VIEWS
        }
        if not paths_by_view["master"] or (
            paths_by_view["master"] != paths_by_view["deployment"]
        ):
            raise ValueError("incomplete exact routing matrix: view path sets differ")
        expected = {
            (view, path, format_id)
            for view in self.REQUIRED_VIEWS
            for path in paths_by_view[view]
            for format_id in self.REQUIRED_FORMATS
        }
        if set(indexed) != expected:
            raise ValueError("incomplete exact routing matrix")

        for current_view, path in (
            (required_view, path)
            for required_view in self.REQUIRED_VIEWS
            for path in paths_by_view[required_view]
        ):
            cells = [
                indexed[(current_view, path, format_id)]
                for format_id in self.REQUIRED_FORMATS
            ]
            parity = {
                (
                    int(cell.get("bits", -1)),
                    str(cell.get("granularity")),
                    int(cell.get("elements", -1)),
                    int(cell.get("scale_count", -1)),
                    int(cell.get("code_bytes", -1)),
                    int(cell.get("metadata_bytes", -1)),
                    str(cell.get("region")),
                )
                for cell in cells
            }
            if len(parity) != 1:
                raise ValueError(
                    f"exact routing fairness parity failed for {current_view}:{path}"
                )
        return indexed

    def build(
        self,
        sources: tuple[ExactRoutingSource, ...],
    ) -> ExactRoutingManifest:
        """Validate all cells and intersect only strict SD4 wins."""

        source_by_parent = {source.parent_name: source for source in sources}
        if len(source_by_parent) != len(sources):
            raise ValueError("duplicate exact routing parent")
        if tuple(source_by_parent) != self.REQUIRED_PARENTS:
            raise ValueError(
                "exact routing requires parents in order: "
                + ", ".join(self.REQUIRED_PARENTS)
            )
        indexed = {
            parent: self._index_source(source_by_parent[parent])
            for parent in self.REQUIRED_PARENTS
        }
        path_sets = {
            parent: {
                path
                for view, path, format_id in parent_index
                if view == "deployment" and format_id == self.UNIFORM_EXACT
            }
            for parent, parent_index in indexed.items()
        }
        first_paths = path_sets[self.REQUIRED_PARENTS[0]]
        if not first_paths or any(paths != first_paths for paths in path_sets.values()):
            raise ValueError("incomplete exact routing matrix: parent path sets differ")

        parent_view_winners: dict[str, dict[str, tuple[str, ...]]] = {}
        exact_audit_counts = Counter()
        exact_audit_gain = {
            "uniform": [],
            "fixed_sd4": [],
        }
        for parent in self.REQUIRED_PARENTS:
            parent_view_winners[parent] = {}
            for view in self.REQUIRED_VIEWS:
                winners: list[str] = []
                for path in sorted(first_paths):
                    control = indexed[parent][(view, path, self.UNIFORM_EXACT)]
                    candidate = indexed[parent][(view, path, self.SD4_EXACT)]
                    control_mse = self._float(
                        self._mapping(control.get("numeric"), label="numeric").get(
                            "mse"
                        ),
                        label="uniform exact mse",
                    )
                    candidate_mse = self._float(
                        self._mapping(candidate.get("numeric"), label="numeric").get(
                            "mse"
                        ),
                        label="SD4 exact mse",
                    )
                    if candidate_mse < control_mse:
                        winners.append(path)
                    for family, exact_id, grid_id in (
                        ("uniform", self.UNIFORM_EXACT, self.UNIFORM_GRID),
                        ("fixed_sd4", self.SD4_EXACT, self.SD4_GRID),
                    ):
                        exact_mse = self._float(
                            self._mapping(
                                indexed[parent][(view, path, exact_id)].get("numeric"),
                                label="numeric",
                            ).get("mse"),
                            label=f"{family} exact mse",
                        )
                        grid_mse = self._float(
                            self._mapping(
                                indexed[parent][(view, path, grid_id)].get("numeric"),
                                label="numeric",
                            ).get("mse"),
                            label=f"{family} grid mse",
                        )
                        tolerance = max(1e-12, abs(grid_mse) * 1e-7)
                        if exact_mse > grid_mse + tolerance:
                            raise ValueError(
                                "exact solver is worse than grid for "
                                f"{parent}:{view}:{path}:{family}"
                            )
                        exact_audit_counts[family] += 1
                        exact_audit_gain[family].append(grid_mse - exact_mse)
                parent_view_winners[parent][view] = tuple(winners)

        stable = set(first_paths)
        for parent in self.REQUIRED_PARENTS:
            for view in self.REQUIRED_VIEWS:
                stable.intersection_update(parent_view_winners[parent][view])
        stable_candidates = tuple(sorted(stable))
        first_index = indexed[self.REQUIRED_PARENTS[0]]
        regions = Counter(
            str(first_index[("deployment", path, self.UNIFORM_EXACT)]["region"])
            for path in stable_candidates
        )

        static_proxy_effect: dict[str, dict[str, float]] = {}
        for parent in self.REQUIRED_PARENTS:
            reference_energy = 0.0
            control_error = 0.0
            hybrid_error = 0.0
            for path in sorted(first_paths):
                control = indexed[parent][("deployment", path, self.UNIFORM_EXACT)]
                candidate = indexed[parent][("deployment", path, self.SD4_EXACT)]
                elements = int(control["elements"])
                control_mse = self._float(
                    self._mapping(control["numeric"], label="numeric")["mse"],
                    label="control mse",
                )
                candidate_mse = self._float(
                    self._mapping(candidate["numeric"], label="numeric")["mse"],
                    label="candidate mse",
                )
                distribution = self._mapping(
                    control["distribution"],
                    label="distribution",
                )
                mean = self._float(distribution["mean"], label="weight mean")
                standard_deviation = self._float(
                    distribution["standard_deviation"],
                    label="weight standard deviation",
                )
                reference_energy += elements * (
                    standard_deviation * standard_deviation + mean * mean
                )
                control_error += elements * control_mse
                hybrid_error += elements * (
                    candidate_mse if path in stable else control_mse
                )
            all_w4_nrmse = math.sqrt(control_error / max(reference_energy, 1e-24))
            hybrid_nrmse = math.sqrt(hybrid_error / max(reference_energy, 1e-24))
            static_proxy_effect[parent] = {
                "all_w4_nrmse": all_w4_nrmse,
                "hybrid_nrmse": hybrid_nrmse,
                "relative_nrmse_reduction": (
                    (all_w4_nrmse - hybrid_nrmse) / max(all_w4_nrmse, 1e-24)
                ),
            }

        return ExactRoutingManifest(
            schema_version=3,
            manifest_id="full35-fixed-sd4-routing-candidates-v3",
            status="exact_static_candidates_only_not_validated_not_authorized",
            comparison={
                "candidate": self.SD4_EXACT,
                "control": self.UNIFORM_EXACT,
                "equal_code_bits": 4,
                "equal_scale_granularity": "per_output_channel",
                "selection_metric": "per_layer_weight_reconstruction_mse",
                "required_parents": list(self.REQUIRED_PARENTS),
                "required_views": list(self.REQUIRED_VIEWS),
                "rule": (
                    "candidate_mse_strictly_less_than_control_in_every_active_"
                    "parent_and_view"
                ),
            },
            source_artifacts=tuple(
                {
                    "parent": source.parent_name,
                    "path": source.artifact_path,
                    "sha256": source.artifact_sha256,
                    "checkpoint_sha256": source.checkpoint_sha256,
                }
                for source in sources
            ),
            counts={
                "paths_per_view_and_parent": len(first_paths),
                "parent_view_winners": {
                    parent: {
                        view: len(paths)
                        for view, paths in parent_view_winners[parent].items()
                    }
                    for parent in self.REQUIRED_PARENTS
                },
                "both_view_cross_parent_winners": len(stable_candidates),
                "regions": dict(sorted(regions.items())),
            },
            stable_candidates=stable_candidates,
            parent_view_winners=parent_view_winners,
            static_proxy_effect=static_proxy_effect,
            exact_solver_audit={
                "uniform_exact_never_worse_than_grid": True,
                "fixed_sd4_exact_never_worse_than_grid": True,
                "uniform_cells": exact_audit_counts["uniform"],
                "fixed_sd4_cells": exact_audit_counts["fixed_sd4"],
                "uniform_total_mse_reduction": sum(exact_audit_gain["uniform"]),
                "fixed_sd4_total_mse_reduction": sum(exact_audit_gain["fixed_sd4"]),
            },
            limitations=(
                "static_weight_reconstruction_only",
                "no_layer_output_nrmse",
                "no_topk_overlap",
                "no_eight_metric_map",
                "no_qat",
                "no_gpu_run",
                "no_bittrue_export_or_hardware_measurement",
                "candidate_manifest_does_not_promote_a_policy",
            ),
        )
