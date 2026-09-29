"""Conservative cross-parent routing for the paper three-value weight format."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class TernaryRoutingSource:
    """One hash-bound active-parent weight-distribution profile."""

    parent_name: str
    checkpoint_sha256: str
    artifact_path: str
    artifact_sha256: str
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if not self.parent_name or not self.artifact_path:
            raise ValueError("ternary routing source identity must not be empty")
        for label, digest in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("artifact_sha256", self.artifact_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TernaryRoutingManifest:
    """Static Paper-TWN candidates; never implicit execution authorization."""

    schema_version: int
    manifest_id: str
    status: str
    thresholds: dict[str, dict[str, float]]
    source_artifacts: tuple[dict[str, str], ...]
    counts: dict[str, object]
    safe_candidates: tuple[str, ...]
    balanced_candidates: tuple[str, ...]
    per_path_worst_case: dict[str, dict[str, float | int | str]]
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
            "format": {
                "family": "paper_twn",
                "format_id": "paper-twn-layerwise",
                "encoded_bits": 2,
                "granularity": "per_tensor",
                "threshold_rule": "delta=0.7*mean(abs(weight))",
                "levels": ["-alpha", "0", "+alpha"],
            },
            "selection": {
                "required_parents": list(PaperTWNRouting.REQUIRED_PARENTS),
                "required_views": list(PaperTWNRouting.REQUIRED_VIEWS),
                "rule": (
                    "thresholds_must_pass_in_every_active_parent_and_both_"
                    "master_and_bn_folded_deployment_views"
                ),
                "thresholds": self.thresholds,
            },
            "source_artifacts": list(self.source_artifacts),
            "counts": self.counts,
            "safe_candidates": list(self.safe_candidates),
            "balanced_candidates": list(self.balanced_candidates),
            "per_path_worst_case": self.per_path_worst_case,
            "limitations": list(self.limitations),
        }


class PaperTWNRouting:
    """Select only layers with consistently ternary-friendly distributions."""

    REQUIRED_PARENTS: ClassVar[tuple[str, ...]] = (
        "qsilu_pq",
        "hardswish",
        "poly_shift",
    )
    REQUIRED_VIEWS: ClassVar[tuple[str, ...]] = ("master", "deployment")
    FORMAT_ID: ClassVar[str] = "paper-twn-layerwise"
    THRESHOLDS: ClassVar[dict[str, dict[str, float]]] = {
        "safe": {"maximum_normalized_rmse": 0.35, "minimum_cosine": 0.93},
        "balanced": {"maximum_normalized_rmse": 0.4, "minimum_cosine": 0.91},
    }

    @staticmethod
    def _mapping(value: object, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"{label} must be a mapping")
        return value

    @staticmethod
    def _finite(value: object, label: str) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
        return float(value)

    def _index(
        self, source: TernaryRoutingSource
    ) -> dict[tuple[str, str], dict[str, Any]]:
        payload = source.payload
        if payload.get("kind") != "full35_static_weight_format_analysis":
            raise ValueError("unexpected Paper-TWN routing source kind")
        if payload.get("profile") != "main":
            raise ValueError("unexpected Paper-TWN routing source profile")
        if payload.get("gpu_used") is not False:
            raise ValueError("Paper-TWN routing source must be CPU-only")
        if payload.get("formal_training") is not False:
            raise ValueError("Paper-TWN routing source cannot be formal training")
        parent = self._mapping(payload.get("parent"), "Paper-TWN parent")
        if parent.get("activation") != source.parent_name:
            raise ValueError("Paper-TWN source parent mismatch")
        if parent.get("checkpoint_sha256") != source.checkpoint_sha256:
            raise ValueError("Paper-TWN source checkpoint mismatch")
        measurements = payload.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise ValueError("Paper-TWN measurements must be a non-empty list")
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in measurements:
            item = self._mapping(raw, "Paper-TWN measurement")
            if item.get("family") != "paper_twn":
                continue
            if (
                item.get("format_id") != self.FORMAT_ID
                or int(item.get("bits", -1)) != 2
                or item.get("granularity") != "per_tensor"
                or item.get("scale_method") != "paper_delta_0.7_mean_abs"
            ):
                raise ValueError("Paper-TWN format contract differs")
            key = (str(item.get("view")), str(item.get("path")))
            if key in indexed:
                raise ValueError(f"duplicate Paper-TWN measurement: {key}")
            numeric = self._mapping(item.get("numeric"), "Paper-TWN numeric")
            nrmse = self._finite(
                numeric.get("normalized_rmse"), "Paper-TWN normalized_rmse"
            )
            cosine = self._finite(numeric.get("cosine"), "Paper-TWN cosine")
            if nrmse < 0.0 or not -1.0 <= cosine <= 1.0:
                raise ValueError("Paper-TWN numeric values are outside valid ranges")
            indexed[key] = item

        paths_by_view = {
            view: {path for indexed_view, path in indexed if indexed_view == view}
            for view in self.REQUIRED_VIEWS
        }
        if not paths_by_view["master"] or (
            paths_by_view["master"] != paths_by_view["deployment"]
        ):
            raise ValueError("incomplete Paper-TWN matrix: view path sets differ")
        if set(indexed) != {
            (view, path)
            for view in self.REQUIRED_VIEWS
            for path in paths_by_view[view]
        }:
            raise ValueError("incomplete Paper-TWN matrix")
        return indexed

    def build(
        self, sources: tuple[TernaryRoutingSource, ...]
    ) -> TernaryRoutingManifest:
        source_by_parent = {source.parent_name: source for source in sources}
        if len(source_by_parent) != len(sources):
            raise ValueError("duplicate Paper-TWN routing parent")
        if tuple(source_by_parent) != self.REQUIRED_PARENTS:
            raise ValueError(
                "Paper-TWN routing requires parents in order: "
                + ", ".join(self.REQUIRED_PARENTS)
            )
        indexed = {
            parent: self._index(source_by_parent[parent])
            for parent in self.REQUIRED_PARENTS
        }
        path_sets = {
            parent: {
                path for view, path in parent_index if view == "deployment"
            }
            for parent, parent_index in indexed.items()
        }
        first_paths = path_sets[self.REQUIRED_PARENTS[0]]
        if not first_paths or any(paths != first_paths for paths in path_sets.values()):
            raise ValueError("incomplete Paper-TWN matrix: parent path sets differ")

        worst: dict[str, dict[str, float | int | str]] = {}
        safe: list[str] = []
        balanced: list[str] = []
        for path in sorted(first_paths):
            cells = [
                indexed[parent][(view, path)]
                for parent in self.REQUIRED_PARENTS
                for view in self.REQUIRED_VIEWS
            ]
            nrmse = max(
                self._finite(
                    self._mapping(cell["numeric"], "Paper-TWN numeric")[
                        "normalized_rmse"
                    ],
                    "Paper-TWN normalized_rmse",
                )
                for cell in cells
            )
            cosine = min(
                self._finite(
                    self._mapping(cell["numeric"], "Paper-TWN numeric")["cosine"],
                    "Paper-TWN cosine",
                )
                for cell in cells
            )
            region_values = {str(cell["region"]) for cell in cells}
            element_values = {int(cell["elements"]) for cell in cells}
            if len(region_values) != 1 or len(element_values) != 1:
                raise ValueError(f"Paper-TWN layer identity drifted: {path}")
            region = region_values.pop()
            elements = element_values.pop()
            worst[path] = {
                "region": region,
                "elements": elements,
                "worst_normalized_rmse": nrmse,
                "worst_cosine": cosine,
            }
            for tier, threshold in self.THRESHOLDS.items():
                if (
                    nrmse <= threshold["maximum_normalized_rmse"]
                    and cosine >= threshold["minimum_cosine"]
                ):
                    (safe if tier == "safe" else balanced).append(path)

        safe_candidates = tuple(safe)
        balanced_candidates = tuple(balanced)
        safe_regions = Counter(str(worst[path]["region"]) for path in safe_candidates)
        balanced_regions = Counter(
            str(worst[path]["region"]) for path in balanced_candidates
        )
        return TernaryRoutingManifest(
            schema_version=1,
            manifest_id="full35-paper-twn-routing-candidates-v1",
            status="static_candidates_only_not_validated_not_authorized",
            thresholds={
                tier: dict(values) for tier, values in self.THRESHOLDS.items()
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
                "safe": len(safe_candidates),
                "balanced": len(balanced_candidates),
                "safe_weight_elements": sum(
                    int(worst[path]["elements"]) for path in safe_candidates
                ),
                "balanced_weight_elements": sum(
                    int(worst[path]["elements"]) for path in balanced_candidates
                ),
                "safe_regions": dict(sorted(safe_regions.items())),
                "balanced_regions": dict(sorted(balanced_regions.items())),
            },
            safe_candidates=safe_candidates,
            balanced_candidates=balanced_candidates,
            per_path_worst_case={
                path: worst[path] for path in balanced_candidates
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
