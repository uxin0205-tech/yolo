"""Locked V19 parent lineage and CPU-only progressive weight preparation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .weight_formats import (
    WeightAnalysisPlan,
    WeightFormatAnalysis,
    WeightFormatAnalyzer,
    WeightFormatMeasurement,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _verified_file(record: object, label: str) -> tuple[Path, str]:
    payload = _mapping(record, label)
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _resolve(payload["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(payload["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return path, actual


@dataclass(frozen=True)
class LockedQATParentSpec:
    """Hash-pinned separation of resumable and deployment QAT checkpoints."""

    config_path: Path
    config_sha256: str
    parent_id: str
    selected_epoch: int
    activation_name: str
    activation_bits: int
    activation_quantizer: str
    weight_format_id: str
    deployment_modules: int
    deployment_weight_elements: int
    plan_path: Path
    plan_sha256: str
    completion_path: Path
    completion_sha256: str
    metrics_path: Path
    metrics_sha256: str
    full_resume_checkpoint: Path
    full_resume_sha256: str
    inference_checkpoint: Path
    inference_sha256: str
    gate_decision: str
    worst_map50_delta: float
    worst_map50_95_delta: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> LockedQATParentSpec:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "locked QAT parent",
        )
        required = {
            "schema_version",
            "parent_id",
            "status",
            "formal_validation",
            "selected_epoch",
            "activation",
            "weight_policy",
            "plan",
            "completion",
            "metrics",
            "checkpoints",
            "gate",
        }
        if set(payload) != required:
            raise ValueError("locked QAT parent fields differ from schema")
        if payload["schema_version"] != 1:
            raise ValueError("locked QAT parent schema_version must be 1")
        if payload["status"] != "locked_search_parent":
            raise ValueError("QAT parent is not locked for search")
        if payload["formal_validation"] is not False:
            raise ValueError("locked search parent cannot authorize formal validation")
        parent_id = str(payload["parent_id"]).strip()
        selected_epoch = int(payload["selected_epoch"])
        if not parent_id or selected_epoch < 0:
            raise ValueError("locked QAT parent identity or epoch is invalid")

        activation = _mapping(payload["activation"], "locked activation")
        if set(activation) != {"name", "bits", "quantizer"}:
            raise ValueError("locked activation fields differ from schema")
        activation_name = str(activation["name"])
        activation_bits = int(activation["bits"])
        activation_quantizer = str(activation["quantizer"])
        if (
            activation_name not in {"qsilu_pq", "hardswish", "poly_shift"}
            or activation_bits < 2
            or activation_bits > 16
            or not activation_quantizer
        ):
            raise ValueError("locked activation declaration is invalid")

        weight_policy = _mapping(payload["weight_policy"], "locked weight policy")
        if set(weight_policy) != {
            "format_id",
            "deployment_modules",
            "deployment_weight_elements",
        }:
            raise ValueError("locked weight policy fields differ from schema")
        weight_format_id = str(weight_policy["format_id"])
        deployment_modules = int(weight_policy["deployment_modules"])
        deployment_weight_elements = int(weight_policy["deployment_weight_elements"])
        if (
            not weight_format_id
            or deployment_modules < 1
            or deployment_weight_elements < 1
        ):
            raise ValueError("locked weight policy declaration is invalid")

        plan_path, plan_sha256 = _verified_file(payload["plan"], "QAT plan")
        completion_path, completion_sha256 = _verified_file(
            payload["completion"], "QAT completion"
        )
        metrics_path, metrics_sha256 = _verified_file(
            payload["metrics"], "selected metrics"
        )
        checkpoints = _mapping(payload["checkpoints"], "locked checkpoints")
        if set(checkpoints) != {"full_resume", "inference"}:
            raise ValueError(
                "locked checkpoints must contain full_resume and inference"
            )
        full_resume, full_resume_sha256 = _verified_file(
            checkpoints["full_resume"], "full-resume checkpoint"
        )
        inference, inference_sha256 = _verified_file(
            checkpoints["inference"], "inference checkpoint"
        )

        completion = _mapping(
            json.loads(completion_path.read_text(encoding="utf-8")),
            "QAT completion payload",
        )
        if (
            completion.get("schema_version") != 1
            or completion.get("arm") != "qat"
            or completion.get("plan_sha256") != plan_sha256
            or completion.get("completed_stages") != ["j3"]
            or int(completion.get("epochs_completed", -1)) <= selected_epoch
        ):
            raise ValueError("QAT completion does not support the selected parent")
        checkpoint_paths = _mapping(
            completion.get("checkpoint_paths"), "completion checkpoint paths"
        )
        checkpoint_hashes = _mapping(
            completion.get("checkpoint_sha256"), "completion checkpoint hashes"
        )
        completed_best = _resolve(checkpoint_paths.get("best_joint", ""))
        if (
            completed_best != full_resume
            or checkpoint_hashes.get("best_joint") != full_resume_sha256
        ):
            raise ValueError("completion best_joint differs from locked full-resume")

        metrics = _mapping(
            json.loads(metrics_path.read_text(encoding="utf-8")),
            "selected metrics payload",
        )
        if (
            metrics.get("schema_version") != 1
            or int(metrics.get("epoch", -1)) != selected_epoch
            or not isinstance(metrics.get("metrics"), dict)
            or not metrics["metrics"]
        ):
            raise ValueError("selected metrics differ from the selected epoch")

        gate = _mapping(payload["gate"], "locked parent gate")
        if set(gate) != {
            "decision",
            "map50_max_drop",
            "map50_95_max_drop",
            "worst_map50_delta",
            "worst_map50_95_delta",
        }:
            raise ValueError("locked parent gate fields differ from schema")
        decision = str(gate["decision"])
        map50_drop = float(gate["map50_max_drop"])
        map50_95_drop = float(gate["map50_95_max_drop"])
        worst_map50 = float(gate["worst_map50_delta"])
        worst_map50_95 = float(gate["worst_map50_95_delta"])
        if (
            decision != "green"
            or map50_drop != 0.015
            or map50_95_drop != 0.04
            or worst_map50 < -map50_drop
            or worst_map50_95 < -map50_95_drop
        ):
            raise ValueError("locked QAT parent does not pass the active dual gate")

        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            parent_id=parent_id,
            selected_epoch=selected_epoch,
            activation_name=activation_name,
            activation_bits=activation_bits,
            activation_quantizer=activation_quantizer,
            weight_format_id=weight_format_id,
            deployment_modules=deployment_modules,
            deployment_weight_elements=deployment_weight_elements,
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            completion_path=completion_path,
            completion_sha256=completion_sha256,
            metrics_path=metrics_path,
            metrics_sha256=metrics_sha256,
            full_resume_checkpoint=full_resume,
            full_resume_sha256=full_resume_sha256,
            inference_checkpoint=inference,
            inference_sha256=inference_sha256,
            gate_decision=decision,
            worst_map50_delta=worst_map50,
            worst_map50_95_delta=worst_map50_95,
        )


@dataclass(frozen=True)
class ProgressivePreparationLayout:
    """Versioned immutable outputs for the V19 progressive CPU handoff."""

    project_root: Path
    parent_manifest: Path
    profile_path: Path
    delivery_path: Path

    @classmethod
    def default(
        cls,
        project_root: Path = _PROJECT_ROOT,
    ) -> ProgressivePreparationLayout:
        root = project_root.expanduser().resolve()
        return cls(
            project_root=root,
            parent_manifest=(
                root / "artifacts/manifests/v19-epoch5-locked-parent-v1.yaml"
            ),
            profile_path=(
                root / "artifacts/reports/"
                "v19-epoch5-progressive-weight-formats-cpu-v1.json"
            ),
            delivery_path=(
                root / "artifacts/manifests/v28-progressive-weight-cpu-delivery-v1.yaml"
            ),
        )


class ProgressiveWeightPreparation:
    """CPU matrix required before progressive backbone-to-head GPU search."""

    REQUIRED_FORMAT_IDS = (
        "uniform-w4-per_output_channel-optimal_scaled_codebook",
        "fixed-sd4-per_output_channel-optimal_scaled_codebook",
        "paper-twn-layerwise",
        "twn-v3-0.75-filterwise",
        "exact-scaled-ternary-per_tensor",
    )
    INTERMEDIATE_UNIFORM_FORMAT_IDS = (
        "uniform-w7-per_output_channel-optimal_scaled_codebook",
        "uniform-w6-per_output_channel-optimal_scaled_codebook",
        "uniform-w5-per_output_channel-optimal_scaled_codebook",
    )
    TERNARY_FORMAT_IDS = (
        "paper-twn-layerwise",
        "twn-v3-0.75-filterwise",
        "exact-scaled-ternary-per_tensor",
    )

    @classmethod
    def required_format_ids(
        cls,
        include_intermediate_uniform_bits: bool = False,
    ) -> tuple[str, ...]:
        if include_intermediate_uniform_bits:
            return cls.INTERMEDIATE_UNIFORM_FORMAT_IDS + cls.REQUIRED_FORMAT_IDS
        return cls.REQUIRED_FORMAT_IDS

    @staticmethod
    def analysis_plan(
        include_intermediate_uniform_bits: bool = False,
    ) -> WeightAnalysisPlan:
        return WeightAnalysisPlan(
            view_names=("deployment",),
            uniform_bits=(7, 6, 5, 4) if include_intermediate_uniform_bits else (4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("optimal_scaled_codebook",),
            fixed_sd4_granularities=("per_output_channel",),
            fixed_sd4_scale_methods=("optimal_scaled_codebook",),
            include_fixed_sd4=True,
            include_paper_twn=True,
            include_filterwise_twn=True,
            include_exact_scaled_ternary=True,
        )

    @staticmethod
    def write_new_json(path: Path, payload: dict[str, object]) -> None:
        destination = path.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {destination}")
        temporary.replace(destination)

    @staticmethod
    def write_new_yaml(path: Path, payload: dict[str, object]) -> None:
        destination = path.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            yaml.safe_dump(
                payload,
                allow_unicode=True,
                sort_keys=False,
                width=100,
            ),
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {destination}")
        temporary.replace(destination)

    @classmethod
    def summarize(
        cls,
        analysis: WeightFormatAnalysis,
        *,
        required_format_ids: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        format_ids = (
            cls.REQUIRED_FORMAT_IDS
            if required_format_ids is None
            else tuple(required_format_ids)
        )
        if not format_ids or len(set(format_ids)) != len(format_ids):
            raise ValueError(
                "progressive required formats must be non-empty and unique"
            )
        if tuple(analysis.view_catalogs) != ("deployment",):
            raise ValueError("progressive profile requires exactly one deployment view")
        grouped: dict[str, list[WeightFormatMeasurement]] = {}
        for item in analysis.measurements:
            if item.view != "deployment":
                raise ValueError("progressive profile contains a non-deployment view")
            grouped.setdefault(item.path, []).append(item)
        if not grouped:
            raise ValueError("progressive profile contains no deployment paths")

        path_rankings: list[dict[str, object]] = []
        aggregates: dict[str, dict[str, float | int]] = {}
        for path, items in grouped.items():
            by_format = {item.format_id: item for item in items}
            if set(by_format) != set(format_ids) or len(items) != len(format_ids):
                raise ValueError(f"progressive format coverage drifted for {path}")
            region = items[0].region
            elements = items[0].elements
            if any(
                item.region != region or item.elements != elements for item in items
            ):
                raise ValueError(f"progressive path identity drifted for {path}")

            encoded: dict[str, dict[str, object]] = {}
            for format_id in format_ids:
                item = by_format[format_id]
                packed_bytes = item.code_bytes + item.metadata_bytes
                encoded[format_id] = {
                    "family": item.family,
                    "bits": item.bits,
                    "granularity": item.granularity,
                    "scale_method": item.scale_method,
                    "scale_count": item.scale_count,
                    "code_bytes": item.code_bytes,
                    "metadata_bytes": item.metadata_bytes,
                    "packed_bytes": packed_bytes,
                    "normalized_rmse": float(item.numeric["normalized_rmse"]),
                    "sqnr_db": float(item.numeric["sqnr_db"]),
                    "cosine": float(item.numeric["cosine"]),
                    "zero_ratio": float(item.numeric["zero_ratio"]),
                    "occupied_codes": int(item.numeric["occupied_codes"]),
                }
                aggregate = aggregates.setdefault(
                    format_id,
                    {
                        "elements": 0,
                        "squared_error": 0.0,
                        "reference_energy": 0.0,
                        "cosine_weighted_sum": 0.0,
                        "zero_elements": 0.0,
                        "code_bytes": 0,
                        "metadata_bytes": 0,
                    },
                )
                distribution = item.distribution
                reference_energy = elements * (
                    float(distribution["standard_deviation"]) ** 2
                    + float(distribution["mean"]) ** 2
                )
                aggregate["elements"] = int(aggregate["elements"]) + elements
                aggregate["squared_error"] = (
                    float(aggregate["squared_error"])
                    + float(item.numeric["mse"]) * elements
                )
                aggregate["reference_energy"] = (
                    float(aggregate["reference_energy"]) + reference_energy
                )
                aggregate["cosine_weighted_sum"] = (
                    float(aggregate["cosine_weighted_sum"])
                    + float(item.numeric["cosine"]) * elements
                )
                aggregate["zero_elements"] = (
                    float(aggregate["zero_elements"])
                    + float(item.numeric["zero_ratio"]) * elements
                )
                aggregate["code_bytes"] = int(aggregate["code_bytes"]) + item.code_bytes
                aggregate["metadata_bytes"] = (
                    int(aggregate["metadata_bytes"]) + item.metadata_bytes
                )

            best_ternary = min(
                (by_format[format_id] for format_id in cls.TERNARY_FORMAT_IDS),
                key=lambda item: (
                    float(item.numeric["normalized_rmse"]),
                    item.code_bytes + item.metadata_bytes,
                    item.format_id,
                ),
            )
            exact_w4 = by_format[
                "uniform-w4-per_output_channel-optimal_scaled_codebook"
            ]
            fixed_sd4 = by_format[
                "fixed-sd4-per_output_channel-optimal_scaled_codebook"
            ]
            parent_w8_bytes = elements + exact_w4.metadata_bytes
            ternary_bytes = best_ternary.code_bytes + best_ternary.metadata_bytes
            path_rankings.append(
                {
                    "path": path,
                    "region": region,
                    "elements": elements,
                    "parent_w8_estimated_bytes": parent_w8_bytes,
                    "best_ternary_format": best_ternary.format_id,
                    "best_ternary_normalized_rmse": float(
                        best_ternary.numeric["normalized_rmse"]
                    ),
                    "best_ternary_gap_to_exact_w4": float(
                        best_ternary.numeric["normalized_rmse"]
                    )
                    - float(exact_w4.numeric["normalized_rmse"]),
                    "best_ternary_gap_to_fixed_sd4": float(
                        best_ternary.numeric["normalized_rmse"]
                    )
                    - float(fixed_sd4.numeric["normalized_rmse"]),
                    "best_ternary_packed_bytes": ternary_bytes,
                    "estimated_savings_from_parent_w8_bytes": (
                        parent_w8_bytes - ternary_bytes
                    ),
                    "formats": encoded,
                }
            )

        path_rankings.sort(
            key=lambda row: (
                float(row["best_ternary_normalized_rmse"]),
                -int(row["estimated_savings_from_parent_w8_bytes"]),
                str(row["path"]),
            )
        )
        region_rankings: dict[str, list[dict[str, object]]] = {}
        for row in path_rankings:
            region_rankings.setdefault(str(row["region"]), []).append(
                {
                    "path": row["path"],
                    "elements": row["elements"],
                    "best_ternary_format": row["best_ternary_format"],
                    "best_ternary_normalized_rmse": row["best_ternary_normalized_rmse"],
                    "best_ternary_gap_to_exact_w4": row["best_ternary_gap_to_exact_w4"],
                    "best_ternary_gap_to_fixed_sd4": row[
                        "best_ternary_gap_to_fixed_sd4"
                    ],
                    "estimated_savings_from_parent_w8_bytes": row[
                        "estimated_savings_from_parent_w8_bytes"
                    ],
                }
            )

        format_aggregates: dict[str, dict[str, float | int]] = {}
        for format_id in format_ids:
            aggregate = aggregates[format_id]
            elements = int(aggregate["elements"])
            squared_error = float(aggregate["squared_error"])
            energy = float(aggregate["reference_energy"])
            code_bytes = int(aggregate["code_bytes"])
            metadata_bytes = int(aggregate["metadata_bytes"])
            format_aggregates[format_id] = {
                "elements": elements,
                "normalized_rmse": math.sqrt(squared_error / max(energy, 1e-24)),
                "mean_cosine_weighted_by_elements": float(
                    aggregate["cosine_weighted_sum"]
                )
                / max(elements, 1),
                "zero_ratio_weighted_by_elements": float(aggregate["zero_elements"])
                / max(elements, 1),
                "code_bytes": code_bytes,
                "metadata_bytes": metadata_bytes,
                "packed_bytes": code_bytes + metadata_bytes,
            }
        return {
            "coverage": {
                "deployment_paths": len(grouped),
                "measurements": len(analysis.measurements),
                "formats_per_path": len(format_ids),
            },
            "required_format_ids": list(format_ids),
            "format_aggregates": format_aggregates,
            "path_rankings": path_rankings,
            "region_rankings": region_rankings,
        }

    def prepare_profile(
        self,
        parent: LockedQATParentSpec,
        *,
        output_path: Path,
        profile_id: str | None = None,
        include_intermediate_uniform_bits: bool = False,
    ) -> dict[str, object]:
        """Reconstruct one locked QAT deployment and seal all CPU rows."""

        required_format_ids = self.required_format_ids(
            include_intermediate_uniform_bits
        )
        resolved_profile_id = (
            "v19-epoch5-progressive-weight-formats-cpu-v1"
            if profile_id is None
            else profile_id.strip()
        )
        if not resolved_profile_id:
            raise ValueError("progressive CPU profile_id must not be empty")
        profile_kind = (
            "full35_v19_locked_parent_progressive_weight_cpu_profile"
            if profile_id is None
            else "full35_locked_parent_progressive_weight_cpu_profile"
        )
        destination = output_path.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {destination}")

        from .qat_plan import Full35QATPlan
        from .qat_runtime import Full35QATRuntime

        started = time.perf_counter()
        plan = Full35QATPlan.from_yaml(parent.plan_path)
        if plan.config_sha256 != parent.plan_sha256:
            raise ValueError("locked parent QAT plan digest changed after parsing")
        if (
            plan.activation.activation != parent.activation_name
            or plan.activation.bits != parent.activation_bits
        ):
            raise ValueError("locked parent activation differs from the QAT plan")
        if any(assignment.spec.format_id != "w8" for assignment in plan.assignments):
            raise ValueError("locked all-W8 parent plan contains a non-W8 assignment")

        loaded = Full35QATRuntime(plan).load_deployment_parent(
            parent.inference_checkpoint,
            checkpoint_sha256=parent.inference_sha256,
            full_resume_sha256=parent.full_resume_sha256,
            epoch=parent.selected_epoch,
        )
        devices = sorted(
            {tensor.device.type for tensor in loaded.model.state_dict().values()}
        )
        if devices != ["cpu"]:
            raise RuntimeError(f"progressive CPU profile escaped CPU: {devices}")
        catalog = loaded.catalog.summary()
        totals = _mapping(catalog.get("totals"), "locked deployment catalog totals")
        if (
            int(totals.get("deployment_modules", -1)) != parent.deployment_modules
            or int(totals.get("deployment_weight_elements", -1))
            != parent.deployment_weight_elements
            or int(totals.get("training_only_modules", -1)) != 0
        ):
            raise RuntimeError("locked deployment catalog differs from parent manifest")

        analysis = WeightFormatAnalyzer().analyze_deployment(
            loaded.model,
            self.analysis_plan(include_intermediate_uniform_bits),
        )
        expected_measurements = parent.deployment_modules * len(required_format_ids)
        if len(analysis.measurements) != expected_measurements:
            raise RuntimeError(
                "progressive CPU profile coverage drifted: "
                f"{len(analysis.measurements)} != {expected_measurements}"
            )
        summary = self.summarize(analysis, required_format_ids=required_format_ids)
        payload: dict[str, object] = analysis.to_dict()
        payload.update(
            {
                "kind": profile_kind,
                "profile_id": resolved_profile_id,
                "status": "completed",
                "execution_authorized": False,
                "formal_training": False,
                "formal_validation": False,
                "map_validation_run": False,
                "parent": {
                    "manifest": str(parent.config_path),
                    "manifest_sha256": parent.config_sha256,
                    "parent_id": parent.parent_id,
                    "selected_epoch": parent.selected_epoch,
                    "activation": {
                        "name": parent.activation_name,
                        "bits": parent.activation_bits,
                        "quantizer": parent.activation_quantizer,
                        "quantizer_count": loaded.activation.quantizer_count,
                    },
                    "weight_policy": {
                        "format_id": parent.weight_format_id,
                        "deployment_modules": parent.deployment_modules,
                        "deployment_weight_elements": (
                            parent.deployment_weight_elements
                        ),
                    },
                    "inference_checkpoint": {
                        "path": str(parent.inference_checkpoint),
                        "sha256": parent.inference_sha256,
                        "role": "materialized_w8_ptq_and_validation_parent",
                    },
                    "full_resume_checkpoint": {
                        "path": str(parent.full_resume_checkpoint),
                        "sha256": parent.full_resume_sha256,
                        "role": "fp32_shadow_optimizer_qat_resume_only",
                    },
                    "metrics": {
                        "path": str(parent.metrics_path),
                        "sha256": parent.metrics_sha256,
                    },
                    "gate": {
                        "decision": parent.gate_decision,
                        "worst_map50_delta": parent.worst_map50_delta,
                        "worst_map50_95_delta": parent.worst_map50_95_delta,
                    },
                },
                "runtime_reconstruction": {
                    "checkpoint_graph_schema": loaded.metadata.get(
                        "deployment_graph_schema"
                    ),
                    "legacy_graph_schema_inferred": loaded.legacy_schema_inferred,
                    "activation_mode": loaded.activation.mode,
                    "catalog": catalog,
                    "cpu_devices_verified": devices,
                },
                "analysis_contract": {
                    "scope": "all_148_locked_deployment_weight_paths",
                    "source_weight_state": "materialized_learned_w8_values",
                    "required_formats": list(required_format_ids),
                    "exact_solver": ("cvpr2021_optimal_scaled_codebook_event_sweep"),
                    "paper_twn_v2": "0.70_layerwise_static_proxy",
                    "paper_twn_v3": "0.75_per_output_filter_static_initialization",
                    "selection_claim": False,
                },
                "summary": summary,
                "elapsed_seconds": time.perf_counter() - started,
                "limitations": [
                    "static_weight_reconstruction_only",
                    "cpu_ranking_is_not_a_task_map_gate",
                    "no_gpu_used",
                    "no_training",
                    "no_formal_validation",
                    "qat_recovery_must_resume_the_pinned_full_resume_checkpoint",
                    "inference_checkpoint_must_not_be_used_as_a_training_resume",
                ],
            }
        )
        self.write_new_json(destination, payload)
        result = {
            "status": "completed",
            "gpu_used": False,
            "parent_id": parent.parent_id,
            "profile": str(destination),
            "profile_sha256": _sha256(destination),
            "deployment_paths": summary["coverage"]["deployment_paths"],
            "measurements": summary["coverage"]["measurements"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }
        del analysis, loaded
        gc.collect()
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile W4, Fixed-SD4 and three ternary controls on the locked V19 "
            "deployment parent without GPU"
        ),
    )
    layout = ProgressivePreparationLayout.default()
    parser.add_argument(
        "--parent-manifest",
        type=Path,
        default=layout.parent_manifest,
    )
    parser.add_argument("--output", type=Path, default=layout.profile_path)
    parser.add_argument("--profile-id")
    parser.add_argument("--include-intermediate-uniform-bits", action="store_true")
    parser.add_argument("--execute-cpu-profile", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_cpu_profile:
        parser.error("execution requires --execute-cpu-profile acknowledgement")
    parent = LockedQATParentSpec.from_yaml(args.parent_manifest)
    result = ProgressiveWeightPreparation().prepare_profile(
        parent,
        output_path=args.output,
        profile_id=args.profile_id,
        include_intermediate_uniform_bits=args.include_intermediate_uniform_bits,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
