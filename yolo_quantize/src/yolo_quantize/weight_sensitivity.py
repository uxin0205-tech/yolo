"""Reviewed PTQ bit-sensitivity matrix for Full35."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .activation_smoke import (
    _atomic_json,
    _compare_outputs,
    _deployment_comparison,
    _letterbox,
    _path_evidence,
)
from .diagnostic_manifest import DiagnosticManifest
from .full35_adapter import Full35ActivationAdapter, Full35ActivationPolicy
from .weight_quantization import (
    Full35WeightRegionCatalog,
    ScaleMethod,
    UniformWeightSpec,
    WeightQuantizationAdapter,
)
from .weight_views import Full35WeightViewAdapter

_RECORDED_RUN_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WeightParent:
    """One indivisible activation policy and its recovery checkpoint."""

    activation: str
    policy_id: str
    role: str
    checkpoint: Path
    sha256: str


@dataclass(frozen=True)
class WeightSensitivityCell:
    """One no-training isolated-region PTQ comparison."""

    parent: WeightParent
    region: str
    bits: int
    scale_method: ScaleMethod
    formal_training: bool = False

    @property
    def format_id(self) -> str:
        return f"w{self.bits}"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.parent.policy_id}--{self.region}--"
            f"{self.format_id}-{self.scale_method}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "activation": self.parent.activation,
            "activation_policy_id": self.parent.policy_id,
            "activation_role": self.parent.role,
            "parent_checkpoint": str(self.parent.checkpoint),
            "parent_sha256": self.parent.sha256,
            "region": self.region,
            "weight_bits": self.bits,
            "weight_format": self.format_id,
            "scale_method": self.scale_method,
            "formal_training": self.formal_training,
        }


@dataclass(frozen=True)
class WeightSensitivityStudy:
    """Validated expansion of the reviewed Full35 PTQ plan."""

    config_path: Path
    config_sha256: str
    parents: tuple[WeightParent, ...]
    regions: tuple[str, ...]
    expected_catalog: dict[str, object]
    expected_master_catalog: dict[str, object]
    graph_view: str
    execution_authorized: bool
    execution_authorization_id: str | None
    authorized_cell_ids: tuple[str, ...]
    diagnostic_manifest: Path | None
    diagnostic_manifest_sha256: str | None
    diagnostic_probe_per_task: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> WeightSensitivityStudy:
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("weight sensitivity plan must be a mapping")
        activation_payload = payload.get("activation_policies", {})
        if activation_payload.get("new_silu_runs") is not False:
            raise ValueError("weight plan must explicitly disable new SiLU runs")
        raw_parents = activation_payload.get("primary_a8")
        if not isinstance(raw_parents, list) or not raw_parents:
            raise ValueError("weight plan has no primary A8 parents")
        parents: list[WeightParent] = []
        for item in raw_parents:
            policy_id = str(item["policy_id"])
            activation = policy_id.split("--", maxsplit=1)[0]
            checkpoint = Path(item["parent_checkpoint"]).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = (config_path.parent / checkpoint).resolve()
            else:
                checkpoint = checkpoint.resolve()
            sha256 = str(item["parent_sha256"])
            if len(sha256) != 64:
                raise ValueError(f"invalid parent SHA-256 for {policy_id}")
            parents.append(
                WeightParent(
                    activation=activation,
                    policy_id=policy_id,
                    role=str(item["role"]),
                    checkpoint=checkpoint,
                    sha256=sha256,
                )
            )
        raw_regions = payload.get("regions")
        if not isinstance(raw_regions, list) or not raw_regions:
            raise ValueError("weight plan has no regions")
        ordered = sorted(raw_regions, key=lambda item: int(item["order"]))
        regions = tuple(str(item["id"]) for item in ordered)
        if len(set(regions)) != len(regions):
            raise ValueError("weight plan region IDs must be unique")
        inventory = payload["graph_inventory"]
        master_inventory = inventory.get("master", inventory)
        deployment_inventory = inventory.get("deployment", inventory)

        def catalog_contract(selected: Mapping[str, object]) -> dict[str, object]:
            return {
                "totals": {
                    "modules": int(selected["total_conv_linear_modules"]),
                    "weight_elements": int(selected["total_weight_elements"]),
                    "deployment_modules": int(selected["deployment_candidate_modules"]),
                    "deployment_weight_elements": int(
                        selected["deployment_candidate_weight_elements"]
                    ),
                    "training_only_modules": int(
                        selected["training_only_excluded_modules"]
                    ),
                    "training_only_weight_elements": int(
                        selected["training_only_excluded_weight_elements"]
                    ),
                    "protected_modules": int(selected["binary_qk_protected_modules"]),
                    "protected_weight_elements": int(
                        selected["binary_qk_protected_weight_elements"]
                    ),
                },
                "deployment_regions": {
                    str(item["id"]): {
                        "modules": int(item["modules"]),
                        "weight_elements": int(item["weights"]),
                    }
                    for item in ordered
                },
            }

        graph_view = str(payload.get("graph_view", "legacy_unfused_master"))
        if graph_view not in {"legacy_unfused_master", "bn_folded_deployment"}:
            raise ValueError("unsupported PTQ graph_view")
        execution_authorized = payload.get("execution_authorized", False)
        if not isinstance(execution_authorized, bool):
            raise TypeError("execution_authorized must be a boolean")
        authorization_id: str | None = None
        authorized_cell_ids: tuple[str, ...] = ()
        if execution_authorized:
            authorization_payload = payload.get("execution_authorization")
            if not isinstance(authorization_payload, dict):
                raise ValueError(
                    "authorized execution requires an execution_authorization mapping"
                )
            authorization_id = str(
                authorization_payload.get("authorization_id", "")
            ).strip()
            if not authorization_id:
                raise ValueError("execution authorization_id must not be empty")
            raw_authorized_cells = authorization_payload.get("cells")
            if not isinstance(raw_authorized_cells, list) or not raw_authorized_cells:
                raise ValueError("execution authorization must list at least one cell")
            authorized_cell_ids = tuple(str(item) for item in raw_authorized_cells)
            if len(set(authorized_cell_ids)) != len(authorized_cell_ids):
                raise ValueError("execution authorization cells must be unique")
        diagnostic_payload = payload.get("diagnostic", {})
        if not isinstance(diagnostic_payload, dict):
            raise TypeError("diagnostic plan must be a mapping")
        raw_manifest = diagnostic_payload.get("manifest")
        diagnostic_manifest: Path | None = None
        if raw_manifest is not None:
            diagnostic_manifest = Path(str(raw_manifest)).expanduser()
            if not diagnostic_manifest.is_absolute():
                diagnostic_manifest = (
                    config_path.parents[2] / diagnostic_manifest
                ).resolve()
            else:
                diagnostic_manifest = diagnostic_manifest.resolve()
        raw_manifest_sha256 = diagnostic_payload.get("manifest_sha256")
        diagnostic_manifest_sha256: str | None = None
        if raw_manifest_sha256 is not None:
            diagnostic_manifest_sha256 = str(raw_manifest_sha256)
            if len(diagnostic_manifest_sha256) != 64:
                raise ValueError("diagnostic manifest SHA-256 must have 64 hex digits")
            try:
                int(diagnostic_manifest_sha256, 16)
            except ValueError as error:
                raise ValueError(
                    "diagnostic manifest SHA-256 must have 64 hex digits"
                ) from error
        if (diagnostic_manifest is None) != (diagnostic_manifest_sha256 is None):
            raise ValueError(
                "diagnostic manifest path and SHA-256 must be declared together"
            )
        diagnostic_probe_per_task = int(
            diagnostic_payload.get("runner_probe_per_task", 1)
        )
        if diagnostic_probe_per_task != 1:
            raise ValueError(
                "the current output-comparison interface requires exactly one "
                "fixed diagnostic probe per task"
            )
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            parents=tuple(parents),
            regions=regions,
            expected_catalog=catalog_contract(deployment_inventory),
            expected_master_catalog=catalog_contract(master_inventory),
            graph_view=graph_view,
            execution_authorized=execution_authorized,
            execution_authorization_id=authorization_id,
            authorized_cell_ids=authorized_cell_ids,
            diagnostic_manifest=diagnostic_manifest,
            diagnostic_manifest_sha256=diagnostic_manifest_sha256,
            diagnostic_probe_per_task=diagnostic_probe_per_task,
        )

    def load_diagnostic_manifest(
        self,
        *,
        verify_files: bool,
    ) -> DiagnosticManifest:
        """Load only the manifest independently pinned by the reviewed plan."""

        if self.diagnostic_manifest is None or self.diagnostic_manifest_sha256 is None:
            raise RuntimeError(
                "reviewed PTQ execution requires a pinned diagnostic manifest"
            )
        manifest = DiagnosticManifest.from_json(
            self.diagnostic_manifest,
            verify_files=verify_files,
        )
        if manifest.sha256 != self.diagnostic_manifest_sha256:
            raise RuntimeError(
                "diagnostic manifest is not pinned by the reviewed plan: "
                f"{manifest.sha256} != {self.diagnostic_manifest_sha256}"
            )
        return manifest

    def cells(
        self,
        *,
        activations: tuple[str, ...] | None = None,
        regions: tuple[str, ...] | None = None,
        bits: tuple[int, ...] = (8, 4),
        scale_method: ScaleMethod = "mse_grid_v1",
    ) -> tuple[WeightSensitivityCell, ...]:
        selected_activations = (
            tuple(parent.activation for parent in self.parents)
            if activations is None
            else activations
        )
        parent_by_activation = {parent.activation: parent for parent in self.parents}
        unknown_activations = tuple(
            name for name in selected_activations if name not in parent_by_activation
        )
        if unknown_activations:
            raise ValueError(
                "unknown activation selection: " + ", ".join(unknown_activations)
            )
        selected_regions = self.regions if regions is None else regions
        unknown_regions = tuple(
            region for region in selected_regions if region not in self.regions
        )
        if unknown_regions:
            raise ValueError("unknown region selection: " + ", ".join(unknown_regions))
        if (
            len(set(selected_activations)) != len(selected_activations)
            or len(set(selected_regions)) != len(selected_regions)
            or len(set(bits)) != len(bits)
        ):
            raise ValueError("activation, region, and bit selections must be unique")
        for value in bits:
            UniformWeightSpec(bits=value, scale_method=scale_method)
        return tuple(
            WeightSensitivityCell(
                parent=parent_by_activation[activation],
                region=region,
                bits=value,
                scale_method=scale_method,
            )
            for activation in selected_activations
            for region in selected_regions
            for value in bits
        )

    def require_execution_authorized(
        self,
        cells: Sequence[WeightSensitivityCell],
    ) -> None:
        """Fail before model loading unless every selected cell is reviewed."""

        if not self.execution_authorized:
            raise RuntimeError(
                "plan has execution_authorized=false; create a reviewed authorized "
                "revision instead of editing historical evidence"
            )
        selected_ids = tuple(cell.cell_id for cell in cells)
        outside = tuple(
            cell_id
            for cell_id in selected_ids
            if cell_id not in self.authorized_cell_ids
        )
        if outside:
            raise RuntimeError(
                "selected cells are outside the reviewed authorization: "
                + ", ".join(outside)
            )

    def authorized_cells(self) -> tuple[WeightSensitivityCell, ...]:
        """Resolve the exact heterogeneous cell whitelist in declared order."""

        if not self.execution_authorized or not self.authorized_cell_ids:
            raise RuntimeError("plan has no executable authorized-cell whitelist")
        parents = {parent.policy_id: parent for parent in self.parents}
        cells: list[WeightSensitivityCell] = []
        for cell_id in self.authorized_cell_ids:
            try:
                policy_id, region, encoded_format = cell_id.rsplit("--", maxsplit=2)
            except ValueError as error:
                raise ValueError(f"invalid authorized cell ID: {cell_id}") from error
            match = re.fullmatch(r"w([0-9]+)-(.+)", encoded_format)
            if match is None or policy_id not in parents or region not in self.regions:
                raise ValueError(f"invalid authorized cell ID: {cell_id}")
            bits = int(match.group(1))
            scale_method = match.group(2)
            UniformWeightSpec(
                bits=bits,
                scale_method=scale_method,  # type: ignore[arg-type]
            )
            cell = WeightSensitivityCell(
                parent=parents[policy_id],
                region=region,
                bits=bits,
                scale_method=scale_method,  # type: ignore[arg-type]
            )
            if cell.cell_id != cell_id:
                raise ValueError(f"authorized cell ID is not canonical: {cell_id}")
            cells.append(cell)
        return tuple(cells)


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("selection must not be empty")
    return items


def _parse_bits(value: str) -> tuple[int, ...]:
    try:
        bits = tuple(int(item) for item in _parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bits must be integers") from error
    return bits


def _build_contract(
    *,
    study: WeightSensitivityStudy,
    cells: tuple[WeightSensitivityCell, ...],
    calibration_paths: Mapping[str, tuple[Path, ...]],
    probe_paths: Mapping[str, Path],
    diagnostic_manifest: DiagnosticManifest,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "full35_isolated_region_weight_ptq_sensitivity",
        "plan": str(study.config_path),
        "plan_sha256": study.config_sha256,
        "execution_authorization": {
            "authorization_id": study.execution_authorization_id,
            "authorized_cell_ids": list(study.authorized_cell_ids),
        },
        "formal_training": False,
        "qat": False,
        "graph_view": study.graph_view,
        "diagnostic_only": True,
        "selection_claim": False,
        "promotion_authorized": False,
        "comparison_reference": "matched_activation_a8_fp32_weights",
        "activation_output_format": "lsq_plus_a8",
        "weight_quantization": {
            "family": "signed_uniform",
            "granularity": "per_output_channel",
            "rounding": "nearest_even",
            "representation": "fake_quant_dequantized_weights",
            "accumulator_contract": "int32",
        },
        "image_size": 640,
        "diagnostic_manifest": {
            "path": str(diagnostic_manifest.source_path),
            "sha256": diagnostic_manifest.sha256,
            "manifest_calibration_per_task": 32,
            "manifest_probe_per_task": 64,
            "runner_probe_per_task": study.diagnostic_probe_per_task,
            "probe_subset_policy": (
                "first_ordered_sample_per_task_from_fixed_64_manifest"
            ),
            "all_file_evidence_verified_before_execution": True,
        },
        "calibration_per_task": len(calibration_paths["detect"]),
        "calibration_split": "canonical_train_exemplars_only",
        "probe_split": "canonical_val_exemplars_only",
        "bbat5_dataset_id": "bbat5-v1",
        "bbat5_assignment_changed": False,
        "calibration_images": {
            task: _path_evidence(paths) for task, paths in calibration_paths.items()
        },
        "probe_images": {
            task: _path_evidence((path,)) for task, path in probe_paths.items()
        },
        "cells": [cell.to_dict() for cell in cells],
    }


def _run_ptq(
    *,
    study: WeightSensitivityStudy,
    cells: tuple[WeightSensitivityCell, ...],
    output: Path,
    device_index: int,
    resume: bool,
) -> int:
    diagnostic_manifest = study.load_diagnostic_manifest(verify_files=True)
    calibration_paths = {
        "detect": diagnostic_manifest.paths("calibration", "coco_detect"),
        "pose": diagnostic_manifest.paths("calibration", "bbat_pose"),
    }
    probe_paths = {
        "detect": diagnostic_manifest.paths("probe", "coco_detect")[0],
        "pose": diagnostic_manifest.paths("probe", "bbat_pose")[0],
    }
    calibration = {
        task: tuple(_letterbox(path, 640) for path in paths)
        for task, paths in calibration_paths.items()
    }
    probes = {task: _letterbox(path, 640) for task, path in probe_paths.items()}
    contract = _build_contract(
        study=study,
        cells=cells,
        calibration_paths=calibration_paths,
        probe_paths=probe_paths,
        diagnostic_manifest=diagnostic_manifest,
    )
    output = output.expanduser().resolve()
    results: dict[str, Any] = {}
    parent_runs: dict[str, Any] = {}
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"output already exists; use --resume or a new path: {output}"
            )
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("contract") != contract:
            raise RuntimeError("existing PTQ report contract differs; refusing resume")
        raw_results = existing.get("results", {})
        raw_parents = existing.get("parents", {})
        if isinstance(raw_results, dict):
            results.update(raw_results)
        if isinstance(raw_parents, dict):
            parent_runs.update(raw_parents)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "full35_isolated_region_weight_ptq_sensitivity",
        "status": "running",
        "contract": contract,
        "catalog_contract": study.expected_catalog,
        "parents": parent_runs,
        "results": results,
    }
    _atomic_json(output, payload)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device_index)
    weight_adapter = WeightQuantizationAdapter()

    for parent in study.parents:
        parent_cells = tuple(
            cell
            for cell in cells
            if cell.parent.policy_id == parent.policy_id
            and not (
                cell.cell_id in results
                and results[cell.cell_id].get("status") in {"passed", "diagnostic_pass"}
            )
        )
        if not parent_cells:
            continue
        print(
            f"[parent] load {parent.policy_id} ({len(parent_cells)} pending cells)",
            flush=True,
        )
        model = None
        references: dict[str, Any] = {}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_index)
            build_started = time.perf_counter()
            built = Full35ActivationAdapter().build(
                Full35ActivationPolicy(
                    activation=parent.activation,
                    bits=8,
                ),
                checkpoint=parent.checkpoint,
                checkpoint_sha256=parent.sha256,
            )
            views = Full35WeightViewAdapter().build(built.model)
            if views.manifest.master_catalog != study.expected_master_catalog:
                raise RuntimeError(
                    "Full35 master weight catalog differs from reviewed plan"
                )
            activation_applied = built.applied.rebind(views.deployment)
            model = activation_applied.model.cuda(device_index).eval()
            build_seconds = time.perf_counter() - build_started
            catalog = Full35WeightRegionCatalog.inspect(model)
            catalog_summary = catalog.summary()
            if catalog_summary != study.expected_catalog:
                raise RuntimeError("Full35 weight catalog differs from reviewed plan")

            calibration_started = time.perf_counter()
            with torch.inference_mode():
                for task, images in calibration.items():
                    for image in images:
                        model(image.cuda(device_index), task=task)
            torch.cuda.synchronize(device_index)
            calibration_seconds = time.perf_counter() - calibration_started
            ranges = activation_applied.observer_ranges()
            invalid_ranges = tuple(
                path
                for path, observed in ranges.items()
                if observed is None or observed[1] <= observed[0]
            )
            if invalid_ranges:
                raise RuntimeError(
                    "activation observers are missing or degenerate: "
                    + ", ".join(invalid_ranges)
                )
            activation_applied.freeze_observers()

            reference_started = time.perf_counter()
            with torch.inference_mode():
                references = {
                    task: model(image.cuda(device_index), task=task)
                    for task, image in probes.items()
                }
            torch.cuda.synchronize(device_index)
            reference_seconds = time.perf_counter() - reference_started
            parent_runs[parent.policy_id] = {
                "activation": parent.activation,
                "role": parent.role,
                "checkpoint": str(parent.checkpoint),
                "checkpoint_sha256": parent.sha256,
                "checkpoint_state_source": built.loaded_checkpoint.state_source,
                "activation_quantizers": activation_applied.quantizer_count,
                "graph_view": study.graph_view,
                "weight_view_parity": views.manifest.to_dict(),
                "activation_observer": {
                    "valid_sites": len(ranges),
                    "invalid_sites": list(invalid_ranges),
                    "minimum": min(
                        observed[0]
                        for observed in ranges.values()
                        if observed is not None
                    ),
                    "maximum": max(
                        observed[1]
                        for observed in ranges.values()
                        if observed is not None
                    ),
                },
                "catalog": catalog_summary,
                "timing_seconds": {
                    "build_and_gpu_transfer": build_seconds,
                    "activation_calibration": calibration_seconds,
                    "matched_reference_probe": reference_seconds,
                },
                "peak_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated(device_index) / 2**20
                ),
                "status": "ready",
            }
            payload["parents"] = parent_runs
            _atomic_json(output, payload)

            for index, cell in enumerate(parent_cells, start=1):
                print(
                    f"[{index}/{len(parent_cells)}] run {cell.cell_id}",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats(device_index)
                cell_started = time.perf_counter()
                try:
                    with weight_adapter.quantized(
                        model,
                        catalog=catalog,
                        region=cell.region,
                        spec=UniformWeightSpec(
                            bits=cell.bits,
                            scale_method=cell.scale_method,
                        ),
                    ) as applied:
                        quantization_seconds = time.perf_counter() - cell_started
                        forward_started = time.perf_counter()
                        with torch.inference_mode():
                            candidates = {
                                task: model(
                                    image.cuda(device_index),
                                    task=task,
                                )
                                for task, image in probes.items()
                            }
                        torch.cuda.synchronize(device_index)
                        forward_seconds = time.perf_counter() - forward_started
                        comparisons = {
                            task: {
                                **_compare_outputs(
                                    references[task],
                                    candidates[task],
                                ),
                                "deployment": _deployment_comparison(
                                    references[task],
                                    candidates[task],
                                    task=task,
                                ),
                            }
                            for task in probes
                        }
                        all_finite = all(
                            item["all_finite"] for item in comparisons.values()
                        )
                        same_structure = all(
                            item["same_structure"] for item in comparisons.values()
                        )
                        result = {
                            **cell.to_dict(),
                            "status": (
                                "diagnostic_pass"
                                if all_finite and same_structure
                                else "diagnostic_failed"
                            ),
                            "selection_claim": False,
                            "promotion_authorized": False,
                            "incremental_reference": (
                                "matched_activation_a8_fp32_weights"
                            ),
                            "coverage": {
                                "quantized_modules": applied.quantized_modules,
                                "weight_elements": applied.weight_elements,
                                "protected_modules": len(catalog.protected_sites),
                                "training_only_modules": len(
                                    catalog.training_only_sites
                                ),
                            },
                            "weight_numeric": applied.numeric,
                            "layer_weight_numeric": list(applied.site_metrics),
                            "tasks": comparisons,
                            "all_finite": all_finite,
                            "same_structure": same_structure,
                            "timing_seconds": {
                                "weight_quantization": quantization_seconds,
                                "probe_forward": forward_seconds,
                                "cell_total": (time.perf_counter() - cell_started),
                            },
                            "peak_gpu_memory_mib": (
                                torch.cuda.max_memory_allocated(device_index) / 2**20
                            ),
                        }
                        del candidates
                    results[cell.cell_id] = result
                except _RECORDED_RUN_ERRORS as error:
                    results[cell.cell_id] = {
                        **cell.to_dict(),
                        "status": "diagnostic_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    print(
                        f"[failed] {cell.cell_id}: {type(error).__name__}: {error}",
                        flush=True,
                    )
                payload["results"] = results
                _atomic_json(output, payload)
        except _RECORDED_RUN_ERRORS as error:
            parent_runs[parent.policy_id] = {
                "activation": parent.activation,
                "role": parent.role,
                "checkpoint": str(parent.checkpoint),
                "checkpoint_sha256": parent.sha256,
                "status": "setup_failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            for cell in parent_cells:
                results.setdefault(
                    cell.cell_id,
                    {
                        **cell.to_dict(),
                        "status": "setup_failed",
                        "error_type": type(error).__name__,
                        "error": f"parent setup failed: {error}",
                    },
                )
            payload["parents"] = parent_runs
            payload["results"] = results
            _atomic_json(output, payload)
            print(
                f"[parent failed] {parent.policy_id}: {type(error).__name__}: {error}",
                flush=True,
            )
        finally:
            del references
            if model is not None:
                del model
            if "built" in locals():
                del built
            if "activation_applied" in locals():
                del activation_applied
            if "views" in locals():
                del views
            gc.collect()
            torch.cuda.empty_cache()

    selected_results = tuple(results.get(cell.cell_id) for cell in cells)
    diagnostic_passed = sum(
        result is not None and result.get("status") in {"passed", "diagnostic_pass"}
        for result in selected_results
    )
    payload["status"] = (
        "diagnostic_completed"
        if diagnostic_passed == len(cells)
        else "diagnostic_completed_with_failures"
    )
    payload["summary"] = {
        "cells": len(cells),
        "diagnostic_passed": diagnostic_passed,
        "failed": len(cells) - diagnostic_passed,
        "formal_training": False,
        "selection_claim": False,
        "promotion_authorized": False,
    }
    _atomic_json(output, payload)
    return 0 if diagnostic_passed == len(cells) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full35 isolated-region PTQ bit sensitivity",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "configs/experiments/weight-region-sensitivity-plan-v2.yaml"
        ),
    )
    parser.add_argument("--activations", type=_parse_csv)
    parser.add_argument("--regions", type=_parse_csv)
    parser.add_argument("--bits", type=_parse_bits)
    parser.add_argument(
        "--scale-method",
        choices=(
            "max",
            "mse_grid_v1",
            "optimal_scaled_codebook",
            "mse",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "artifacts/reports/weight-ptq-sensitivity-v2.json"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--authorized-cells",
        action="store_true",
        help="select the plan's exact heterogeneous authorized-cell whitelist",
    )
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)

    study = WeightSensitivityStudy.from_yaml(args.plan)
    if args.authorized_cells and any(
        value is not None
        for value in (
            args.activations,
            args.regions,
            args.bits,
            args.scale_method,
        )
    ):
        parser.error(
            "--authorized-cells cannot be combined with activation, region, bit, "
            "or scale-method filters"
        )
    cells = (
        study.authorized_cells()
        if args.authorized_cells
        else study.cells(
            activations=args.activations,
            regions=args.regions,
            bits=(8, 4) if args.bits is None else args.bits,
            scale_method=(
                "mse_grid_v1" if args.scale_method is None else args.scale_method
            ),
        )
    )
    if args.list_only:
        print(
            json.dumps(
                {
                    "plan": str(study.config_path),
                    "plan_sha256": study.config_sha256,
                    "graph_view": study.graph_view,
                    "execution_authorized": study.execution_authorized,
                    "execution_authorization_id": (study.execution_authorization_id),
                    "authorized_cell_ids": list(study.authorized_cell_ids),
                    "diagnostic_manifest": (
                        str(study.diagnostic_manifest)
                        if study.diagnostic_manifest is not None
                        else None
                    ),
                    "diagnostic_manifest_sha256": (study.diagnostic_manifest_sha256),
                    "diagnostic_probe_per_task": (study.diagnostic_probe_per_task),
                    "diagnostic_only": True,
                    "selection_claim": False,
                    "cell_count": len(cells),
                    "cells": [cell.to_dict() for cell in cells],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.execute_reviewed_plan:
        parser.error(
            "execution requires the explicit --execute-reviewed-plan acknowledgement"
        )
    try:
        study.require_execution_authorized(cells)
    except RuntimeError as error:
        parser.error(str(error))
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        parser.error(f"CUDA device {args.device} is unavailable")
    return _run_ptq(
        study=study,
        cells=cells,
        output=args.output,
        device_index=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
