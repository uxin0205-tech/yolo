"""CPU-only preparation of exact W4/Fixed-SD4 Full35 evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from .exact_routing import ExactRoutingSource, ExactWeightRouting
from .full35_adapter import Full35ActivationAdapter, Full35ActivationPolicy
from .integer_boundaries import IntegerBoundaryContract
from .preparation import active_parent_names, active_parent_specs, parent_spec
from .weight_formats import WeightAnalysisPlan, WeightFormatAnalyzer
from .weight_views import Full35WeightViewAdapter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExactPreparationLayout:
    """Versioned output paths for one complete CPU delivery."""

    project_root: Path
    report_paths: dict[str, Path]
    boundary_path: Path
    routing_path: Path
    delivery_path: Path

    @classmethod
    def default(cls, project_root: Path = _PROJECT_ROOT) -> ExactPreparationLayout:
        root = project_root.expanduser().resolve()
        reports = root / "artifacts/reports"
        manifests = root / "artifacts/manifests"
        return cls(
            project_root=root,
            report_paths={
                "qsilu_pq": reports
                / "weight-format-analysis-qsilu-pq-a8-exact-w4-sd4-v1.json",
                "hardswish": reports
                / "weight-format-analysis-hardswish-a8-exact-w4-sd4-v1.json",
                "poly_shift": reports
                / "weight-format-analysis-poly-shift-a8-exact-w4-sd4-v1.json",
            },
            boundary_path=(
                manifests / "full35-integer-boundary-contract-qsilu-pq-a8-v1.json"
            ),
            routing_path=manifests / "fixed-sd4-routing-candidates-v3.yaml",
            delivery_path=manifests / "exact-w4-sd4-cpu-delivery-v1.yaml",
        )


class ExactWeightPreparation:
    """Adapter from reviewed Full35 parents to immutable CPU evidence artifacts."""

    @staticmethod
    def analysis_plan() -> WeightAnalysisPlan:
        """Return the minimal fair grid-vs-exact W4/Fixed-SD4 matrix."""

        return WeightAnalysisPlan(
            view_names=("master", "deployment"),
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=(
                "mse_grid_v1",
                "optimal_scaled_codebook",
            ),
            fixed_sd4_granularities=("per_output_channel",),
            fixed_sd4_scale_methods=(
                "mse_grid_v1",
                "optimal_scaled_codebook",
            ),
            include_fixed_sd4=True,
            include_paper_twn=False,
        )

    @staticmethod
    def format_ids() -> tuple[str, ...]:
        return ExactWeightRouting.REQUIRED_FORMATS

    @staticmethod
    def write_new_json(path: Path, payload: dict[str, object]) -> None:
        """Atomically create JSON and refuse any historical overwrite."""

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
        """Atomically create YAML and refuse any historical overwrite."""

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

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path.resolve())

    def prepare_parent(
        self,
        parent_name: str,
        *,
        report_path: Path,
        boundary_path: Path | None = None,
    ) -> dict[str, object]:
        """Build one complete parent profile; never trains or validates mAP."""

        if parent_name not in active_parent_names():
            raise ValueError(f"inactive or unknown exact parent: {parent_name}")
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {report_path}")
        if boundary_path is not None and boundary_path.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {boundary_path}")
        if (parent_name == "qsilu_pq") != (boundary_path is not None):
            raise ValueError("only qsilu_pq must produce the integer boundary artifact")

        parent = parent_spec(parent_name)
        policy = Full35ActivationPolicy(activation=parent.activation, bits=8)
        built = Full35ActivationAdapter().build(
            policy,
            checkpoint=parent.checkpoint,
            checkpoint_sha256=parent.sha256,
        )
        views = Full35WeightViewAdapter().build(built.model)
        devices = sorted(
            {
                tensor.device.type
                for view in (views.master, views.deployment)
                for tensor in view.state_dict().values()
            }
        )
        if devices != ["cpu"]:
            raise RuntimeError(f"exact preparation escaped CPU: {devices}")

        analysis = WeightFormatAnalyzer().analyze(views, self.analysis_plan())
        expected_measurements = 148 * 2 * len(self.format_ids())
        if len(analysis.measurements) != expected_measurements:
            raise RuntimeError(
                "exact profile coverage drifted: "
                f"{len(analysis.measurements)} != {expected_measurements}"
            )
        payload = analysis.to_dict()
        payload.update(
            {
                "kind": "full35_exact_w4_sd4_static_analysis",
                "profile": "exact_w4_sd4_v1",
                "execution_authorized": False,
                "map_validation_run": False,
                "cpu_devices_verified": devices,
                "parent": {
                    "name": parent_name,
                    "activation": parent.activation,
                    "policy_id": policy.policy_id,
                    "role": parent.role,
                    "checkpoint": str(parent.checkpoint),
                    "checkpoint_sha256": parent.sha256,
                    "activation_counts": built.activation_counts,
                },
                "analysis_plan": {
                    "views": list(ExactWeightRouting.REQUIRED_VIEWS),
                    "formats": list(self.format_ids()),
                    "selection_scope": "all_148_deployment_weight_sites",
                    "scale_granularity": "per_output_channel",
                    "legacy_solver": "mse_grid_v1_ten_point_absmax",
                    "exact_solver": "cvpr2021_optimal_scaled_codebook_event_sweep",
                },
                "weight_view_parity": views.manifest.to_dict(),
                "limitations": [
                    "static_weight_reconstruction_only",
                    "no_layer_output_nrmse",
                    "no_topk_overlap",
                    "no_eight_metric_map",
                    "no_qat",
                    "no_gpu_run",
                ],
            }
        )
        self.write_new_json(report_path, payload)

        boundary_digest: str | None = None
        if boundary_path is not None:
            boundary = IntegerBoundaryContract().inspect_full35(
                views.deployment,
                activation_policy_id=policy.policy_id,
                checkpoint_sha256=parent.sha256,
                deployment_state_sha256=views.manifest.deployment_state_sha256,
            )
            boundary_payload = boundary.to_dict()
            boundary_payload.update(
                {
                    "kind": "full35_hybrid_integer_boundary_contract",
                    "weight_view_parity": views.manifest.to_dict(),
                    "source_profile": self._relative(report_path, _PROJECT_ROOT),
                }
            )
            self.write_new_json(boundary_path, boundary_payload)
            boundary_digest = _sha256(boundary_path)

        result: dict[str, object] = {
            "parent": parent_name,
            "gpu_used": False,
            "formal_training": False,
            "report": str(report_path.resolve()),
            "report_sha256": _sha256(report_path),
            "measurements": len(analysis.measurements),
        }
        if boundary_path is not None:
            result.update(
                {
                    "boundary": str(boundary_path.resolve()),
                    "boundary_sha256": boundary_digest,
                }
            )
        del analysis, views, built
        gc.collect()
        return result

    def build_routing(self, layout: ExactPreparationLayout) -> dict[str, object]:
        """Build v3 only after all three immutable exact reports exist."""

        if layout.routing_path.exists():
            raise FileExistsError(
                f"refusing to overwrite artifact: {layout.routing_path}"
            )
        sources: list[ExactRoutingSource] = []
        for parent_name, parent in active_parent_specs():
            path = layout.report_paths[parent_name]
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError(f"exact report root must be a mapping: {path}")
            sources.append(
                ExactRoutingSource(
                    parent_name=parent_name,
                    checkpoint_sha256=parent.sha256,
                    artifact_path=self._relative(path, layout.project_root),
                    artifact_sha256=_sha256(path),
                    payload=payload,
                )
            )
        manifest = ExactWeightRouting().build(tuple(sources))
        payload = manifest.to_dict()
        payload.update(
            {
                "date": "2026-09-01",
                "supersedes": "full35-fixed-sd4-routing-candidates-v2",
                "historical_manifest": (
                    "artifacts/manifests/fixed-sd4-routing-candidates-v2.yaml"
                ),
                "next_stage": "gpu_w8_bridge_then_output_sensitivity",
            }
        )
        self.write_new_yaml(layout.routing_path, payload)
        return {
            "gpu_used": False,
            "routing": str(layout.routing_path.resolve()),
            "routing_sha256": _sha256(layout.routing_path),
            "stable_candidates": len(manifest.stable_candidates),
        }

    def write_delivery(self, layout: ExactPreparationLayout) -> dict[str, object]:
        """Seal the CPU handoff and identify the first GPU-required action."""

        if layout.delivery_path.exists():
            raise FileExistsError(
                f"refusing to overwrite artifact: {layout.delivery_path}"
            )
        required = (
            *layout.report_paths.values(),
            layout.boundary_path,
            layout.routing_path,
        )
        missing = tuple(path for path in required if not path.is_file())
        if missing:
            raise FileNotFoundError(
                "CPU delivery inputs are missing: " + ", ".join(map(str, missing))
            )
        artifacts = [
            {
                "kind": f"exact_profile_{parent}",
                "path": self._relative(path, layout.project_root),
                "sha256": _sha256(path),
            }
            for parent, path in layout.report_paths.items()
        ]
        artifacts.extend(
            (
                {
                    "kind": "integer_boundary_contract",
                    "path": self._relative(layout.boundary_path, layout.project_root),
                    "sha256": _sha256(layout.boundary_path),
                },
                {
                    "kind": "fixed_sd4_routing_v3",
                    "path": self._relative(layout.routing_path, layout.project_root),
                    "sha256": _sha256(layout.routing_path),
                },
            )
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "delivery_id": "full35-exact-w4-sd4-cpu-delivery-v1",
            "date": "2026-09-01",
            "status": "cpu_prerequisites_complete_stopped_before_gpu",
            "gpu_used": False,
            "formal_training": False,
            "map_validation_run": False,
            "execution_authorized": False,
            "completed": [
                "hybrid_integer_boundary_reference_contract",
                "all_148_layer_exact_w4_fixed_sd4_profiles_for_three_active_parents",
                "cross_parent_cross_view_fixed_sd4_routing_v3",
            ],
            "artifacts": artifacts,
            "historical_evidence_retained": [
                "artifacts/manifests/fixed-sd4-routing-candidates-v2.yaml",
                "artifacts/reports/weight-format-analysis-qsilu-pq-a8-main-v2.json",
                "artifacts/reports/weight-format-analysis-hardswish-a8-main-v3.json",
                "artifacts/reports/weight-format-analysis-poly-shift-a8-main-v2.json",
            ],
            "first_gpu_required_stage": {
                "stage": "V4_W8_bridge",
                "requires_gpu": True,
                "execution_authorized": False,
                "work": [
                    "calibrate_and_freeze_lsq_plus_scales_and_offsets",
                    "measure_add_concat_saturation_and_output_scale_alignment",
                    "run_layer_output_nrmse_and_top300_overlap",
                    "run_eight_metric_COCO_person_and_BBAT5_validation",
                ],
            },
            "still_blocked_after_gpu_bridge": {
                "qat": "needs_fold_aware_bn_affine_contract_before_any_qat_run",
                "hardware_claim": (
                    "needs_target_profile_integer_reciprocal_kernels_and_board_measurement"
                ),
            },
        }
        self.write_new_yaml(layout.delivery_path, payload)
        return {
            "gpu_used": False,
            "delivery": str(layout.delivery_path.resolve()),
            "delivery_sha256": _sha256(layout.delivery_path),
            "status": payload["status"],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare exact W4/Fixed-SD4 CPU evidence; never uses GPU",
    )
    parser.add_argument("--parent", choices=active_parent_names())
    parser.add_argument("--routing-only", action="store_true")
    parser.add_argument("--delivery-only", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="Use a new root to rebuild without overwriting sealed artifacts",
    )
    args = parser.parse_args(argv)
    selected = sum(
        bool(value) for value in (args.parent, args.routing_only, args.delivery_only)
    )
    if selected != 1:
        parser.error("choose exactly one of --parent, --routing-only, --delivery-only")

    layout = ExactPreparationLayout.default(args.output_root)
    preparation = ExactWeightPreparation()
    if args.parent:
        result = preparation.prepare_parent(
            args.parent,
            report_path=layout.report_paths[args.parent],
            boundary_path=layout.boundary_path if args.parent == "qsilu_pq" else None,
        )
    elif args.routing_only:
        result = preparation.build_routing(layout)
    else:
        result = preparation.write_delivery(layout)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
