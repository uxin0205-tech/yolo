"""CPU-only V1–V3 artifact preparation command."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .diagnostic_manifest import DiagnosticManifestBuilder, DiagnosticManifestSpec
from .full35_adapter import Full35ActivationAdapter, Full35ActivationPolicy
from .weight_formats import WeightAnalysisPlan, WeightFormatAnalyzer
from .weight_views import Full35WeightViewAdapter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WeightParentSpec:
    """One immutable activation-conditioned weight-analysis parent."""

    activation: str
    policy_id: str
    role: str
    checkpoint: Path
    sha256: str
    future_active: bool = True


_PARENTS = {
    "qsilu_pq": WeightParentSpec(
        activation="qsilu_pq",
        policy_id="qsilu_pq--lsq-plus-a8",
        role="accuracy",
        checkpoint=Path(
            "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
            "short-recovery-v2-lr01-uniform-qsilu-pq-seed1/"
            "inference/best_joint.pt"
        ),
        sha256="7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e",
    ),
    "hardswish": WeightParentSpec(
        activation="hardswish",
        policy_id="hardswish--lsq-plus-a8",
        role="standard_hardware",
        checkpoint=Path(
            "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
            "short-recovery-v2-lr01-uniform-hardswish-seed1/"
            "inference/best_joint.pt"
        ),
        sha256="79e0e4f615a7d8b82da4fd244165d2c035d7a523681833392d30b66e57177731",
    ),
    "poly_quality": WeightParentSpec(
        activation="poly_quality",
        policy_id="poly_quality--lsq-plus-a8",
        role="numeric",
        checkpoint=Path(
            "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
            "short-recovery-v2-lr01-uniform-poly-quality-seed1/"
            "inference/best_joint.pt"
        ),
        sha256="eedd48007345bc8dfd2179819b932bdf1dd40b25a654a7eb82c53796885ac968",
        future_active=False,
    ),
    "poly_shift": WeightParentSpec(
        activation="poly_shift",
        policy_id="poly_shift--lsq-plus-a8",
        role="hardware",
        checkpoint=Path(
            "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
            "short-recovery-v2-lr01-uniform-poly-shift-seed1/"
            "inference/best_joint.pt"
        ),
        sha256="8783248a513329ae80e3f659e6f3ce617e4bcca9fca9530b1f95448de5e19713",
    ),
}


def active_parent_names() -> tuple[str, ...]:
    """Return parents admitted to new experiments in deterministic order."""

    return tuple(name for name, parent in _PARENTS.items() if parent.future_active)


def historical_parent_names() -> tuple[str, ...]:
    """Return excluded parents retained only for artifact reconstruction."""

    return tuple(name for name, parent in _PARENTS.items() if not parent.future_active)


def active_parent_specs() -> tuple[tuple[str, WeightParentSpec], ...]:
    """Return active parent names and immutable provenance in matrix order."""

    return tuple(
        (name, parent) for name, parent in _PARENTS.items() if parent.future_active
    )


def parent_spec(name: str) -> WeightParentSpec:
    """Resolve one reviewed parent without exposing the mutable registry."""

    try:
        return _PARENTS[name]
    except KeyError as error:
        raise ValueError(f"unknown weight parent: {name}") from error


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _regions(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("regions must not be empty")
    return result


def activation_region_assignment(value: str) -> tuple[str, str]:
    """Parse one explicit ``REGION=ACTIVATION`` placement."""

    region, separator, activation = value.partition("=")
    region = region.strip()
    activation = activation.strip()
    if not separator or not region or not activation:
        raise argparse.ArgumentTypeError(
            "activation region assignment must be REGION=ACTIVATION"
        )
    return region, activation


def weight_analysis_plan(
    profile: str,
    *,
    regions: tuple[str, ...] | None = None,
) -> WeightAnalysisPlan:
    """Build one named CPU matrix without conflating static and learned formats."""

    if profile == "main":
        return WeightAnalysisPlan(
            regions=regions,
            uniform_bits=(8, 7, 6, 5, 4),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("mse",),
            fixed_sd4_granularities=("per_tensor", "per_output_channel"),
            fixed_sd4_scale_methods=("max", "mse"),
            include_fixed_sd4=True,
            include_paper_twn=True,
        )
    if profile == "structured":
        return WeightAnalysisPlan(
            regions=regions,
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("optimal_scaled_codebook",),
            fixed_sd4_granularities=("per_output_channel",),
            fixed_sd4_scale_methods=("optimal_scaled_codebook",),
            include_fixed_sd4=True,
            include_paper_twn=True,
            include_filterwise_twn=True,
            include_exact_scaled_ternary=True,
        )
    if profile == "full":
        return WeightAnalysisPlan(regions=regions)
    raise ValueError(f"unknown weight analysis profile: {profile}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Full35 quantization V1-V3 artifacts without GPU",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "artifacts/manifests/full35-diagnostic-cal32-probe64-v1.json"
        ),
    )
    parent_group = parser.add_mutually_exclusive_group()
    parent_group.add_argument("--parent", choices=active_parent_names())
    parent_group.add_argument(
        "--historical-parent",
        choices=historical_parent_names(),
        help="Rebuild frozen evidence only; never adds the parent to active matrices",
    )
    parser.add_argument("--regions", type=_regions)
    parser.add_argument(
        "--profile",
        choices=("main", "structured", "full"),
        default="main",
        help=(
            "main is the legacy matrix; structured adds optimal SD4, Paper-TWN, "
            "filterwise TWN, exact ternary and one exact W4 control; full adds "
            "all scale ablations"
        ),
    )
    parser.add_argument(
        "--activation-region",
        action="append",
        type=activation_region_assignment,
        default=[],
        metavar="REGION=ACTIVATION",
        help="Repeatable regional override on the selected checkpoint root",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Build dual-view parity without repeating weight-only analysis",
    )
    parser.add_argument("--view-output", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args(argv)

    diagnostic = DiagnosticManifestBuilder().build(DiagnosticManifestSpec())
    _atomic_json(args.diagnostic_output, diagnostic.to_dict())
    parent_name = args.parent or args.historical_parent
    if args.activation_region and parent_name is None:
        parser.error("--activation-region requires --parent or --historical-parent")
    if args.view_only and parent_name is None:
        parser.error("--view-only requires --parent or --historical-parent")
    if parent_name is None:
        print(
            json.dumps(
                {
                    "gpu_used": False,
                    "diagnostic_output": str(args.diagnostic_output.resolve()),
                    "diagnostic_sha256": diagnostic.sha256,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.view_output is None or (
        args.analysis_output is None and not args.view_only
    ):
        parser.error(
            "a parent requires --view-output and, unless --view-only, --analysis-output"
        )
    if args.view_only and args.analysis_output is not None:
        parser.error("--view-only cannot be combined with --analysis-output")

    parent = _PARENTS[parent_name]
    policy = Full35ActivationPolicy(
        activation=parent.activation,
        bits=8,
        region_assignments=tuple(args.activation_region),
    )
    built = Full35ActivationAdapter().build(
        policy,
        checkpoint=parent.checkpoint,
        checkpoint_sha256=parent.sha256,
    )
    views = Full35WeightViewAdapter().build(built.model)
    view_payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "full35_weight_dual_view_manifest",
        "formal_training": False,
        "gpu_used": False,
        "parent": {
            "activation": parent.activation,
            "policy_id": policy.policy_id,
            "role": parent.role,
            "checkpoint": str(parent.checkpoint),
            "checkpoint_sha256": parent.sha256,
        },
        "parity": views.manifest.to_dict(),
    }
    if policy.region_assignments:
        view_payload["parent"].update(  # type: ignore[union-attr]
            {
                "checkpoint_root_policy_id": parent.policy_id,
                "activation_policy_id": policy.activation_policy_id,
                "region_assignments": [
                    {"region": region, "activation": activation}
                    for region, activation in policy.region_assignments
                ],
                "activation_counts": built.activation_counts,
            }
        )
    _atomic_json(args.view_output, view_payload)

    if args.view_only:
        print(
            json.dumps(
                {
                    "gpu_used": False,
                    "parent": policy.policy_id,
                    "view_output": str(args.view_output.resolve()),
                    "analysis_skipped": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    analysis_plan = weight_analysis_plan(args.profile, regions=args.regions)
    analysis = WeightFormatAnalyzer().analyze(views, analysis_plan)
    analysis_payload = analysis.to_dict()
    analysis_payload.update(
        {
            "kind": "full35_static_weight_format_analysis",
            "profile": args.profile,
            "parent": view_payload["parent"],
            "diagnostic_manifest_sha256": diagnostic.sha256,
            "view_manifest": str(args.view_output.resolve()),
        }
    )
    assert args.analysis_output is not None
    _atomic_json(args.analysis_output, analysis_payload)
    print(
        json.dumps(
            {
                "gpu_used": False,
                "parent": policy.policy_id,
                "view_output": str(args.view_output.resolve()),
                "analysis_output": str(args.analysis_output.resolve()),
                "measurements": len(analysis.measurements),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
