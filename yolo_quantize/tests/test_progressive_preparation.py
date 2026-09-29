from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from yolo_quantize.progressive_preparation import (
    LockedQATParentSpec,
    ProgressivePreparationLayout,
    ProgressiveWeightPreparation,
)
from yolo_quantize.weight_formats import WeightFormatAnalyzer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_parent_manifest(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.yaml"
    plan.write_text("plan: fixture\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps({"schema_version": 1, "epoch": 5, "metrics": {"x": 1.0}}),
        encoding="utf-8",
    )
    full_resume = tmp_path / "full-resume.pt"
    full_resume.write_bytes(b"full-resume")
    inference = tmp_path / "inference.pt"
    inference.write_bytes(b"inference")
    completion = tmp_path / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arm": "qat",
                "plan_sha256": _sha256(plan),
                "completed_stages": ["j3"],
                "epochs_completed": 11,
                "checkpoint_paths": {"best_joint": str(full_resume)},
                "checkpoint_sha256": {"best_joint": _sha256(full_resume)},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "locked-parent.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "parent_id": "v19-epoch5",
                "status": "locked_search_parent",
                "formal_validation": False,
                "selected_epoch": 5,
                "activation": {
                    "name": "poly_shift",
                    "bits": 8,
                    "quantizer": "lsq_plus",
                },
                "weight_policy": {
                    "format_id": "all-w8",
                    "deployment_modules": 148,
                    "deployment_weight_elements": 22571840,
                },
                "plan": {"path": str(plan), "sha256": _sha256(plan)},
                "completion": {
                    "path": str(completion),
                    "sha256": _sha256(completion),
                },
                "metrics": {"path": str(metrics), "sha256": _sha256(metrics)},
                "checkpoints": {
                    "full_resume": {
                        "path": str(full_resume),
                        "sha256": _sha256(full_resume),
                    },
                    "inference": {
                        "path": str(inference),
                        "sha256": _sha256(inference),
                    },
                },
                "gate": {
                    "decision": "green",
                    "map50_max_drop": 0.015,
                    "map50_95_max_drop": 0.04,
                    "worst_map50_delta": -0.014,
                    "worst_map50_95_delta": -0.013,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def test_locked_qat_parent_verifies_both_checkpoint_roles_and_epoch(
    tmp_path: Path,
) -> None:
    spec = LockedQATParentSpec.from_yaml(_locked_parent_manifest(tmp_path))

    assert spec.parent_id == "v19-epoch5"
    assert spec.selected_epoch == 5
    assert spec.activation_name == "poly_shift"
    assert spec.activation_bits == 8
    assert spec.full_resume_checkpoint.name == "full-resume.pt"
    assert spec.inference_checkpoint.name == "inference.pt"
    assert spec.gate_decision == "green"


def test_locked_qat_parent_rejects_metric_epoch_drift(tmp_path: Path) -> None:
    manifest = _locked_parent_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    metrics = Path(payload["metrics"]["path"])
    metrics.write_text(
        json.dumps({"schema_version": 1, "epoch": 4, "metrics": {"x": 1.0}}),
        encoding="utf-8",
    )
    payload["metrics"]["sha256"] = _sha256(metrics)
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="selected epoch"):
        LockedQATParentSpec.from_yaml(manifest)


def test_progressive_cpu_plan_compares_w4_sd4_and_three_ternary_controls() -> None:
    plan = ProgressiveWeightPreparation.analysis_plan()

    assert plan.view_names == ("deployment",)
    assert plan.uniform_bits == (4,)
    assert plan.uniform_granularities == ("per_output_channel",)
    assert plan.uniform_scale_methods == ("optimal_scaled_codebook",)
    assert plan.fixed_sd4_granularities == ("per_output_channel",)
    assert plan.fixed_sd4_scale_methods == ("optimal_scaled_codebook",)
    assert plan.include_fixed_sd4 is True
    assert plan.include_paper_twn is True
    assert plan.include_filterwise_twn is True
    assert plan.include_exact_scaled_ternary is True


class _TinyDeployment(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = nn.Module()
        self.graph.model = nn.ModuleList([nn.Module()])
        self.graph.model[0].conv = nn.Conv2d(1, 2, (1, 5), bias=False)
        with torch.no_grad():
            self.graph.model[0].conv.weight.copy_(
                torch.linspace(-1.0, 1.0, 10).reshape(2, 1, 1, 5)
            )


def test_progressive_summary_keeps_matched_formats_and_ranks_each_region() -> None:
    preparation = ProgressiveWeightPreparation()
    analysis = WeightFormatAnalyzer().analyze_deployment(
        _TinyDeployment(),
        preparation.analysis_plan(),
    )

    summary = preparation.summarize(analysis)

    assert summary["coverage"] == {
        "deployment_paths": 1,
        "measurements": 5,
        "formats_per_path": 5,
    }
    row = summary["path_rankings"][0]
    assert row["path"] == "graph.model.0.conv"
    assert row["region"] == "backbone_early"
    assert set(row["formats"]) == set(preparation.REQUIRED_FORMAT_IDS)
    assert row["best_ternary_format"] in preparation.TERNARY_FORMAT_IDS
    assert summary["region_rankings"]["backbone_early"][0]["path"] == row["path"]


def test_progressive_cpu_plan_can_include_w7_w6_w5_for_qsilu() -> None:
    preparation = ProgressiveWeightPreparation()
    format_ids = preparation.required_format_ids(include_intermediate_uniform_bits=True)
    plan = preparation.analysis_plan(include_intermediate_uniform_bits=True)
    analysis = WeightFormatAnalyzer().analyze_deployment(
        _TinyDeployment(),
        plan,
    )

    summary = preparation.summarize(
        analysis,
        required_format_ids=format_ids,
    )

    assert plan.uniform_bits == (7, 6, 5, 4)
    assert format_ids[:3] == (
        "uniform-w7-per_output_channel-optimal_scaled_codebook",
        "uniform-w6-per_output_channel-optimal_scaled_codebook",
        "uniform-w5-per_output_channel-optimal_scaled_codebook",
    )
    assert summary["coverage"] == {
        "deployment_paths": 1,
        "measurements": 8,
        "formats_per_path": 8,
    }
    assert set(summary["path_rankings"][0]["formats"]) == set(format_ids)


def test_progressive_layout_is_versioned_and_writer_refuses_overwrite(
    tmp_path: Path,
) -> None:
    layout = ProgressivePreparationLayout.default(tmp_path)
    assert layout.profile_path.name == (
        "v19-epoch5-progressive-weight-formats-cpu-v1.json"
    )
    assert layout.delivery_path.name == ("v28-progressive-weight-cpu-delivery-v1.yaml")
    layout.profile_path.parent.mkdir(parents=True)
    layout.profile_path.write_text('{"sealed": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ProgressiveWeightPreparation.write_new_json(
            layout.profile_path,
            {"sealed": False},
        )
