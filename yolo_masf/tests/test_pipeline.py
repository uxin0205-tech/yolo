from __future__ import annotations

import json
from pathlib import Path

import pytest

from masf_yolo.evaluation.profiling import HardwareProfile
from masf_yolo.contracts import PipelineState, sha256_file
from masf_yolo.pipeline import FormalPipeline, candidate_from_artifacts, normalize_profile, run_final_audit
from masf_yolo.workflow import PHASE1_STAGES, PipelineWorkflow, StageResult


def _complete_class_metrics() -> dict[str, object]:
    return {
        "per_class": {"ball": {"ap": 0.4}, "bat": {"ap": 0.5}},
        "class_diagnostics": {
            "ball": {"gt_count": 2, "prediction_count": 2},
            "bat": {"gt_count": 1, "prediction_count": 1},
        },
    }


def test_candidate_uses_exact_quality_and_hardware_fields() -> None:
    metrics = {
        "map50_95": 0.51,
        "ap_s": 0.41,
        "ball_recall": 0.71,
        "ball_ap_s": 0.37,
        "ball_subsets": {
            "tiny": {"recall": 0.61},
            "blur_proxy": {"recall": 0.56},
        },
    }
    profile = HardwareProfile(20, 30, 0.00000006, 40, 50, 2, 3, 60)

    candidate = candidate_from_artifacts("M2", metrics, profile)

    assert candidate.variant_id == "M2"
    assert candidate.map50_95 == 0.51
    assert candidate.tiny_recall == 0.61
    assert candidate.gflops == profile.gflops
    assert candidate.peak_activation == 50


def test_b0_profile_does_not_mislabel_p3_as_p2_activation() -> None:
    profile = HardwareProfile(20, 30, 0.00000006, 40, 50, 2, 3, 60)

    assert normalize_profile("B0", profile).p2_activation_bytes is None
    assert normalize_profile("M7", profile) == profile


def test_final_audit_fails_closed_when_any_required_artifact_is_missing(tmp_path: Path) -> None:
    result = run_final_audit(tmp_path)

    assert result["ok"] is False
    assert result["errors"]
    assert (tmp_path / "final_audit.json").is_file()


def test_final_audit_rejects_missing_or_selection_eligible_b0_reference(tmp_path: Path) -> None:
    result = run_final_audit(tmp_path)
    assert "missing B0 reference manifest" in result["errors"]

    reference = tmp_path / "references"
    reference.mkdir()
    (reference / "b0.json").write_text(
        json.dumps(
            {
                "reference_id": "B0",
                "checkpoint_hash": "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d",
                "data_exposed": True,
                "selection_eligible": True,
            }
        )
    )

    result = run_final_audit(tmp_path)
    assert "B0 reference must be data-exposed and selection-ineligible" in result["errors"]


def test_final_audit_rejects_metrics_without_ball_and_bat_sections(tmp_path: Path) -> None:
    metrics = tmp_path / "evaluation" / "val" / "b0" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}")

    result = run_final_audit(tmp_path)

    assert "missing val per_class class ball: B0" in result["errors"]
    assert "missing val per_class class bat: B0" in result["errors"]
    assert "missing val class_diagnostics class ball: B0" in result["errors"]
    assert "missing val class_diagnostics class bat: B0" in result["errors"]


def test_final_audit_accepts_one_complete_phase1_matrix(tmp_path: Path) -> None:
    training_stages = (
        "b1_a", "b1_b", "formal_m7", "formal_m0", "formal_m1", "formal_m2",
        "formal_m3", "formal_p3m", "sp2_a", "sp2_b", "sp2p_a", "sp2p_b",
    )
    for stage in training_stages:
        path = tmp_path / "training" / stage
        path.mkdir(parents=True)
        checkpoint = path / "canonical.pt"
        checkpoint.write_bytes(stage.encode())
        variant = "SP2M3" if stage.startswith("sp2p") else "SP2" if stage.startswith("sp2") else stage
        (path / "run.json").write_text(
            json.dumps(
                {
                    "canonical": str(checkpoint),
                    "canonical_hash": f"{stage}-hash",
                    "strict_reload": True,
                    "variant": variant,
                }
            )
        )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"selected": "M3", "val_hashes": {"M2": "a", "M3": "b"}})
    )
    for stage in ("sp2p_a", "sp2p_b"):
        path = tmp_path / "training" / stage / "run.json"
        record = json.loads(path.read_text())
        record.update(
            {
                "display_variant": "SP2P",
                "architecture_variant": "SP2M3",
                "selected_partial": "M3",
                "parent_hashes": {
                    "sp2_b": "sp2_b-hash",
                    "formal_m3": "formal_m3-hash",
                },
                "selection_hash": sha256_file(selection_path),
            }
        )
        path.write_text(json.dumps(record))
    for split in ("val", "test"):
        for variant in ("B0", "B1", "M7", "M0", "M1", "M2", "M3", "P3M", "SP2", "SP2P"):
            path = tmp_path / "evaluation" / split / variant.lower()
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(json.dumps(_complete_class_metrics()))
    for variant in ("B0", "B1", "M7", "M0", "M1", "M2", "M3", "P3M", "SP2", "SP2P"):
        path = tmp_path / "profiles" / variant.lower()
        path.mkdir(parents=True)
        (path / "profile.json").write_text("{}")
    reference = tmp_path / "references"
    reference.mkdir()
    (reference / "b0.json").write_text(
        json.dumps(
            {
                "reference_id": "B0",
                "checkpoint_hash": "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d",
                "data_exposed": True,
                "selection_eligible": False,
            }
        )
    )

    result = run_final_audit(tmp_path)

    assert result == {"ok": True, "errors": [], "best_partial": "M3"}


def test_final_audit_rejects_sp2p_lineage_that_disagrees_with_selection(tmp_path: Path) -> None:
    result = run_final_audit(tmp_path)
    assert result["ok"] is False

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"selected": "M3", "val_hashes": {"M2": "a", "M3": "b"}}))
    stage = tmp_path / "training" / "sp2p_b"
    stage.mkdir(parents=True)
    checkpoint = stage / "canonical.pt"
    checkpoint.write_bytes(b"sp2p")
    (stage / "run.json").write_text(
        json.dumps(
            {
                "canonical": str(checkpoint),
                "strict_reload": True,
                "architecture_variant": "SP2M2",
                "selected_partial": "M2",
                "selection_hash": sha256_file(selection_path),
                "parent_hashes": {"sp2_b": "x", "formal_m2": "y"},
            }
        )
    )

    result = run_final_audit(tmp_path)

    assert "SP2P lineage does not match BEST_PARTIAL" in result["errors"]


def test_formal_pipeline_reuses_predecessors_and_restarts_only_stale_smoke_m7(
    tmp_path: Path,
) -> None:
    workflow = PipelineWorkflow(
        tmp_path,
        pipeline_id="p1",
        common_input_hashes={"config": "c1"},
    )
    names = [stage.name for stage in PHASE1_STAGES]
    for name in names[: names.index("smoke_m7")]:
        workflow.run_stage(name, lambda name=name: StageResult({name: f"{name}-hash"}))
    stale = PipelineState(
        pipeline_id="p1",
        stage="smoke_m7",
        status="running",
        attempt=1,
        epoch=None,
        input_hashes={"config": "c1", "m7_gate:m7_gate": "m7_gate-hash"},
        output_hashes={},
    )
    smoke_path = tmp_path / "stages" / "smoke_m7.json"
    smoke_path.write_text(json.dumps(stale.to_dict()))
    predecessor_calls: list[str] = []
    training_calls: list[str] = []

    class StopAfterSmoke(RuntimeError):
        pass

    def unexpected(name: str):
        def action() -> StageResult:
            predecessor_calls.append(name)
            return StageResult({name: "unexpected"})

        return action

    def train(stage: str, variant_id: str) -> StageResult:
        if stage != "smoke_m7":
            raise StopAfterSmoke(stage)
        training_calls.append(f"{stage}:{variant_id}")
        return StageResult({"canonical": "smoke-m7-hash"})

    pipeline = object.__new__(FormalPipeline)
    pipeline.artifact_root = tmp_path
    pipeline.workflow = workflow
    pipeline._train = train
    for method_name in (
        "_audit",
        "_verify",
        "_preflight",
        "_batch_probe",
        "_m7_gate",
        "_baseline_b0",
        "_select",
        "_profile_all",
        "_final_audit",
        "_report",
    ):
        setattr(pipeline, method_name, unexpected(method_name))
    pipeline._evaluate = lambda split: unexpected(f"evaluate_{split}")()

    with pytest.raises(StopAfterSmoke, match="formal_m7"):
        pipeline.execute()

    assert predecessor_calls == []
    assert training_calls == ["smoke_m7:M7"]
    recovered = PipelineState.from_dict(json.loads(smoke_path.read_text()))
    assert recovered.status == "completed"
    assert recovered.attempt == 2


def test_val_all_reuses_selection_locked_m2_m3_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    from masf_yolo.contracts import sha256_file
    from masf_yolo.variants import EVALUATED_MODELS

    for variant in ("M2", "M3"):
        path = tmp_path / "evaluation" / "val" / variant.lower() / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"variant": variant, "locked": True}), encoding="utf-8")
    val_hashes = {
        variant: sha256_file(
            tmp_path / "evaluation" / "val" / variant.lower() / "metrics.json"
        )
        for variant in ("M2", "M3")
    }
    (tmp_path / "selection.json").write_text(
        json.dumps({"selected": "M3", "val_hashes": val_hashes}), encoding="utf-8"
    )
    calls: list[str] = []

    def fake_evaluate(checkpoint, data_yaml, coco_json, *, split, output_dir, device):
        variant = output_dir.name.upper()
        calls.append(variant)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"variant": variant}

    monkeypatch.setattr("masf_yolo.pipeline.run_variant_evaluation", fake_evaluate)
    pipeline = object.__new__(FormalPipeline)
    pipeline.artifact_root = tmp_path
    pipeline.data_yaml = tmp_path / "dataset" / "data.yaml"
    pipeline.source_weights = tmp_path / "source.pt"
    pipeline._best_checkpoint = lambda variant: tmp_path / f"{variant}.pt"

    pipeline._evaluate("val")

    assert set(calls) == set(EVALUATED_MODELS) - {"M2", "M3"}
    for variant in ("M2", "M3"):
        path = tmp_path / "evaluation" / "val" / variant.lower() / "metrics.json"
        assert sha256_file(path) == val_hashes[variant]
