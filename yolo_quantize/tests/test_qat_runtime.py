from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import yolo_quantize.qat_runtime as qat_runtime_module
from yolo_quantize.qat_metrics import QAT_GATE_METRICS
from yolo_quantize.qat_plan import QATWarmStartSpec
from yolo_quantize.qat_runtime import (
    Full35QATRuntime,
    _apply_qat_full_resume_warm_start,
    _save_qat_deployment_inference_weights,
    _tensor_objective,
    main,
)
from yolo_quantize.qat_weights import (
    FoldedQATWeightAdapter,
    TrainableWeightFakeQuantizer,
)
from yolo_quantize.weight_quantization import (
    ExactTernaryWeightSpec,
    Full35WeightRegionCatalog,
    UniformWeightSpec,
    WeightRegionAssignment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"


class _ContractFixture(torch.nn.Module):
    def __init__(self, *, fused: bool) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.cv2 = None if fused else ("training-box-head",)

    def contract(self) -> dict[str, object]:
        return {
            "model_kind": "qat-contract-fixture",
            "feature_channels": [len(value) for value in self.cv2],
        }


def test_qat_deployment_checkpoint_uses_prefuse_contract(tmp_path: Path) -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)
    resume_module = runtime._imports()["resume"]
    source = _ContractFixture(fused=False)
    deployment = _ContractFixture(fused=True)
    destination = tmp_path / "deployment.pt"

    _save_qat_deployment_inference_weights(
        resume_module,
        destination,
        deployment_model=deployment,
        contract_source=source,
        metadata={"epoch": 0},
    )

    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert payload["contract"] == source.contract()
    assert payload["state_dict"].keys() == deployment.state_dict().keys()
    assert payload["metadata"]["deployment_graph_schema"] == (
        "full35-bn-folded-a8-weight-materialized-v1"
    )


def test_qat_runtime_reconstructs_deployment_graph_before_loading_locked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)
    contract_source = _ContractFixture(fused=False)
    deployment = _ContractFixture(fused=True)
    with torch.no_grad():
        deployment.projection.weight.fill_(3.0)
        deployment.projection.bias.fill_(4.0)
    expected_state = {
        name: tensor.detach().clone()
        for name, tensor in deployment.state_dict().items()
    }
    checkpoint = tmp_path / "locked-parent.pt"
    full_resume_sha256 = "a" * 64
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_kind": "inference_only",
            "contract": contract_source.contract(),
            "source": "live",
            "state_dict": expected_state,
            "metadata": {
                "epoch": 5,
                "full_resume_sha256": full_resume_sha256,
                "qat_plan_sha256": runtime.plan.config_sha256,
                "qat_arm": "qat",
                "weight_blend_ratio": 1.0,
                "loader": "yolo_quantize.qat_runtime",
            },
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rebound_activation = SimpleNamespace(
        model=deployment, mode="fake_quant", quantizer_count=124
    )
    handles = SimpleNamespace(
        model=contract_source,
        weights=object(),
        activation=SimpleNamespace(
            rebind=lambda model: rebound_activation,
            quantizer_count=124,
        ),
        source=object(),
    )
    catalog = SimpleNamespace(
        summary=lambda: runtime.plan.expected_deployment_catalog,
        training_only_sites=(),
    )
    monkeypatch.setattr(runtime, "_build_graph", lambda: handles)
    monkeypatch.setattr(
        qat_runtime_module,
        "materialize_qat_deployment_graph",
        lambda *args, **kwargs: deployment,
        raising=False,
    )
    monkeypatch.setattr(
        qat_runtime_module,
        "Full35WeightRegionCatalog",
        SimpleNamespace(inspect=lambda model: catalog),
        raising=False,
    )

    parent = runtime.load_deployment_parent(
        checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        full_resume_sha256=full_resume_sha256,
        epoch=5,
    )

    assert parent.model is deployment
    assert parent.activation is rebound_activation
    assert parent.catalog is catalog
    assert parent.checkpoint_sha256 == checkpoint_sha256
    assert torch.equal(
        parent.model.projection.weight, expected_state["projection.weight"]
    )


def test_tensor_objective_covers_nested_differentiable_outputs() -> None:
    value = torch.tensor([1.0, 2.0], requires_grad=True)

    objective, tensors = _tensor_objective(
        {"detect": (value, {"pose": value * 2.0}), "metadata": "ignored"}
    )
    objective.backward()

    assert tensors == 2
    assert torch.isfinite(objective)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    with pytest.raises(ValueError, match="differentiable tensor"):
        _tensor_objective({"metadata": "only"})


def test_qat_runtime_cpu_preflight_preserves_search_and_dual_metric_contracts() -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)

    report = runtime.preflight(verify_graph=False, verify_sample_files=False)

    assert report.ready, report.blockers
    assert report.resolved["formal_validation"] is False
    assert report.resolved["pose_data"].endswith("pose-search.yaml")
    assert report.resolved["gates"] == {
        "map50_max_drop": 0.015,
        "map50_95_max_drop": 0.04,
    }
    assert report.resolved["required_metric_count"] == 16


def test_qat_gpu_smoke_requires_explicit_execution_acknowledgement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--plan", str(PLAN), "--gpu-smoke-only"])

    assert "GPU smoke requires --execute-reviewed-plan" in capsys.readouterr().err


def test_qat_runtime_derives_matched_arms_and_full35_hyperparameters() -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)

    assert runtime.weight_blend_ratio("sham") == 0.0
    assert runtime.weight_blend_ratio("qat") == 1.0
    assert runtime.run_name("sham") == ("v19-poly-shift-all-w8-qat-pilot-v1-sham-seed1")
    config = runtime.training_config("qat")
    assert config.stages == ()
    assert config.enable_j3 is True
    assert config.detect_batch_size == 128
    assert config.pose_batch_size == 16
    assert config.optimizer == "AdamW"
    assert config.gradient_clip_norm == 10.0
    assert config.validation_backends == ("bittrue",)
    assert config.selection_backend == "bittrue"
    assert config.maximum_map_drop == 0.015
    resolved = config.as_dict()
    assert resolved["pose_data"] == str(runtime.plan.data.pose_search)
    assert resolved["qat_experiment"]["data"] == {
        "coco": str(runtime.plan.data.coco),
        "bbat5_registry": str(runtime.plan.data.registry),
        "bbat5_pose_search": str(runtime.plan.data.pose_search),
        "bbat5_pose_runtime_view_role": "canonical-search-runtime-view",
        "assignment_changed": False,
    }
    assert all(
        "paths" in assignment
        for assignment in resolved["qat_experiment"]["weight_assignments"]
    )


def test_qat_arm_requires_completed_hash_pinned_matched_sham(tmp_path: Path) -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)
    runtime.plan = replace(runtime.plan, run_root=tmp_path)

    with pytest.raises(RuntimeError, match="matched sham completion evidence"):
        runtime._assert_sham_ready()

    run_dir = tmp_path / runtime.run_name("sham")
    checkpoint = run_dir / "checkpoints" / "best_joint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"matched-sham")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "plan_sha256": runtime.plan.config_sha256,
        "candidate_id": runtime.plan.candidate_id,
        "arm": "sham",
        "run_name": runtime.run_name("sham"),
        "completed_stages": ["j3"],
        "epochs_completed": runtime.plan.training.epochs,
        "checkpoint_paths": {"best_joint": str(checkpoint)},
        "checkpoint_sha256": {"best_joint": digest},
        "formal_validation": False,
    }
    (run_dir / "qat-experiment.json").write_text(json.dumps(manifest))

    evidence = runtime._assert_sham_ready()

    assert evidence["checkpoint"] == str(checkpoint)
    assert evidence["checkpoint_sha256"] == digest


def test_qat_runtime_reports_external_control_without_local_sham(
    tmp_path: Path,
) -> None:
    source = (
        PROJECT_ROOT
        / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1/short-qat-mixed-final-v1/generated/qat-plan.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    completion = (
        PROJECT_ROOT
        / "artifacts/runs/qat/v35-qsilu-mixed-final-short-v1/v35-qsilu-mixed-lssd4-w6-w4-short-qat-v1-sham-seed1/qat-experiment.json"
    )
    metrics = (
        PROJECT_ROOT
        / "artifacts/runs/qat/v35-qsilu-mixed-final-short-v1/v35-qsilu-mixed-lssd4-w6-w4-short-qat-v1-sham-seed1/validation/epoch-0003/bittrue/metrics.json"
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    payload["execution_authorization"]["arms"] = ["qat"]
    payload["execution_authorization"]["external_control"] = {
        "mode": "v35_external_sham_reference",
        "plan": {"path": str(source), "sha256": digest(source)},
        "completion": {"path": str(completion), "sha256": digest(completion)},
        "metrics": {"path": str(metrics), "sha256": digest(metrics)},
    }
    plan_path = tmp_path / "external-control-plan.yaml"
    plan_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    runtime = Full35QATRuntime.from_yaml(plan_path)

    evidence = runtime._external_control_evidence()

    assert evidence["evidence_kind"] == "external_control"
    assert evidence["unpaired_continuation"] is True
    assert evidence["mode"] == "v35_external_sham_reference"


def test_completed_sham_can_reconcile_a_drift_feasible_checkpoint(
    tmp_path: Path,
) -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)
    accepted_path = tmp_path / "accepted.json"
    matched_path = tmp_path / "matched.json"
    accepted = {name: 0.8 for name in QAT_GATE_METRICS}
    matched = {name: 0.78 for name in QAT_GATE_METRICS}
    accepted_path.write_text(json.dumps({"metrics": accepted}))
    matched_path.write_text(json.dumps({"metrics": matched}))
    runtime.plan = replace(
        runtime.plan,
        run_root=tmp_path,
        accepted_metrics=accepted_path,
        matched_metrics=matched_path,
        training=replace(runtime.plan.training, epochs=1),
    )
    run_dir = tmp_path / runtime.run_name("sham")
    metrics_path = run_dir / "validation/epoch-0000/bittrue/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    candidate = {name: 0.781 for name in QAT_GATE_METRICS}
    metrics_path.write_text(json.dumps({"epoch": 0, "metrics": candidate}))
    checkpoint = run_dir / "checkpoints/best_pose.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"progress": {"next_epoch": 1}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "plan_sha256": runtime.plan.config_sha256,
        "candidate_id": runtime.plan.candidate_id,
        "arm": "sham",
        "run_name": runtime.run_name("sham"),
        "completed_stages": ["j3"],
        "epochs_completed": runtime.plan.training.epochs,
        "epochs_planned": runtime.plan.training.epochs,
        "effective_patience": runtime.plan.training.patience,
        "checkpoint_paths": {"best_pose": str(checkpoint)},
        "checkpoint_sha256": {"best_pose": digest},
        "formal_validation": False,
    }
    (run_dir / "qat-experiment.json").write_text(json.dumps(manifest))

    report = runtime.reconcile_matched_sham()
    evidence = runtime._assert_sham_ready()

    assert report["selected_epoch"] == 0
    assert report["deployment_passed"] is False
    assert report["matched_sham_passed"] is True
    assert report["checkpoint_role"] == "best_pose"
    assert evidence["checkpoint"] == str(checkpoint)
    assert evidence["evidence_kind"] == "reconciled"


def test_qat_runtime_cpu_graph_preflight_builds_the_reviewed_full35_graph() -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)

    report = runtime.preflight(verify_graph=True, verify_sample_files=False)

    assert report.ready, report.blockers
    assert report.resolved["graph_verified"] is True
    assert report.resolved["graph"]["weight_quantizers"] == 148
    assert report.resolved["graph"]["activation_quantizers"] == 124
    assert report.resolved["graph"]["quantizer_optimizer_parameters"] == 396
    assert report.resolved["graph"]["quantizer_optimizer_groups"] == 6
    assert report.resolved["graph"]["final_batch_norm_modules"] == 0
    assert report.resolved["graph"]["binary_qk_protected_modules"] == 4


def test_qat_training_patch_is_scoped_and_restores_upstream_globals() -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)
    modules = runtime._imports()
    implementation = modules["formal_impl"]
    formal = modules["formal_training"]
    joint_data = modules["joint_data"]
    originals = {
        "stages": implementation.JOINT_STAGES,
        "factory": implementation.FusionModelFactory,
        "validator": implementation.JointValidator,
        "gate": implementation.AccuracyGate,
        "selectors": implementation.CheckpointSelectors,
        "prepare_view": formal.prepare_bbt5_view,
        "pose_validator": joint_data.validate_canonical_pose_source,
    }

    with runtime._patched_training(
        modules,
        arm="qat",
        run_name="test-patch-scope",
        device=torch.device("cpu"),
    ) as session_class:
        assert implementation.JOINT_STAGES["j3"].epochs == 15
        assert implementation.FusionModelFactory is not originals["factory"]
        assert implementation.JointValidator is not originals["validator"]
        assert implementation.AccuracyGate is not originals["gate"]
        assert implementation.CheckpointSelectors is not originals["selectors"]
        assert formal.prepare_bbt5_view is not originals["prepare_view"]
        assert (
            joint_data.validate_canonical_pose_source is not originals["pose_validator"]
        )
        assert issubclass(session_class, formal.FormalJointTrainingSession)

    assert implementation.JOINT_STAGES is originals["stages"]
    assert implementation.FusionModelFactory is originals["factory"]
    assert implementation.JointValidator is originals["validator"]
    assert implementation.AccuracyGate is originals["gate"]
    assert implementation.CheckpointSelectors is originals["selectors"]
    assert formal.prepare_bbt5_view is originals["prepare_view"]
    assert joint_data.validate_canonical_pose_source is originals["pose_validator"]


class _WarmStartGraph(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = torch.nn.Module()
        self.graph.model = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        self.graph.model[0].conv = torch.nn.Conv2d(2, 3, 1, bias=True)
        self.graph.model[1].conv = torch.nn.Conv2d(3, 4, 1, bias=True)


def _warm_start_policy(model: torch.nn.Module, *, exact_first: bool):
    assignments = [
        WeightRegionAssignment(
            region="backbone_early",
            spec=UniformWeightSpec(bits=8, scale_method="mse_grid_v1"),
        )
    ]
    if exact_first:
        assignments.append(
            WeightRegionAssignment(
                region="backbone_early",
                paths=("graph.model.0.conv",),
                spec=ExactTernaryWeightSpec(),
            )
        )
    return FoldedQATWeightAdapter().apply(
        model,
        catalog=Full35WeightRegionCatalog.inspect(model),
        assignments=tuple(assignments),
    )


def test_explicit_reset_preserves_inherited_override_scale(tmp_path):
    source = _WarmStartGraph()
    _warm_start_policy(source, exact_first=False)
    source.graph.model[1].conv.weight_quantizer.scale_parameter.data.fill_(0.37)
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        {
            "checkpoint_kind": "full_resume",
            "progress": {"next_epoch": 2},
            "ema_state": source.state_dict(),
        },
        checkpoint,
    )
    target = _WarmStartGraph()
    assignments = (
        WeightRegionAssignment("backbone_early", UniformWeightSpec(8, "mse_grid_v1")),
        WeightRegionAssignment(
            "backbone_early", ExactTernaryWeightSpec(), paths=("graph.model.0.conv",)
        ),
        WeightRegionAssignment(
            "backbone_early",
            UniformWeightSpec(8, "mse_grid_v1"),
            paths=("graph.model.1.conv",),
        ),
    )
    policy = FoldedQATWeightAdapter().apply(
        target,
        catalog=Full35WeightRegionCatalog.inspect(target),
        assignments=assignments,
    )
    warm = QATWarmStartSpec(
        parent_id="fixture",
        selected_epoch=1,
        parent_manifest=tmp_path / "manifest",
        parent_manifest_sha256="a" * 64,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        state_key="ema_state",
        reset_path_override_quantizers=True,
        reset_paths=("graph.model.0.conv",),
    )
    report = _apply_qat_full_resume_warm_start(target, policy, warm)
    assert report["reset_paths"] == ["graph.model.0.conv"]
    assert torch.equal(
        target.graph.model[1].conv.weight_quantizer.scale_parameter,
        source.graph.model[1].conv.weight_quantizer.scale_parameter,
    )


def test_full_resume_warm_start_loads_shadow_and_resets_only_override_qparam(
    tmp_path: Path,
) -> None:
    source_model = _WarmStartGraph()
    source_policy = _warm_start_policy(source_model, exact_first=False)
    with torch.no_grad():
        source_model.graph.model[0].conv.weight.copy_(
            torch.linspace(-1.5, 0.75, 6).reshape(3, 2, 1, 1)
        )
        source_model.graph.model[1].conv.weight.fill_(0.375)
        source_model.graph.model[0].conv.weight_quantizer.scale_parameter.fill_(7.0)
        source_model.graph.model[1].conv.weight_quantizer.scale_parameter.fill_(0.25)
    checkpoint = tmp_path / "best_joint.pt"
    torch.save(
        {
            "schema_version": 2,
            "checkpoint_kind": "full_resume",
            "ema_state": source_model.state_dict(),
            "optimizer_state": {"must_not_be_loaded": True},
            "progress": {"next_epoch": 6},
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    target_model = _WarmStartGraph()
    target_policy = _warm_start_policy(target_model, exact_first=True)
    warm = QATWarmStartSpec(
        parent_id="v19-fixture",
        selected_epoch=5,
        parent_manifest=tmp_path / "parent.yaml",
        parent_manifest_sha256="a" * 64,
        checkpoint=checkpoint,
        checkpoint_sha256=digest,
        state_key="ema_state",
        reset_path_override_quantizers=True,
        load_optimizer_state=False,
    )

    report = _apply_qat_full_resume_warm_start(
        target_model,
        target_policy,
        warm,
    )

    target_first = target_model.graph.model[0].conv
    target_second = target_model.graph.model[1].conv
    expected_exact = TrainableWeightFakeQuantizer.from_weight(
        source_model.graph.model[0].conv.weight,
        ExactTernaryWeightSpec(),
    )
    assert torch.equal(target_first.weight, source_model.graph.model[0].conv.weight)
    assert torch.equal(target_second.weight, source_model.graph.model[1].conv.weight)
    assert torch.allclose(
        target_first.weight_quantizer.scale,
        expected_exact.scale,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.equal(
        target_second.weight_quantizer.scale_parameter,
        source_model.graph.model[1].conv.weight_quantizer.scale_parameter,
    )
    assert target_first.weight_quantizer.blend_ratio == 0.0
    assert report["reset_override_quantizers"] == 1
    assert report["loaded_tensors"] > 0
    assert report["optimizer_state_loaded"] is False
    assert source_policy.quantized_modules == target_policy.quantized_modules == 2


def test_qat_arm_accepts_legitimate_patience5_matched_sham_early_stop(
    tmp_path: Path,
) -> None:
    runtime = Full35QATRuntime.from_yaml(PLAN)
    runtime.plan = replace(
        runtime.plan,
        run_root=tmp_path,
        training=replace(runtime.plan.training, patience=5),
    )
    run_name = runtime.run_name("sham")
    run_dir = tmp_path / run_name
    checkpoint = run_dir / "checkpoints" / "best_joint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"matched-sham-early-stop")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "plan_sha256": runtime.plan.config_sha256,
        "candidate_id": runtime.plan.candidate_id,
        "arm": "sham",
        "run_name": run_name,
        "completed_stages": ["j3"],
        "epochs_completed": 6,
        "epochs_planned": runtime.plan.training.epochs,
        "effective_patience": 5,
        "checkpoint_paths": {"best_joint": str(checkpoint)},
        "checkpoint_sha256": {"best_joint": digest},
        "early_stop": {
            "kind": "early_stop",
            "step": 5,
            "values": {
                "patience": 5.0,
                "stale_epochs": 5.0,
                "should_stop": 1.0,
            },
        },
        "formal_validation": False,
    }
    (run_dir / "qat-experiment.json").write_text(json.dumps(manifest))

    evidence = runtime._assert_sham_ready()

    assert evidence["epochs_completed"] == 6
