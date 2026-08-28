"""Float20 後 C2/C3 正式資料訓練與 fail-closed 接續判定。"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import SPEC_PATH, SPEC_VERSION, file_sha256
from .screen_training import (
    PROJECT_ROOT,
    ScreenRunConfig,
    _apply_training_mode,
    _engine,
    _is_oom,
    _make_runtime,
    _save_snapshot,
    _write_json,
    probe_screen_memory,
)
from .screen_validation import ScreenValidator, ThresholdSet

_ARCHITECTURE_YAMLS = {
    "C2": PROJECT_ROOT / "configs/candidates/c2-n1.yaml",
    "C3": PROJECT_ROOT / "configs/candidates/c3-mixed.yaml",
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必須是 mapping")
    return dict(value)


def _path(value: Any, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class FullRunConfig:
    """在既有 Float20 配方上只覆寫正式資料、預算與輸出位置。"""

    path: Path
    payload: dict[str, Any]
    base: ScreenRunConfig
    candidates: tuple[str, ...]
    float_results: Path
    run_root: Path
    detect_data: Path
    pose_data: Path
    diagnostic_detect_data: Path
    screen_root: Path
    epochs: int
    patience: int
    stage_name: str = "float-full"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @classmethod
    def load(cls, path: str | Path) -> FullRunConfig:
        source = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("full run YAML 根節點必須是 mapping")
        required = {
            "schema_version",
            "spec_version",
            "spec_sha256",
            "kind",
            "config_id",
            "title_zh",
            "status",
            "authorization",
            "source_screen_run",
            "float_results",
            "candidates",
            "datasets",
            "training",
            "quantization",
            "queue",
            "lineage",
        }
        if set(payload) != required:
            raise ValueError(
                f"full run YAML 欄位漂移：missing={sorted(required - set(payload))} "
                f"unknown={sorted(set(payload) - required)}"
            )
        if (
            payload["schema_version"] != 1
            or payload["kind"] != "float_full_quant_continuation"
            or payload["spec_version"] != SPEC_VERSION
            or payload["spec_sha256"] != file_sha256(SPEC_PATH)
        ):
            raise ValueError("full run YAML schema/spec metadata 漂移")
        authorization = _mapping(payload["authorization"], "authorization")
        if (
            authorization.get("gpu") is not True
            or authorization.get("pose") is not True
            or authorization.get("full_training") is not True
            or authorization.get("ptq") is not True
            or authorization.get("qat_lite") is not True
        ):
            raise PermissionError("full/PTQ/QAT-lite 必須全部有明確使用者授權")
        base_path = _path(payload["source_screen_run"], base=PROJECT_ROOT)
        base = ScreenRunConfig.load(base_path)
        candidates = tuple(str(value).upper() for value in payload["candidates"])
        if candidates != ("C2", "C3"):
            raise ValueError("自動接續候選必須固定為 C2、C3")
        datasets = _mapping(payload["datasets"], "datasets")
        training = _mapping(payload["training"], "training")
        if set(training) != {"run_root", "epochs", "patience", "runtime_cache_root"}:
            raise ValueError("training 欄位漂移")
        result = cls(
            path=source,
            payload=payload,
            base=base,
            candidates=candidates,
            float_results=_path(payload["float_results"], base=PROJECT_ROOT),
            run_root=_path(training["run_root"], base=PROJECT_ROOT),
            detect_data=_path(datasets["detect"], base=PROJECT_ROOT),
            pose_data=_path(datasets["pose"], base=PROJECT_ROOT),
            diagnostic_detect_data=_path(datasets["diagnostic_detect"], base=PROJECT_ROOT),
            screen_root=_path(training["runtime_cache_root"], base=PROJECT_ROOT),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.epochs != 100 or self.patience != 20:
            raise ValueError("正式 C2/C3 必須最多100 epochs、patience20")
        if not self.base.pose_enabled:
            raise PermissionError("正式比較必須 Pose=true")
        for path in (
            self.detect_data,
            self.pose_data,
            self.diagnostic_detect_data,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        quant = _mapping(self.payload["quantization"], "quantization")
        expected_quant = {
            "simulation_only",
            "calibration_batches_per_task",
            "qat_lite_steps",
            "observer_update_steps",
            "validation_interval_steps",
            "max_accuracy_drop",
            "accuracy_fields",
            "require_cost_reduction",
            "result_root",
        }
        if set(quant) != expected_quant:
            raise ValueError("quantization 欄位漂移")
        if (
            quant["simulation_only"] is not True
            or int(quant["calibration_batches_per_task"]) < 1
            or int(quant["qat_lite_steps"]) != 200
            or int(quant["observer_update_steps"]) != 50
            or int(quant["validation_interval_steps"]) != 50
        ):
            raise ValueError("PTQ/QAT-lite 固定契約漂移")
        if float(quant["max_accuracy_drop"]) != 0.008:
            raise ValueError("接近門檻固定為 0.008")
        expected_fields = {
            "coco_box_map50_95",
            "bbat5_pose_box_map50_95",
            "bbat5_keypoint_map50_95",
            "macro_f1",
        }
        if set(quant["accuracy_fields"]) != expected_fields:
            raise ValueError("accuracy_fields 漂移")

    def require_execution_enabled(self) -> None:
        """Reject execution because this downstream revision is archived."""

        status = self.payload["status"]
        raise PermissionError(
            "C2/C3完整訓練、PTQ與QAT-lite未獲執行資格；"
            f"目前狀態={status}。本階段已封存且不採用，不得排入queue"
        )

    def resolved_payload(self, *, candidate: str, microbatch: int) -> dict[str, Any]:
        return {
            "config": self.payload,
            "config_path": str(self.path),
            "config_sha256": file_sha256(self.path),
            "candidate": candidate,
            "effective_detect_microbatch": microbatch,
            "pose_enabled": True,
            "formal_split_used": True,
            "source_screen_config_sha256": file_sha256(self.base.path),
        }


def _verify_export_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    metrics_path = root / "metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Float20 正式匯出尚未完成")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Float20 manifest 未完成")
    files = _mapping(manifest.get("files"), "Float20 manifest.files")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Float20 匯出檔案雜湊漂移：{relative}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("status") != "completed_screening"
        or metrics.get("formal_split_used") is not False
        or metrics.get("pose_executed") is not True
    ):
        raise ValueError("Float20 metrics 契約漂移")
    entries = metrics.get("candidates")
    if not isinstance(entries, list):
        raise TypeError("Float20 candidates 必須是 list")
    ids = [item.get("metrics", {}).get("candidate_id") for item in entries]
    if ids != ["C0", "C1", "C2", "C3"]:
        raise ValueError("Float20 candidate 順序或內容漂移")
    return metrics


def eligible_full_candidates(config: FullRunConfig) -> dict[str, Any]:
    """依使用者核准的 0.008 精度門檻與正成本收益判定 C2/C3。"""

    payload = _verify_export_manifest(config.float_results)
    indexed = {
        item["metrics"]["candidate_id"]: item["metrics"]
        for item in payload["candidates"]
    }
    c0 = indexed["C0"]
    policy = _mapping(config.payload["quantization"], "quantization")
    max_drop = float(policy["max_accuracy_drop"])
    fields = tuple(str(value) for value in policy["accuracy_fields"])
    decisions: list[dict[str, Any]] = []
    eligible: list[str] = []
    for candidate in config.candidates:
        metrics = indexed[candidate]
        drops = {name: float(c0[name]) - float(metrics[name]) for name in fields}
        cost_reductions = {
            name: float(c0[name]) - float(metrics[name])
            for name in ("params", "gflops", "latency_ms")
        }
        accuracy_pass = all(value <= max_drop + 1e-12 for value in drops.values())
        cost_pass = any(value > 0 for value in cost_reductions.values())
        passed = accuracy_pass and (
            cost_pass if policy["require_cost_reduction"] is True else True
        )
        if passed:
            eligible.append(candidate)
        decisions.append(
            {
                "candidate": candidate,
                "eligible": passed,
                "accuracy_drops_vs_c0": drops,
                "max_accuracy_drop": max_drop,
                "cost_reductions_vs_c0": cost_reductions,
                "accuracy_pass": accuracy_pass,
                "cost_pass": cost_pass,
            }
        )
    report = {
        "schema_version": 1,
        "status": "completed",
        "source": str(config.float_results / "metrics.json"),
        "source_sha256": file_sha256(config.float_results / "metrics.json"),
        "eligible_candidates": eligible,
        "decisions": decisions,
        "rule_zh": "四項主要精度相對C0下降均不得超過0.008，且至少一項成本必須下降。",
    }
    _write_json(config.run_root / "eligibility.json", report)
    return report


def _thresholds(config: FullRunConfig) -> ThresholdSet:
    path = config.base.run_root / "shared-controls/c0-f1-thresholds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ThresholdSet.from_mapping(_mapping(payload["thresholds"], "thresholds"))


def _lineage(
    config: FullRunConfig,
    runtime: Any,
    *,
    candidate: str,
    microbatch: int,
) -> dict[str, Any]:
    architecture = _ARCHITECTURE_YAMLS[candidate]
    return {
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "architecture_yaml": str(architecture),
        "architecture_yaml_sha256": file_sha256(architecture),
        "training_yaml": str(config.path),
        "training_yaml_sha256": file_sha256(config.path),
        "detect_dataset_yaml": str(config.detect_data),
        "detect_dataset_yaml_sha256": file_sha256(config.detect_data),
        "pose_dataset_yaml": str(config.pose_data),
        "pose_dataset_yaml_sha256": file_sha256(config.pose_data),
        "handoff_manifest": str(config.handoff_manifest),
        "handoff_manifest_sha256": file_sha256(config.handoff_manifest),
        "parent_checkpoint": str(runtime.release.checkpoint),
        "parent_checkpoint_sha256": file_sha256(runtime.release.checkpoint),
        "candidate": candidate,
        "seed": config.seed,
        "effective_detect_microbatch": microbatch,
        "formal_split_used": True,
    }


def _formal_scores(validation: Any) -> dict[str, float]:
    detect = float(validation.metrics["detect"]["box"]["ap"]["map50_95"])
    pose_research = float(validation.metrics["pose"]["keypoints"]["ap"]["map50_95"])
    pose_official = float(validation.metrics["pose"]["official_combined_fitness"])
    return {
        "detect": detect,
        "pose_research": pose_research,
        "pose_official": pose_official,
        "joint_formal": (detect + 0.25 * pose_research) / 1.25,
    }


def run_full_candidate(
    config: FullRunConfig,
    *,
    candidate: str,
    microbatch: int,
) -> dict[str, Any]:
    config.require_execution_enabled()
    candidate = candidate.upper()
    if candidate not in config.candidates:
        raise ValueError(f"候選不在正式 full 矩陣：{candidate}")
    run_dir = config.run_root / f"{candidate.lower()}-full-seed{config.seed}"
    completed = run_dir / "complete.json"
    if completed.is_file():
        payload = json.loads(completed.read_text(encoding="utf-8"))
        payload["already_complete"] = True
        return payload

    runtime = _make_runtime(config, candidate=candidate, microbatch=microbatch)
    resolved_config = config.resolved_payload(candidate=candidate, microbatch=microbatch)
    lineage = _lineage(config, runtime, candidate=candidate, microbatch=microbatch)
    provenance = {
        "lineage": lineage,
        "full35_factory": runtime.parent.factory_report,
        "parent_checkpoint": runtime.parent.checkpoint_report,
        "candidate_build": runtime.build_report.to_dict(),
        "optimizer": asdict(runtime.optimizer_report),
        "screening_only": False,
        "formal_split_used": True,
        "transition": "fresh_from_same_c0_handoff",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run-manifest.json", provenance)
    (run_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    validator = ScreenValidator(
        runtime.release,
        runtime.parent.source,
        runtime.resolved,
        detect_data=config.detect_data,
        pose_data=config.pose_data,
        output_root=run_dir / "validation",
        imgsz=config.imgsz,
        detect_batch=config.detect_val_batch,
        pose_batch=config.pose_val_batch,
        detect_workers=config.detect_workers,
        pose_workers=config.pose_workers,
        device=config.device,
        pose_enabled=True,
        runtime_cache_root=config.screen_root / "validation-cache",
        scope="formal_coco_val2017_and_bbat5_val",
        formal_split_used=True,
    )
    fixed_thresholds = _thresholds(config)
    best_state: dict[str, Any] = {
        "detect": {"score": -1.0, "epoch": -1},
        "pose_research": {"score": -1.0, "epoch": -1},
        "pose_official": {"score": -1.0, "epoch": -1},
        "joint_formal": {"score": -1.0, "epoch": -1},
        "epochs_without_joint_improvement": 0,
    }
    start_epoch = 0
    global_macro = 0
    last_checkpoint = run_dir / "checkpoints/last.pt"
    if last_checkpoint.is_file():
        restored = runtime.api.load_training_snapshot(
            last_checkpoint,
            model=runtime.model,
            ema=runtime.ema,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            scaler=runtime.scaler,
            criteria=runtime.router,
            restore_rng=True,
        )
        if restored.resolved_config != resolved_config:
            raise ValueError("full resume effective config 漂移")
        start_epoch = restored.progress.next_epoch
        global_macro = restored.progress.global_macro_step
        best_state = restored.best_state
    else:
        runtime.api.seed_everything(config.seed)

    engine = _engine(runtime, config)
    loader_state: dict[str, Any] = {
        "effective_detect_microbatch": microbatch,
        "detect_logical_batch": config.detect_logical_batch,
        "detect_physical_microbatches_per_macro": runtime.microbatches_per_macro,
        "pose_batch": config.pose_batch,
        "validation_detect_batch": config.detect_val_batch,
        "validation_pose_batch": config.pose_val_batch,
    }
    stopped_early = False
    epochs_completed = start_epoch
    with runtime.api.ExperimentLogger(run_dir / "logs", tensorboard="auto") as logger:
        runner = runtime.api.JointEpochRunner(
            engine=engine,
            detect_loader=runtime.detect_loader.loader,
            pose_loader=runtime.pose_loader.loader,
            scheduler=runtime.scheduler,
            logger=logger,
            apply_training_mode=lambda: _apply_training_mode(runtime),
            assert_hardware_contract=lambda: runtime.guard.assert_unchanged(runtime.model),
            detect_batches_per_macro=runtime.microbatches_per_macro,
            gradient_statistics_interval=100,
        )
        for epoch in range(start_epoch, config.epochs):
            detect_seed = runtime.api.reseed_loader_for_epoch(
                runtime.detect_loader.loader,
                seed=config.seed,
                epoch=epoch,
                offset=0,
            )
            pose_seed = runtime.api.reseed_loader_for_epoch(
                runtime.pose_loader.loader,
                seed=config.seed,
                epoch=epoch,
                offset=1,
            )
            training = runner.run_epoch(
                epoch=epoch,
                global_macro_step=global_macro,
                stage=config.stage_name,
            )
            global_macro = training.next_global_macro_step
            validation = validator.validate(
                runtime.ema.ema,
                epoch=epoch,
                fixed_thresholds=fixed_thresholds,
            )
            logger.log(
                "validation",
                step=epoch,
                values=validation.flat_metrics,
                context={
                    "candidate": candidate,
                    "backend": "float",
                    "formal_split_used": True,
                },
            )
            scores = _formal_scores(validation)
            selected: list[str] = []
            joint_improved = False
            for label, score in scores.items():
                if score > float(best_state[label]["score"]):
                    best_state[label] = {"score": score, "epoch": epoch}
                    selected.append(f"best-{label.replace('_', '-')}")
                    if label == "joint_formal":
                        joint_improved = True
            best_state["epochs_without_joint_improvement"] = (
                0
                if joint_improved
                else int(best_state["epochs_without_joint_improvement"]) + 1
            )
            loader_state.update(
                {
                    "next_epoch": epoch + 1,
                    "detect_seed": detect_seed,
                    "pose_seed": pose_seed,
                }
            )
            checkpoint_metrics = {
                "scores": scores,
                "thresholds": validation.thresholds.to_dict(),
            }
            _save_snapshot(
                runtime,
                config,
                run_dir,
                label="last",
                next_epoch=epoch + 1,
                global_macro_step=global_macro,
                resolved_config=resolved_config,
                provenance=provenance,
                loader_state=loader_state,
                best_state=best_state,
                metrics=checkpoint_metrics,
            )
            for label in selected:
                _save_snapshot(
                    runtime,
                    config,
                    run_dir,
                    label=label,
                    next_epoch=epoch + 1,
                    global_macro_step=global_macro,
                    resolved_config=resolved_config,
                    provenance=provenance,
                    loader_state=loader_state,
                    best_state=best_state,
                    metrics=checkpoint_metrics,
                )
            epochs_completed = epoch + 1
            if int(best_state["epochs_without_joint_improvement"]) >= config.patience:
                stopped_early = True
                break

    summary = {
        "schema_version": 1,
        "candidate": candidate,
        "status": "completed_formal_training",
        "screening_only": False,
        "formal_split_used": True,
        "epochs_completed": epochs_completed,
        "max_epochs": config.epochs,
        "patience": config.patience,
        "stopped_early": stopped_early,
        "global_macro_steps": global_macro,
        "best_state": best_state,
        "best_inference_checkpoint": str(
            run_dir / "inference/best-joint-formal.pt"
        ),
        "selection_status": "pending_joint_full_and_quant_comparison",
        "run_dir": str(run_dir),
        "lineage": lineage,
    }
    _write_json(completed, summary)
    return summary


def run_full_matrix(config_path: str | Path, *, execute: bool) -> dict[str, Any]:
    config = FullRunConfig.load(config_path)
    if execute:
        config.require_execution_enabled()
    eligibility = eligible_full_candidates(config)
    selected = tuple(str(value) for value in eligibility["eligible_candidates"])
    plan = {
        "config": str(config.path),
        "config_sha256": file_sha256(config.path),
        "requested_candidates": list(config.candidates),
        "eligible_candidates": list(selected),
        "pose_enabled": True,
        "epochs": config.epochs,
        "patience": config.patience,
        "execute": execute,
    }
    if not execute:
        plan["status"] = "dry_run_only"
        return plan
    if not selected:
        matrix = {
            **plan,
            "status": "completed_no_eligible_candidates",
            "results": [],
            "next_gate": "stop_without_wasting_gpu",
        }
        _write_json(config.run_root / "matrix-complete.json", matrix)
        return matrix

    batch_plan_path = config.run_root / "shared-controls/batch-plan.json"
    if batch_plan_path.is_file():
        batch_plan = json.loads(batch_plan_path.read_text(encoding="utf-8"))
        microbatch = int(batch_plan["selected_detect_microbatch"])
    else:
        probes: list[dict[str, Any]] = []
        try:
            report = probe_screen_memory(config, microbatch=config.detect_microbatch)
            probes.append(report)
            microbatch = config.detect_microbatch
        except Exception as error:
            if not _is_oom(error):
                raise
            probes.append(
                {
                    "passed": False,
                    "physical_detect_microbatch": config.detect_microbatch,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
            report = probe_screen_memory(
                config,
                microbatch=config.detect_oom_fallback_microbatch,
            )
            probes.append(report)
            microbatch = config.detect_oom_fallback_microbatch
        batch_plan = {
            "schema_version": 1,
            "fairness_scope": "-".join(selected),
            "logical_detect_batch": config.detect_logical_batch,
            "selected_detect_microbatch": microbatch,
            "detect_physical_microbatches_per_macro": (
                config.detect_logical_batches_per_macro
                * config.detect_logical_batch
                // microbatch
            ),
            "pose_batch": config.pose_batch,
            "validation_detect_batch": config.detect_val_batch,
            "validation_pose_batch": config.pose_val_batch,
            "probes": probes,
        }
        _write_json(batch_plan_path, batch_plan)

    results = [
        run_full_candidate(config, candidate=candidate, microbatch=microbatch)
        for candidate in selected
    ]
    matrix = {
        **plan,
        "status": "completed_formal_training_matrix",
        "selected_detect_microbatch": microbatch,
        "results": results,
        "next_gate": "automatic_ptq_then_qat_lite",
    }
    _write_json(config.run_root / "matrix-complete.json", matrix)
    return matrix
