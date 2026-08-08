"""Concrete Phase 1 actions executed by the hash-gated workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import torch

from masf_yolo.artifacts.checkpoints import save_canonical_checkpoint
from masf_yolo.artifacts.finalize import build_finalize_command
from masf_yolo.artifacts.io import PipelineLock, atomic_write_json
from masf_yolo.contracts import DatasetManifest, canonical_json, sha256_file, sha256_value
from masf_yolo.data.audit import audit_dataset
from masf_yolo.evaluation.profiling import HardwareProfile, profile_module
from masf_yolo.evaluation.runner import run_variant_evaluation
from masf_yolo.evaluation.selection import (
    CandidateMetrics,
    freeze_selection,
    require_selection_before_test,
    select_best_partial,
)
from masf_yolo.models.builder import build_model
from masf_yolo.models.transfer import transfer_b1_canonical
from masf_yolo.runtime import pipeline_identity, verify_environment
from masf_yolo.training.preflight import probe_common_batch, run_finite_loss_batch, run_optimizer_step
from masf_yolo.training.profiles import b1_a_profile, formal_profile, smoke_profile
from masf_yolo.training.completion import ensure_complete_training_output
from masf_yolo.training.resume import PermanentTrainingError, TransientTrainingError, execute_with_retries
from masf_yolo.training.worker import TrainingWorkerRequest, launch_worker_process
from masf_yolo.variants import CORE_VARIANTS, EVALUATED_MODELS, get_variant
from masf_yolo.workflow import PHASE1_STAGES, PipelineWorkflow, StageResult


PREFLIGHT_VARIANTS = CORE_VARIANTS
EVALUATION_MODELS = EVALUATED_MODELS
CANONICAL_TRAINING_STAGES = (
    "b1_a",
    "b1_b",
    "formal_m7",
    "formal_m0",
    "formal_m1",
    "formal_m2",
    "formal_m3",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: object) -> Any:
    return json.loads(canonical_json(value).decode("utf-8"))


def candidate_from_artifacts(
    variant_id: str,
    metrics: dict[str, Any],
    profile: HardwareProfile,
) -> CandidateMetrics:
    return CandidateMetrics(
        variant_id=variant_id,
        map50_95=metrics["map50_95"],
        ap_s=metrics["ap_s"],
        ball_recall=metrics["ball_recall"],
        ball_ap_s=metrics["ball_ap_s"],
        tiny_recall=metrics["ball_subsets"]["tiny"]["recall"],
        blur_recall=metrics["ball_subsets"]["blur_proxy"]["recall"],
        gflops=profile.gflops,
        params=float(profile.params),
        peak_activation=float(profile.peak_live_activation_bytes),
        traffic=float(profile.feature_traffic_bytes),
    )


def normalize_profile(variant_id: str, profile: HardwareProfile) -> HardwareProfile:
    """Avoid labelling B0's first P3 Detect input as a nonexistent P2 activation."""
    return replace(profile, p2_activation_bytes=None) if variant_id == "B0" else profile


def run_final_audit(artifact_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    for stage in CANONICAL_TRAINING_STAGES:
        record_path = artifact_root / "training" / stage / "run.json"
        if not record_path.is_file():
            errors.append(f"missing training record: {stage}")
            continue
        record = _read_json(record_path)
        canonical = Path(record.get("canonical", ""))
        if not canonical.is_file():
            errors.append(f"missing canonical checkpoint: {stage}")
        if record.get("strict_reload") is not True:
            errors.append(f"strict reload not proven: {stage}")
    selection_path = artifact_root / "selection.json"
    selected: str | None = None
    if not selection_path.is_file():
        errors.append("missing selection.json")
    else:
        selection = _read_json(selection_path)
        selected = selection.get("selected")
        if selected not in {"M2", "M3"}:
            errors.append("BEST_PARTIAL must be M2 or M3")
    reference_path = artifact_root / "references" / "b0.json"
    if not reference_path.is_file():
        errors.append("missing B0 reference manifest")
    else:
        reference = _read_json(reference_path)
        if reference.get("checkpoint_hash") != (
            "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d"
        ):
            errors.append("B0 reference checkpoint hash mismatch")
        if reference.get("data_exposed") is not True or reference.get("selection_eligible") is not False:
            errors.append("B0 reference must be data-exposed and selection-ineligible")
    for split in ("val", "test"):
        for variant in EVALUATION_MODELS:
            if not (artifact_root / "evaluation" / split / variant.lower() / "metrics.json").is_file():
                errors.append(f"missing {split} metrics: {variant}")
    for variant in EVALUATION_MODELS:
        if not (artifact_root / "profiles" / variant.lower() / "profile.json").is_file():
            errors.append(f"missing hardware profile: {variant}")
    result = {"ok": not errors, "errors": errors, "best_partial": selected}
    atomic_write_json(artifact_root / "final_audit.json", result)
    return result


class FormalPipeline:
    def __init__(self, config_path: Path) -> None:
        from masf_yolo.cli import load_config

        self.config_path = config_path.resolve()
        self.root = self.config_path.parent.parent
        self.config = load_config(self.config_path)
        self.values = self.config.values
        self.artifact_root = self.root / self.values["artifacts_root"]
        self.dataset = DatasetManifest.from_dict(
            _read_json(self.artifact_root / "dataset" / "manifest.json")
        )
        self.environment = verify_environment(self.config_path, require_cuda=True)
        self.pipeline_id = pipeline_identity(
            self.config.config_hash,
            self.dataset.dataset_hash,
            self.environment.manifest_hash,
        )
        self.source_weights = (self.root / self.values["model"]["source_weights"]).resolve()
        self.data_yaml = self.artifact_root / "dataset" / "data.yaml"
        self.workflow = PipelineWorkflow(
            self.artifact_root,
            pipeline_id=self.pipeline_id,
            common_input_hashes={
                "config": self.config.config_hash,
                "data": self.dataset.dataset_hash,
                "environment": self.environment.manifest_hash,
                "source_weights": self.environment.source_weights_hash,
            },
        )

    def execute(self) -> None:
        handlers: dict[str, Callable[[], StageResult]] = {
            "audit": self._audit,
            "verify": self._verify,
            "preflight": self._preflight,
            "batch_probe": self._batch_probe,
            "b1_a": lambda: self._train("b1_a", "B1"),
            "b1_b": lambda: self._train("b1_b", "B1"),
            "m7_gate": self._m7_gate,
            "smoke_m7": lambda: self._train("smoke_m7", "M7"),
            "formal_m7": lambda: self._train("formal_m7", "M7"),
            "smoke_m0": lambda: self._train("smoke_m0", "M0"),
            "smoke_m1": lambda: self._train("smoke_m1", "M1"),
            "smoke_m2": lambda: self._train("smoke_m2", "M2"),
            "smoke_m3": lambda: self._train("smoke_m3", "M3"),
            "formal_m0": lambda: self._train("formal_m0", "M0"),
            "formal_m1": lambda: self._train("formal_m1", "M1"),
            "formal_m2": lambda: self._train("formal_m2", "M2"),
            "formal_m3": lambda: self._train("formal_m3", "M3"),
            "baseline_b0": self._baseline_b0,
            "val_all": lambda: self._evaluate("val"),
            "selection": self._select,
            "test_all": lambda: self._evaluate("test"),
            "profile_all": self._profile_all,
            "final_audit": self._final_audit,
            "report": self._report,
        }
        with PipelineLock(self.artifact_root / "pipeline.lock"):
            for stage in PHASE1_STAGES:
                self.workflow.run_stage(stage.name, handlers[stage.name])

    def _audit(self) -> StageResult:
        manifest = audit_dataset(
            self.root / self.values["dataset"]["source"],
            self.artifact_root / "dataset",
            seed=self.values["dataset"]["seed"],
            minimum_ball_count=self.values["dataset"]["minimum_ball_count"],
        )
        if manifest.dataset_hash != self.dataset.dataset_hash:
            raise RuntimeError("dataset hash changed after pipeline start")
        path = self.artifact_root / "dataset" / "manifest.json"
        return StageResult({"manifest": sha256_file(path)})

    def _verify(self) -> StageResult:
        environment = verify_environment(self.config_path, require_cuda=True)
        if environment.manifest_hash != self.environment.manifest_hash:
            raise RuntimeError("environment hash changed after pipeline start")
        path = self.artifact_root / "environment.json"
        atomic_write_json(path, environment.to_dict())
        return StageResult({"environment": sha256_file(path)})

    @staticmethod
    def _synthetic_batch(batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "img": torch.rand(batch_size, 3, 640, 640),
            "batch_idx": torch.arange(batch_size),
            "cls": torch.zeros(batch_size, 1),
            "bboxes": torch.tensor([[0.5, 0.5, 0.02, 0.02]]).repeat(batch_size, 1),
        }

    def _strict_reload(
        self, checkpoint: Path, variant_id: str, *, validate: bool = False
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            "-m",
            "masf_yolo.artifacts.strict_reload",
            "--checkpoint",
            str(checkpoint),
            "--variant",
            variant_id,
            "--data-hash",
            self.dataset.dataset_hash,
            "--config-hash",
            self.config.config_hash,
            "--environment-hash",
            self.environment.manifest_hash,
        ]
        if validate:
            command.extend(
                ["--data", str(self.data_yaml), "--device", "0", "--imgsz", "640", "--split", "val"]
            )
        result = subprocess.run(
            command,
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout.splitlines()[-1])

    def _preflight(self) -> StageResult:
        output = self.artifact_root / "preflight"
        output.mkdir(parents=True, exist_ok=True)
        reports: dict[str, Any] = {}
        device = torch.device("cuda:0")
        for variant_id in PREFLIGHT_VARIANTS:
            model = build_model(variant_id, source_weights=self.source_weights)
            loss = run_finite_loss_batch(model, self._synthetic_batch(1), device=device)
            checkpoint = output / f"{variant_id.lower()}.pt"
            manifest = save_canonical_checkpoint(
                model,
                checkpoint,
                get_variant(variant_id),
                data_hash=self.dataset.dataset_hash,
                config_hash=self.config.config_hash,
                environment_hash=self.environment.manifest_hash,
            )
            strict = self._strict_reload(checkpoint, variant_id)
            reports[variant_id] = {
                "loss": loss,
                "checkpoint_hash": manifest.checkpoint_hash,
                "strict_reload": strict,
                "transfer": getattr(model, "masf_transfer_report", None),
            }
            del model
            torch.cuda.empty_cache()
        path = output / "report.json"
        atomic_write_json(path, reports)
        return StageResult({"preflight": sha256_file(path)})

    def _batch_probe(self) -> StageResult:
        def probe(variant_id: str, batch_size: int) -> bool:
            model = None
            try:
                model = build_model(variant_id, source_weights=self.source_weights)
                run_optimizer_step(
                    model,
                    self._synthetic_batch(batch_size),
                    device=torch.device("cuda:0"),
                    amp=True,
                )
                return True
            except torch.cuda.OutOfMemoryError:
                return False
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()

        batch = probe_common_batch(probe, tuple(self.values["training"]["batch_candidates"]))
        path = self.artifact_root / "common_batch.json"
        atomic_write_json(path, {"batch": batch, "variants": list(PREFLIGHT_VARIANTS)})
        return StageResult({"common_batch": sha256_file(path)})

    def _common_batch(self) -> int:
        return int(_read_json(self.artifact_root / "common_batch.json")["batch"])

    def _m7_gate(self) -> StageResult:
        output = self.artifact_root / "m7_gate"
        output.mkdir(parents=True, exist_ok=True)
        b1 = build_model(
            "B1", checkpoint=self.artifact_root / "training" / "b1_b" / "canonical.pt"
        )
        model = build_model("M7")
        transfer = transfer_b1_canonical(model, b1.state_dict())
        del b1
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        loss = run_optimizer_step(
            model,
            self._synthetic_batch(self._common_batch()),
            device=device,
            amp=True,
        )
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        checkpoint = output / "m7.pt"
        manifest = save_canonical_checkpoint(
            model,
            checkpoint,
            get_variant("M7"),
            data_hash=self.dataset.dataset_hash,
            config_hash=self.config.config_hash,
            environment_hash=self.environment.manifest_hash,
        )
        strict = self._strict_reload(checkpoint, "M7")
        report = {
            "variant": "M7",
            "batch": self._common_batch(),
            "loss": loss,
            "peak_gpu_memory_bytes": peak_memory,
            "checkpoint_hash": manifest.checkpoint_hash,
            "variant_hash": get_variant("M7").config_hash,
            "strict_reload": strict,
            "transfer": transfer.to_dict(),
        }
        path = output / "report.json"
        atomic_write_json(path, report)
        del model
        torch.cuda.empty_cache()
        return StageResult({"m7_gate": sha256_file(path)})

    def _baseline_b0(self) -> StageResult:
        output = self.artifact_root / "references" / "b0.json"
        command = [
            sys.executable,
            "-m",
            "masf_yolo.evaluation.reference",
            "--definition",
            str(self.root / "configs" / "b0-reference.yaml"),
            "--output",
            str(output),
        ]
        result = subprocess.run(
            command,
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PermanentTrainingError(
                f"B0 reference inspection failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        if not output.is_file():
            raise PermanentTrainingError("B0 reference inspection omitted its manifest")
        return StageResult({"b0_reference": sha256_file(output)})

    def _training_record(self, stage: str) -> dict[str, Any]:
        return _read_json(self.artifact_root / "training" / stage / "run.json")

    def _initial_model(self, stage: str, variant_id: str) -> tuple[torch.nn.Module, dict[str, Any] | None]:
        if stage == "b1_a":
            model = build_model("B1", source_weights=self.source_weights)
            return model, getattr(model, "masf_transfer_report", None)
        if stage == "b1_b":
            from ultralytics import YOLO

            model = YOLO(self._training_record("b1_a")["best"], task="detect").model
            model.masf_variant = "B1"
            model.masf_variant_hash = get_variant("B1").config_hash
            return model, None
        b1 = build_model("B1", checkpoint=Path(self._training_record("b1_b")["canonical"]))
        model = build_model(variant_id)
        transfer = transfer_b1_canonical(model, b1.state_dict())
        return model, transfer.to_dict()

    def _profile_for_stage(self, stage: str, variant_id: str) -> dict[str, Any]:
        training_root = str((self.artifact_root / ("smoke_runs" if stage.startswith("smoke") else "runs")).resolve())
        if stage == "b1_a":
            profile = b1_a_profile(str(self.source_weights), training_root)
        elif stage == "b1_b":
            profile = formal_profile("B1", self._training_record("b1_a")["best"], training_root, epochs=90)
            profile["name"] = "b1-b"
        elif stage.startswith("smoke"):
            profile = smoke_profile(variant_id, str(self.artifact_root / "training" / "b1_b" / "canonical.pt"), training_root)
        else:
            profile = formal_profile(variant_id, str(self.artifact_root / "training" / "b1_b" / "canonical.pt"), training_root, epochs=100)
        profile.update(
            {
                "data": str(self.data_yaml.resolve()),
                "batch": self._common_batch(),
                "device": 0,
                "exist_ok": True,
            }
        )
        return profile

    def _train(self, stage: str, variant_id: str) -> StageResult:
        stage_root = self.artifact_root / "training" / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        profile = self._profile_for_stage(stage, variant_id)
        atomic_write_json(stage_root / "resolved_args.json", _jsonable(profile))
        run_dir = Path(profile["project"]) / profile["name"]
        worker_request = stage_root / "worker_request.json"
        worker_result = stage_root / "worker_result.json"

        def launch_worker(resume_path: Path | None) -> None:
            request = TrainingWorkerRequest(
                config_path=self.config_path,
                stage=stage,
                variant_id=variant_id,
                profile=_jsonable(profile),
                resume_path=resume_path,
            )
            launch_worker_process(
                request,
                request_path=worker_request,
                output_path=worker_result,
                python=Path(sys.executable).absolute(),
            )

        training_output = ensure_complete_training_output(
            run_dir,
            int(profile["epochs"]),
            launch_worker,
            max_attempts=int(self.values["pipeline"]["max_attempts"]),
        )
        if training_output.best is None or training_output.last is None:
            raise PermanentTrainingError("complete training output is missing best or last checkpoint")
        canonical = stage_root / "canonical.pt"
        finalize_report_path = stage_root / "finalize.json"
        command = build_finalize_command(
            python=Path(sys.executable).absolute(),
            source=training_output.best,
            checkpoint=canonical,
            variant_id=variant_id,
            data_hash=self.dataset.dataset_hash,
            config_hash=self.config.config_hash,
            environment_hash=self.environment.manifest_hash,
            output=finalize_report_path,
        )

        def finalize_operation(_attempt: int, _resume: bool) -> dict[str, Any]:
            finalize_report_path.unlink(missing_ok=True)
            result = subprocess.run(command, cwd=self.root, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                diagnostics = result.stderr.strip() or result.stdout.strip()
                if result.returncode in {-9, 137}:
                    raise TransientTrainingError(
                        f"canonical finalizer exited {result.returncode}: {diagnostics}"
                    )
                raise PermanentTrainingError(
                    f"canonical finalizer exited {result.returncode}: {diagnostics}"
                )
            if not finalize_report_path.is_file():
                raise PermanentTrainingError("canonical finalizer omitted its result manifest")
            return _read_json(finalize_report_path)

        finalize_report = execute_with_retries(
            finalize_operation,
            max_attempts=int(self.values["pipeline"]["max_attempts"]),
        )
        strict = self._strict_reload(canonical, variant_id, validate=True)
        transfer = _read_json(worker_result).get("transfer") if worker_result.is_file() else None
        record = {
            "stage": stage,
            "variant": variant_id,
            "best": str(training_output.best),
            "last": str(training_output.last),
            "canonical": str(canonical.resolve()),
            "canonical_hash": finalize_report["checkpoint_hash"],
            "strict_reload": strict.get("strict_load") is True,
            "strict_reload_report": strict,
            "profile_hash": sha256_value(profile),
            "transfer": transfer,
            "training_output": _jsonable(asdict(training_output)),
            "results_hash": training_output.results_hash,
            "best_hash": training_output.best_hash,
            "last_hash": training_output.last_hash,
            "finalize_report": finalize_report,
        }
        path = stage_root / "run.json"
        atomic_write_json(path, record)
        return StageResult(
            {"run": sha256_file(path), "canonical": finalize_report["checkpoint_hash"]}
        )

    def _best_checkpoint(self, variant_id: str) -> Path:
        if variant_id == "B0":
            from masf_yolo.evaluation.reference import load_b0_definition

            return load_b0_definition(self.root / "configs" / "b0-reference.yaml").checkpoint_path
        stage = "b1_b" if variant_id == "B1" else f"formal_{variant_id.lower()}"
        return Path(self._training_record(stage)["best"])

    def _evaluate(self, split: str) -> StageResult:
        if split == "test":
            require_selection_before_test(self.artifact_root / "selection.json")
        aggregate: dict[str, str] = {}
        for variant_id in EVALUATION_MODELS:
            output = self.artifact_root / "evaluation" / split / variant_id.lower()
            metrics = run_variant_evaluation(
                self._best_checkpoint(variant_id),
                self.data_yaml,
                self.artifact_root / "dataset" / f"{split}.coco.json",
                split=split,
                output_dir=output,
                device=0,
            )
            aggregate[variant_id] = sha256_value(metrics)
        path = self.artifact_root / "evaluation" / split / "aggregate.json"
        atomic_write_json(path, aggregate)
        return StageResult({split: sha256_file(path)})

    def _profile_variant(self, variant_id: str, output_root: Path) -> HardwareProfile:
        from ultralytics import YOLO

        model = YOLO(str(self._best_checkpoint(variant_id)), task="detect").model.float().cpu()
        profile = normalize_profile(
            variant_id,
            profile_module(model, torch.zeros(1, 3, 640, 640)),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_root / "profile.json", asdict(profile))
        return profile

    def _select(self) -> StageResult:
        candidates: dict[str, CandidateMetrics] = {}
        val_hashes: dict[str, str] = {}
        for variant_id in ("M2", "M3"):
            metrics_path = self.artifact_root / "evaluation" / "val" / variant_id.lower() / "metrics.json"
            metrics = _read_json(metrics_path)
            profile = self._profile_variant(
                variant_id, self.artifact_root / "selection_profiles" / variant_id.lower()
            )
            candidates[variant_id] = candidate_from_artifacts(variant_id, metrics, profile)
            val_hashes[variant_id] = sha256_file(metrics_path)
        result = select_best_partial(candidates["M2"], candidates["M3"])
        path = self.artifact_root / "selection.json"
        freeze_selection(path, result, val_hashes=val_hashes)
        return StageResult({"selection": sha256_file(path)})

    def _profile_all(self) -> StageResult:
        hashes: dict[str, str] = {}
        for variant_id in EVALUATION_MODELS:
            output = self.artifact_root / "profiles" / variant_id.lower()
            self._profile_variant(variant_id, output)
            hashes[variant_id] = sha256_file(output / "profile.json")
        path = self.artifact_root / "profiles" / "aggregate.json"
        atomic_write_json(path, hashes)
        return StageResult({"profiles": sha256_file(path)})

    def _final_audit(self) -> StageResult:
        result = run_final_audit(self.artifact_root)
        if not result["ok"]:
            raise RuntimeError(f"final audit failed: {result['errors']}")
        path = self.artifact_root / "final_audit.json"
        return StageResult({"final_audit": sha256_file(path)})

    def _report(self) -> StageResult:
        from masf_yolo.reporting import rebuild_report

        path = Path(rebuild_report(self.config_path))
        return StageResult({"report": sha256_file(path)})


def execute_formal_pipeline(config_path: Path) -> None:
    FormalPipeline(config_path).execute()
