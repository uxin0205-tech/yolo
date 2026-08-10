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
from masf_yolo.evaluation.profiling import HardwareProfile, profile_gpu_latency, profile_module
from masf_yolo.evaluation.runner import run_variant_evaluation
from masf_yolo.evaluation.selection import (
    CandidateMetrics,
    freeze_selection,
    require_selection_before_test,
    select_best_partial,
)
from masf_yolo.models.builder import build_model
from masf_yolo.models.transfer import transfer_b1_canonical
from masf_yolo.runtime import (
    pipeline_identity,
    verify_environment,
    wait_for_gpu_idle,
    wait_for_predecessor_units,
)
from masf_yolo.training.preflight import probe_common_batch, run_finite_loss_batch, run_optimizer_step
from masf_yolo.training.profiles import (
    b1_a_profile,
    formal_profile,
    frozen_stage_profile,
    smoke_profile,
)
from masf_yolo.training.completion import ensure_complete_training_output
from masf_yolo.training.resume import PermanentTrainingError, TransientTrainingError, execute_with_retries
from masf_yolo.training.worker import TrainingWorkerRequest, launch_worker_process
from masf_yolo.variants import (
    EVALUATED_MODELS,
    TRAINED_VARIANTS,
    get_variant,
    sp2p_variant_id,
)
from masf_yolo.workflow import PHASE1_STAGES, PipelineWorkflow, StageResult


PREFLIGHT_VARIANTS = TRAINED_VARIANTS
EVALUATION_MODELS = EVALUATED_MODELS
CANONICAL_TRAINING_STAGES = (
    "b1_a",
    "b1_b",
    "formal_m7",
    "formal_m0",
    "formal_m1",
    "formal_m2",
    "formal_m3",
    "formal_p3m",
    "sp2_a",
    "sp2_b",
    "sp2p_a",
    "sp2p_b",
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
        else:
            expected_architecture = sp2p_variant_id(selected)
            expected_selection_hash = sha256_file(selection_path)
            expected_parent_stages = ("sp2_b", f"formal_{selected.lower()}")
            expected_parent_hashes: dict[str, str] = {}
            for parent_stage in expected_parent_stages:
                parent_record = artifact_root / "training" / parent_stage / "run.json"
                if parent_record.is_file():
                    expected_parent_hashes[parent_stage] = _read_json(parent_record).get(
                        "canonical_hash", ""
                    )
            for stage in ("sp2p_a", "sp2p_b"):
                record_path = artifact_root / "training" / stage / "run.json"
                if not record_path.is_file():
                    continue
                record = _read_json(record_path)
                if (
                    record.get("display_variant") != "SP2P"
                    or record.get("architecture_variant") != expected_architecture
                    or record.get("selected_partial") != selected
                ):
                    errors.append("SP2P lineage does not match BEST_PARTIAL")
                    continue
                if record.get("selection_hash") != expected_selection_hash:
                    errors.append("SP2P selection hash mismatch")
                if record.get("parent_hashes") != expected_parent_hashes:
                    errors.append("SP2P parent hashes mismatch")
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
            metrics_path = artifact_root / "evaluation" / split / variant.lower() / "metrics.json"
            if not metrics_path.is_file():
                errors.append(f"missing {split} metrics: {variant}")
                continue
            try:
                metrics = _read_json(metrics_path)
            except (OSError, UnicodeError, json.JSONDecodeError):
                errors.append(f"invalid {split} metrics: {variant}")
                continue
            for section in ("per_class", "class_diagnostics"):
                class_values = metrics.get(section)
                for class_name in ("ball", "bat"):
                    if not isinstance(class_values, dict) or not isinstance(
                        class_values.get(class_name), dict
                    ):
                        errors.append(
                            f"missing {split} {section} class {class_name}: {variant}"
                        )
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
            "gpu_wait": self._gpu_wait,
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
            "smoke_p3m": lambda: self._train("smoke_p3m", "P3M"),
            "smoke_sp2": lambda: self._train("smoke_sp2", "SP2"),
            "formal_m0": lambda: self._train("formal_m0", "M0"),
            "formal_m1": lambda: self._train("formal_m1", "M1"),
            "formal_m2": lambda: self._train("formal_m2", "M2"),
            "formal_m3": lambda: self._train("formal_m3", "M3"),
            "formal_p3m": lambda: self._train("formal_p3m", "P3M"),
            "sp2_a": lambda: self._train("sp2_a", "SP2"),
            "sp2_b": lambda: self._train("sp2_b", "SP2"),
            "val_partial": lambda: self._evaluate_variants("val", ("M2", "M3")),
            "selection": self._select,
            "smoke_sp2p": lambda: self._train("smoke_sp2p", self._sp2p_variant()),
            "sp2p_a": lambda: self._train("sp2p_a", self._sp2p_variant()),
            "sp2p_b": lambda: self._train("sp2p_b", self._sp2p_variant()),
            "baseline_b0": self._baseline_b0,
            "val_all": lambda: self._evaluate("val"),
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

    def _gpu_wait(self) -> StageResult:
        path = self.artifact_root / "gpu_wait.json"
        predecessor = wait_for_predecessor_units(
            tuple(self.values["pipeline"]["wait_for_units"])
        )
        atomic_write_json(
            path,
            {
                "predecessor_services": predecessor,
                "gpu_idle": wait_for_gpu_idle(),
                "ready": True,
            },
        )
        return StageResult({"gpu_wait": sha256_file(path)})

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
        elif stage in {"sp2_a", "sp2p_a"}:
            parent = self.artifact_root / "training" / (
                "b1_b" if stage == "sp2_a" else "sp2_b"
            ) / "canonical.pt"
            profile = frozen_stage_profile(
                variant_id,
                str(parent),
                training_root,
                name=stage.replace("_", "-"),
            )
        elif stage in {"sp2_b", "sp2p_b"}:
            parent_stage = "sp2_a" if stage == "sp2_b" else "sp2p_a"
            profile = formal_profile(
                variant_id,
                self._training_record(parent_stage)["best"],
                training_root,
                epochs=90,
            )
            profile["name"] = stage.replace("_", "-")
        elif stage.startswith("smoke"):
            parent_stage = "sp2_b" if stage == "smoke_sp2p" else "b1_b"
            profile = smoke_profile(
                variant_id,
                str(self.artifact_root / "training" / parent_stage / "canonical.pt"),
                training_root,
            )
            if stage == "smoke_sp2p":
                profile["name"] = "sp2p-smoke"
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
            "data_exposed": True,
            "initializer": str(self.source_weights),
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
        if variant_id in {"SP2M2", "SP2M3"}:
            selection_path = self.artifact_root / "selection.json"
            selection = _read_json(selection_path)
            selected_partial = selection["selected"]
            partial_stage = f"formal_{selected_partial.lower()}"
            record.update(
                {
                    "display_variant": "SP2P",
                    "architecture_variant": variant_id,
                    "selected_partial": selected_partial,
                    "parent_hashes": {
                        "sp2_b": self._training_record("sp2_b")["canonical_hash"],
                        partial_stage: self._training_record(partial_stage)["canonical_hash"],
                    },
                    "selection_hash": sha256_file(selection_path),
                }
            )
        path = stage_root / "run.json"
        atomic_write_json(path, record)
        return StageResult(
            {"run": sha256_file(path), "canonical": finalize_report["checkpoint_hash"]}
        )

    def _best_checkpoint(self, variant_id: str) -> Path:
        if variant_id == "B0":
            from masf_yolo.evaluation.reference import load_b0_definition

            return load_b0_definition(self.root / "configs" / "b0-reference.yaml").checkpoint_path
        if variant_id == "B1":
            stage = "b1_b"
        elif variant_id == "SP2":
            stage = "sp2_b"
        elif variant_id == "SP2P":
            stage = "sp2p_b"
        else:
            stage = f"formal_{variant_id.lower()}"
        return Path(self._training_record(stage)["best"])

    def _sp2p_variant(self) -> str:
        selection = _read_json(self.artifact_root / "selection.json")
        return sp2p_variant_id(selection.get("selected"))

    def _evaluate(self, split: str) -> StageResult:
        reuse_locked = split == "val"
        return self._evaluate_variants(
            split,
            EVALUATION_MODELS,
            reuse_selection_locked=reuse_locked,
        )

    def _evaluate_variants(
        self,
        split: str,
        variants: tuple[str, ...],
        *,
        reuse_selection_locked: bool = False,
    ) -> StageResult:
        if split == "test":
            require_selection_before_test(self.artifact_root / "selection.json")
        selection = (
            require_selection_before_test(self.artifact_root / "selection.json")
            if reuse_selection_locked
            else None
        )
        aggregate: dict[str, str] = {}
        for variant_id in variants:
            output = self.artifact_root / "evaluation" / split / variant_id.lower()
            metrics_path = output / "metrics.json"
            if selection is not None and variant_id in {"M2", "M3"}:
                expected_hash = selection["val_hashes"][variant_id]
                if not metrics_path.is_file() or sha256_file(metrics_path) != expected_hash:
                    raise RuntimeError(
                        f"selection-locked validation metrics changed: {variant_id}"
                    )
                metrics = _read_json(metrics_path)
            else:
                metrics = run_variant_evaluation(
                    self._best_checkpoint(variant_id),
                    self.data_yaml,
                    self.artifact_root / "dataset" / f"{split}.coco.json",
                    split=split,
                    output_dir=output,
                    device=0,
                )
                metrics["data_exposed"] = True
                metrics["initializer"] = str(self.source_weights)
                atomic_write_json(metrics_path, metrics)
            aggregate[variant_id] = sha256_value(metrics)
        aggregate_name = "aggregate.json" if variants == EVALUATION_MODELS else "partial.aggregate.json"
        path = self.artifact_root / "evaluation" / split / aggregate_name
        atomic_write_json(path, aggregate)
        return StageResult({split: sha256_file(path)})

    def _profile_variant(
        self, variant_id: str, output_root: Path, *, measure_latency: bool = False
    ) -> HardwareProfile:
        from ultralytics import YOLO

        model = YOLO(str(self._best_checkpoint(variant_id)), task="detect").model.float().cpu()
        profile = normalize_profile(
            variant_id,
            profile_module(model, torch.zeros(1, 3, 640, 640)),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = asdict(profile)
        if measure_latency:
            latency = self.values["profiling"]
            gpu_model = YOLO(str(self._best_checkpoint(variant_id)), task="detect").model
            payload["latency"] = asdict(
                profile_gpu_latency(
                    gpu_model,
                    device=torch.device("cuda:0"),
                    imgsz=int(self.values["model"]["imgsz"]),
                    precision=str(latency["precision"]),
                    batch=int(latency["batch"]),
                    warmup=int(latency["warmup"]),
                    iterations=int(latency["iterations"]),
                )
            )
            del gpu_model
            torch.cuda.empty_cache()
        atomic_write_json(output_root / "profile.json", payload)
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
            self._profile_variant(variant_id, output, measure_latency=True)
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
