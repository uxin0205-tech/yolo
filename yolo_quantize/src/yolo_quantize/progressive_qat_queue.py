"""Materialize and execute the conditional V29 matched short-QAT queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .progressive_queue import ProgressiveQueuePlan
from .qat_plan import Full35QATPlan

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PROJECT_ROOT = _PROJECT_ROOT
_DEFAULT_BASE_QAT_PLAN = (
    _PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
)


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


def _write_same_or_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_same_or_new(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _safe_id(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    if not result:
        raise ValueError("short-QAT candidate id has no safe path representation")
    return result


def _format_payload(format_id: str) -> dict[str, object]:
    uniform_bits = {
        "exact_w4": 4,
        "exact_w5": 5,
        "exact_w6": 6,
        "exact_w7": 7,
    }
    if format_id in uniform_bits:
        return {
            "family": "uniform",
            "bits": uniform_bits[format_id],
            "scale_method": "optimal_scaled_codebook",
        }
    if format_id == "fixed_sd4":
        return {
            "family": "ls_sd4",
            "scale_method": "optimal_scaled_codebook",
        }
    if format_id == "twn_v3_filterwise":
        return {
            "family": "twn_filterwise",
            "threshold_multiplier": 0.75,
        }
    if format_id == "exact_scaled_ternary":
        return {
            "family": "exact_scaled_ternary",
            "scale_method": "optimal_scaled_codebook",
        }
    if format_id == "paper_twn_v2":
        return {
            "family": "paper_twn",
            "threshold_multiplier": 0.7,
        }
    raise ValueError(f"unsupported progressive short-QAT format: {format_id}")


class ProgressiveShortQATMaterializer:
    """Turn PTQ recovery records into immutable, parser-verified paired QAT plans."""

    def __init__(
        self,
        source_queue: str | Path,
        *,
        output_root: str | Path | None = None,
        base_qat_plan: str | Path = _DEFAULT_BASE_QAT_PLAN,
        plan_namespace: str = "v29",
        qat_run_root: str | Path | None = None,
        authorization_id: str = "user-2026-09-04-arrange-and-continue-training-queue",
        plan_date: str = "2026-09-04",
    ) -> None:
        self.source_queue = Path(source_queue).expanduser().resolve()
        self.base_qat_plan = Path(base_qat_plan).expanduser().resolve()
        self.plan_namespace = plan_namespace.strip()
        self.qat_run_root = (
            (_PROJECT_ROOT / "artifacts/runs/qat/v29-progressive-short").resolve()
            if qat_run_root is None
            else Path(qat_run_root).expanduser().resolve()
        )
        self.authorization_id = authorization_id.strip()
        self.plan_date = plan_date.strip()
        if not all((self.plan_namespace, self.authorization_id, self.plan_date)):
            raise ValueError("short-QAT namespace, authorization and date are required")
        self._requested_output_root = (
            None if output_root is None else Path(output_root).expanduser().resolve()
        )

    def _load(self) -> tuple[dict[str, Any], ProgressiveQueuePlan]:
        if not self.source_queue.is_file():
            raise FileNotFoundError(self.source_queue)
        payload = _mapping(
            json.loads(self.source_queue.read_text(encoding="utf-8")),
            "short-QAT source queue",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("short-QAT source queue schema_version must be 1")
        source_plan_path = Path(str(payload["source_queue_plan"])).resolve()
        plan = ProgressiveQueuePlan.from_yaml(source_plan_path)
        if (
            payload.get("source_queue_plan_sha256") != plan.config_sha256
            or payload.get("queue_id") != f"{plan.queue_id}--short-qat"
            or payload.get("source_parent_manifest_sha256") != plan.parent.config_sha256
            or Path(str(payload.get("source_parent_manifest"))).resolve()
            != plan.parent.config_path
            or payload.get("execution_authorized") is not True
            or payload.get("matched_sham_required") is not True
            or payload.get("formal_validation") is not False
            or payload.get("long_qat_authorized") is not False
        ):
            raise ValueError("short-QAT source queue differs from the reviewed plan")
        recipe = _mapping(payload.get("recipe"), "short-QAT recipe")
        schedule = _mapping(recipe.get("schedule"), "short-QAT schedule")
        batch = _mapping(recipe.get("batch"), "short-QAT batch")
        expected_recipe = {
            "optimizer": plan.short_qat.optimizer,
            "epochs": plan.short_qat.epochs,
            "patience": plan.short_qat.patience,
            "fp32_epochs": plan.short_qat.fp32_epochs,
            "ramp_epochs": plan.short_qat.progressive_ramp_epochs,
            "full_epochs": plan.short_qat.full_quant_epochs,
            "detect_logical": plan.short_qat.detect_logical_batch,
            "detect_microbatch": plan.short_qat.detect_microbatch,
            "pose": plan.short_qat.pose_batch,
            "added_noise": plan.short_qat.added_noise,
            "augmentation": "accepted_full35_exact",
        }
        actual_recipe = {
            "optimizer": recipe.get("optimizer"),
            "epochs": recipe.get("epochs"),
            "patience": recipe.get("patience"),
            "fp32_epochs": schedule.get("fp32_epochs"),
            "ramp_epochs": schedule.get("progressive_ramp_epochs"),
            "full_epochs": schedule.get("full_quant_epochs"),
            "detect_logical": batch.get("detect_logical"),
            "detect_microbatch": batch.get("detect_microbatch"),
            "pose": batch.get("pose"),
            "added_noise": recipe.get("added_noise"),
            "augmentation": recipe.get("augmentation"),
        }
        if actual_recipe != expected_recipe:
            raise ValueError("short-QAT source recipe drifted")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list) or len(jobs) > plan.short_qat.maximum_candidates:
            raise ValueError("short-QAT source job count is invalid")
        expected_status = (
            "ready_for_qat_plan_materialization"
            if jobs
            else "no_eligible_recovery_candidates"
        )
        if payload.get("status") != expected_status:
            raise ValueError("short-QAT source status differs from its jobs")
        return payload, plan

    @staticmethod
    def _job_assignments(
        base: Mapping[str, object],
        job: Mapping[str, object],
    ) -> list[dict[str, object]]:
        policy = _mapping(base.get("weight_policy"), "base QAT weight policy")
        defaults = policy.get("assignments")
        if not isinstance(defaults, list) or len(defaults) != 10:
            raise ValueError("base short-QAT policy must contain ten W8 defaults")
        result = [dict(_mapping(item, "base W8 assignment")) for item in defaults]
        raw_assignments = job.get("assignments")
        if not isinstance(raw_assignments, list) or not raw_assignments:
            raise ValueError("short-QAT job requires cumulative path assignments")
        seen: set[str] = set()
        for raw in raw_assignments:
            item = _mapping(raw, "short-QAT path assignment")
            path = str(item["path"])
            if path in seen:
                raise ValueError("short-QAT path assignments must be unique")
            seen.add(path)
            result.append(
                {
                    "region": str(item["region"]),
                    "paths": [path],
                    "format": _format_payload(str(item["format_id"])),
                }
            )
        return result

    def _materialize_job(
        self,
        *,
        index: int,
        job: Mapping[str, object],
        queue_plan: ProgressiveQueuePlan,
        base: Mapping[str, object],
        output_root: Path,
    ) -> dict[str, object]:
        candidate_id = str(job["candidate_id"])
        if job.get("status") != "completed":
            raise ValueError("short-QAT candidate is incomplete")
        gate = _mapping(job.get("gate"), "short-QAT candidate gate")
        if gate.get("decision") != "recover":
            raise ValueError("only PTQ recover candidates may enter short QAT")
        if int(job.get("savings_from_locked_parent_bytes", 0)) <= 0:
            raise ValueError("short-QAT candidate has no material capacity saving")
        job_root = output_root / "jobs" / f"{index:02d}-{_safe_id(candidate_id)}"
        evidence_path = job_root / "candidate-evidence.json"
        evidence = {
            "schema_version": 1,
            "status": "completed",
            "source_queue": str(self.source_queue),
            "source_queue_sha256": _sha256(self.source_queue),
            "results": {candidate_id: dict(job)},
            "formal_validation": False,
        }
        _write_json(evidence_path, evidence)
        evidence_sha256 = _sha256(evidence_path)
        dual_path = job_root / "candidate-dual-regate.json"
        dual = {
            "schema_version": 1,
            "status": "completed",
            "source": {
                "report": str(evidence_path),
                "report_sha256": evidence_sha256,
            },
            "candidates": {
                candidate_id: {
                    "decision": "recover",
                    "worst_total_map50_delta": gate.get("worst_total_map50_delta"),
                    "worst_total_map50_95_delta": gate.get(
                        "worst_total_map50_95_delta"
                    ),
                }
            },
            "formal_validation": False,
        }
        _write_json(dual_path, dual)

        plan_payload = yaml.safe_load(yaml.safe_dump(dict(base), sort_keys=False))
        if not isinstance(plan_payload, dict):
            raise TypeError("base QAT plan copy is not a mapping")
        plan_id = (
            f"{self.plan_namespace}-short-qat-{index:02d}-{_safe_id(candidate_id)}"
        )
        plan_payload["plan_id"] = plan_id
        plan_payload["date"] = self.plan_date
        plan_payload["execution_authorization"] = {
            "authorization_id": self.authorization_id,
            "arms": ["sham", "qat"],
            "scope": "paired_progressive_short_qat_search_only",
        }
        sources = _mapping(plan_payload.get("sources"), "generated QAT sources")
        sources["matched_metrics"] = {
            "path": str(queue_plan.locked_parent_metrics_path),
            "sha256": _sha256(queue_plan.locked_parent_metrics_path),
        }
        sources["weight_plan"] = {
            "path": str(queue_plan.config_path),
            "sha256": queue_plan.config_sha256,
        }
        sources["candidate_evidence"] = {
            "path": str(evidence_path),
            "sha256": evidence_sha256,
        }
        sources["candidate_dual_regate"] = {
            "path": str(dual_path),
            "sha256": _sha256(dual_path),
        }
        for record in sources.values():
            source_record = _mapping(record, "generated QAT source record")
            configured = Path(str(source_record["path"])).expanduser()
            if not configured.is_absolute():
                source_record["path"] = str(
                    (_SOURCE_PROJECT_ROOT / configured).resolve()
                )
        data = _mapping(plan_payload.get("data"), "generated QAT data")
        for record in data.values():
            if not isinstance(record, dict) or "path" not in record:
                continue
            configured = Path(str(record["path"])).expanduser()
            if not configured.is_absolute():
                record["path"] = str((_SOURCE_PROJECT_ROOT / configured).resolve())
        plan_payload["warm_start"] = {
            "locked_parent": {
                "path": str(queue_plan.parent.config_path),
                "sha256": queue_plan.parent.config_sha256,
            },
            "checkpoint_role": "full_resume",
            "state_key": "ema_state",
            "reset_path_override_quantizers": True,
            "optimizer_state": "fresh",
        }
        plan_payload["weight_policy"] = {
            "candidate_id": candidate_id,
            "assignments": self._job_assignments(base, job),
        }
        training = _mapping(plan_payload.get("training"), "generated QAT training")
        training.update(
            {
                "epochs": queue_plan.short_qat.epochs,
                "patience": queue_plan.short_qat.patience,
                "warmup_epochs": 1,
                "scale_only_epochs": 9,
                "progressive_start_epoch": 2,
                "progressive_full_epoch": 9,
                "detect_logical_batch": (queue_plan.short_qat.detect_logical_batch),
                "detect_microbatch": (queue_plan.short_qat.detect_microbatch),
                "pose_batch": queue_plan.short_qat.pose_batch,
                "added_noise": False,
            }
        )
        plan_payload["run_root"] = str(self.qat_run_root)
        plan_path = job_root / "qat-plan.yaml"
        plan_text = yaml.safe_dump(
            plan_payload,
            allow_unicode=True,
            sort_keys=False,
        )
        _write_same_or_new(plan_path, plan_text)
        parsed = Full35QATPlan.from_yaml(plan_path)
        if (
            parsed.plan_id != plan_id
            or parsed.candidate_id != candidate_id
            or parsed.warm_start is None
        ):
            raise RuntimeError("materialized short-QAT plan did not round-trip")
        return {
            "index": index,
            "candidate_id": candidate_id,
            "plan_id": plan_id,
            "plan": str(plan_path),
            "plan_sha256": parsed.config_sha256,
            "evidence": str(evidence_path),
            "evidence_sha256": evidence_sha256,
            "dual_regate": str(dual_path),
            "dual_regate_sha256": _sha256(dual_path),
            "arms": ["sham", "qat"],
            "status": "queued",
        }

    def materialize(self) -> dict[str, object]:
        source, queue_plan = self._load()
        output_root = (
            queue_plan.run_root / "short-qat"
            if self._requested_output_root is None
            else self._requested_output_root
        )
        jobs = source["jobs"]
        if not isinstance(jobs, list):
            raise TypeError("short-QAT jobs must be a list")
        if not self.base_qat_plan.is_file():
            raise FileNotFoundError(self.base_qat_plan)
        base = _mapping(
            yaml.safe_load(self.base_qat_plan.read_text(encoding="utf-8")),
            "base QAT plan",
        )
        materialized = [
            self._materialize_job(
                index=index,
                job=_mapping(job, "short-QAT job"),
                queue_plan=queue_plan,
                base=base,
                output_root=output_root,
            )
            for index, job in enumerate(jobs)
        ]
        manifest = {
            "schema_version": 1,
            "queue_id": f"{queue_plan.queue_id}--materialized-short-qat",
            "status": "ready" if materialized else "complete_no_jobs",
            "output_root": str(output_root),
            "source_queue": str(self.source_queue),
            "source_queue_sha256": _sha256(self.source_queue),
            "source_queue_plan": str(queue_plan.config_path),
            "source_queue_plan_sha256": queue_plan.config_sha256,
            "parent_id": queue_plan.parent.parent_id,
            "parent_manifest_sha256": queue_plan.parent.config_sha256,
            "jobs": materialized,
            "execution_authorized": queue_plan.short_qat.execution_authorized,
            "formal_validation": False,
            "long_qat_authorized": False,
        }
        manifest_path = output_root / "materialized-queue.json"
        _write_json(manifest_path, manifest)
        return manifest


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ProgressiveShortQATRunner:
    """Persistent executor: wait for PTQ, then run each matched sham before QAT."""

    def __init__(
        self,
        source_queue: str | Path,
        *,
        output_root: str | Path | None = None,
        runtime_factory: Callable[[Path], Any] | None = None,
        gpu_pid_probe: Callable[[int], tuple[int, ...]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        maximum_retries_per_arm: int = 1,
        base_qat_plan: str | Path = _DEFAULT_BASE_QAT_PLAN,
        plan_namespace: str = "v29",
        qat_run_root: str | Path | None = None,
        authorization_id: str = "user-2026-09-04-arrange-and-continue-training-queue",
        plan_date: str = "2026-09-04",
    ) -> None:
        self.source_queue = Path(source_queue).expanduser().resolve()
        self.output_root = (
            None if output_root is None else Path(output_root).expanduser().resolve()
        )
        if runtime_factory is None:
            from .qat_runtime import Full35QATRuntime

            runtime_factory = Full35QATRuntime.from_yaml
        if gpu_pid_probe is None:
            from .progressive_queue import _foreign_gpu_pids

            gpu_pid_probe = _foreign_gpu_pids
        if maximum_retries_per_arm < 0:
            raise ValueError("short-QAT retries cannot be negative")
        self.runtime_factory = runtime_factory
        self.gpu_pid_probe = gpu_pid_probe
        self.sleep = sleep
        self.maximum_retries_per_arm = maximum_retries_per_arm
        self.base_qat_plan = Path(base_qat_plan).expanduser().resolve()
        self.plan_namespace = plan_namespace
        self.qat_run_root = (
            (_PROJECT_ROOT / "artifacts/runs/qat/v29-progressive-short").resolve()
            if qat_run_root is None
            else Path(qat_run_root).expanduser().resolve()
        )
        self.authorization_id = authorization_id
        self.plan_date = plan_date

    @staticmethod
    def _completion(runtime: Any, arm: str) -> Path | None:
        run_dir = runtime.plan.run_root / runtime.run_name(arm)
        manifest = run_dir / "qat-experiment.json"
        if not manifest.is_file():
            return None
        payload = _mapping(
            json.loads(manifest.read_text(encoding="utf-8")),
            "short-QAT arm completion",
        )
        expected = {
            "schema_version": 1,
            "plan_sha256": runtime.plan.config_sha256,
            "candidate_id": runtime.plan.candidate_id,
            "arm": arm,
            "run_name": runtime.run_name(arm),
            "completed_stages": ["j3"],
            "formal_validation": False,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("short-QAT arm completion manifest drifted")
        return manifest

    @staticmethod
    def _last_checkpoint(runtime: Any, arm: str) -> Path | None:
        path = runtime.plan.run_root / runtime.run_name(arm) / "checkpoints" / "last.pt"
        return path if path.is_file() else None

    @staticmethod
    def _event(root: Path, kind: str, **values: object) -> None:
        payload = {
            "schema_version": 1,
            "kind": kind,
            "time_unix": time.time(),
            "values": values,
        }
        root.mkdir(parents=True, exist_ok=True)
        with (root / "execution-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        _atomic_json(root / "execution-status.json", payload)

    def _wait_for_gpu(
        self,
        *,
        root: Path,
        device: int,
        poll_seconds: int,
        candidate_id: str,
        arm: str,
    ) -> None:
        while True:
            pids = self.gpu_pid_probe(device)
            if not pids:
                self._event(
                    root,
                    "gpu_acquired",
                    candidate_id=candidate_id,
                    arm=arm,
                    device=device,
                )
                return
            self._event(
                root,
                "waiting_for_gpu",
                candidate_id=candidate_id,
                arm=arm,
                device=device,
                foreign_pids=list(pids),
                poll_seconds=poll_seconds,
            )
            self.sleep(float(poll_seconds))

    def _load_state(
        self,
        *,
        root: Path,
        materialized: Mapping[str, object],
    ) -> dict[str, Any]:
        path = root / "execution-state.json"
        if path.is_file():
            state = _mapping(
                json.loads(path.read_text(encoding="utf-8")),
                "short-QAT execution state",
            )
            if state.get("queue_id") != materialized.get("queue_id") or state.get(
                "source_queue_sha256"
            ) != materialized.get("source_queue_sha256"):
                raise ValueError("existing short-QAT execution state drifted")
            return state
        jobs = materialized.get("jobs")
        if not isinstance(jobs, list):
            raise TypeError("materialized short-QAT jobs must be a list")
        state = {
            "schema_version": 1,
            "queue_id": materialized["queue_id"],
            "source_queue_sha256": materialized["source_queue_sha256"],
            "status": "running",
            "jobs": [
                {
                    "candidate_id": _mapping(job, "materialized QAT job")[
                        "candidate_id"
                    ],
                    "plan": _mapping(job, "materialized QAT job")["plan"],
                    "plan_sha256": _mapping(job, "materialized QAT job")["plan_sha256"],
                    "arms": {
                        "sham": {"status": "pending"},
                        "qat": {"status": "pending"},
                    },
                }
                for job in jobs
            ],
            "formal_validation": False,
            "long_qat_authorized": False,
        }
        _atomic_json(path, state)
        return state

    def run(self) -> dict[str, Any]:
        materialized = ProgressiveShortQATMaterializer(
            self.source_queue,
            output_root=self.output_root,
            base_qat_plan=self.base_qat_plan,
            plan_namespace=self.plan_namespace,
            qat_run_root=self.qat_run_root,
            authorization_id=self.authorization_id,
            plan_date=self.plan_date,
        ).materialize()
        root = Path(str(materialized["output_root"])).resolve()
        state = self._load_state(root=root, materialized=materialized)
        if state.get("status") in {"complete", "complete_no_jobs"}:
            return state
        jobs = materialized.get("jobs")
        state_jobs = state.get("jobs")
        if not isinstance(jobs, list) or not isinstance(state_jobs, list):
            raise TypeError("short-QAT execution jobs must be lists")
        if not jobs:
            state["status"] = "complete_no_jobs"
            _atomic_json(root / "execution-state.json", state)
            self._event(root, "complete_no_jobs")
            return state

        queue_plan = ProgressiveQueuePlan.from_yaml(
            Path(str(materialized["source_queue_plan"]))
        )
        for job, state_job in zip(jobs, state_jobs, strict=True):
            job_record = _mapping(job, "materialized short-QAT job")
            execution = _mapping(state_job, "short-QAT execution job")
            if execution.get("candidate_id") != job_record.get("candidate_id"):
                raise ValueError("short-QAT execution job identity drifted")
            runtime = self.runtime_factory(Path(str(job_record["plan"])))
            arms = _mapping(execution.get("arms"), "short-QAT execution arms")
            for arm in ("sham", "qat"):
                arm_state = _mapping(arms.get(arm), f"short-QAT {arm} state")
                if arm_state.get("status") == "completed":
                    continue
                completion = self._completion(runtime, arm)
                if completion is not None:
                    arms[arm] = {"status": "completed"}
                    _atomic_json(root / "execution-state.json", state)
                    continue
                last_error: Exception | None = None
                for attempt in range(self.maximum_retries_per_arm + 1):
                    self._wait_for_gpu(
                        root=root,
                        device=queue_plan.device_index,
                        poll_seconds=queue_plan.poll_seconds,
                        candidate_id=str(job_record["candidate_id"]),
                        arm=arm,
                    )
                    resume = self._last_checkpoint(runtime, arm)
                    arms[arm] = {
                        "status": "running",
                        "attempt": attempt,
                        "resume": None if resume is None else str(resume),
                    }
                    _atomic_json(root / "execution-state.json", state)
                    self._event(
                        root,
                        "arm_started",
                        candidate_id=job_record["candidate_id"],
                        arm=arm,
                        attempt=attempt,
                        resume=None if resume is None else str(resume),
                    )
                    try:
                        runtime.run(
                            arm,
                            device_index=queue_plan.device_index,
                            resume=resume,
                        )
                        completion = self._completion(runtime, arm)
                        if completion is None:
                            raise RuntimeError(
                                "short-QAT runtime returned without completion manifest"
                            )
                        arms[arm] = {"status": "completed"}
                        _atomic_json(root / "execution-state.json", state)
                        self._event(
                            root,
                            "arm_completed",
                            candidate_id=job_record["candidate_id"],
                            arm=arm,
                            completion=str(completion),
                        )
                        last_error = None
                        break
                    except Exception as error:  # noqa: BLE001 - persist runtime failures
                        last_error = error
                        retryable = getattr(error, "retryable", True) is not False
                        arms[arm] = {
                            "status": "failed_attempt",
                            "attempt": attempt,
                            "failure_type": type(error).__name__,
                            "message": str(error),
                            "retryable": retryable,
                        }
                        _atomic_json(root / "execution-state.json", state)
                        self._event(
                            root,
                            "arm_failed",
                            candidate_id=job_record["candidate_id"],
                            arm=arm,
                            attempt=attempt,
                            failure_type=type(error).__name__,
                            message=str(error),
                            retryable=retryable,
                        )
                        if not retryable:
                            break
                if last_error is not None:
                    state["status"] = "failed"
                    _atomic_json(root / "execution-state.json", state)
                    raise last_error
            execution["status"] = "completed"
            _atomic_json(root / "execution-state.json", state)

        state["status"] = "complete"
        _atomic_json(root / "execution-state.json", state)
        self._event(root, "complete", jobs=len(jobs))
        return state

    def wait_and_run(self, *, poll_seconds: int) -> dict[str, Any]:
        if poll_seconds < 30:
            raise ValueError("short-QAT queue poll_seconds must be at least 30")
        wait_root = (
            self.output_root
            if self.output_root is not None
            else self.source_queue.parent / "short-qat"
        )
        while not self.source_queue.is_file():
            self._event(
                wait_root,
                "waiting_for_ptq_queue",
                source_queue=str(self.source_queue),
                poll_seconds=poll_seconds,
            )
            self.sleep(float(poll_seconds))
        return self.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize V29 PTQ recover candidates into paired short-QAT plans"
    )
    parser.add_argument(
        "--source-queue",
        type=Path,
        default=(
            _PROJECT_ROOT / "artifacts/queues/v29-v19-progressive-ptq-to-short-qat-v1/"
            "short-qat-queue.json"
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=0)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize-reviewed-queue", action="store_true")
    action.add_argument("--execute-reviewed-queue", action="store_true")
    args = parser.parse_args(argv)
    if args.materialize_reviewed_queue:
        while not args.source_queue.is_file():
            if args.wait_seconds < 30:
                raise FileNotFoundError(args.source_queue)
            time.sleep(args.wait_seconds)
        result = ProgressiveShortQATMaterializer(
            args.source_queue,
            output_root=args.output_root,
        ).materialize()
    else:
        runner = ProgressiveShortQATRunner(
            args.source_queue,
            output_root=args.output_root,
        )
        if args.source_queue.is_file():
            result = runner.run()
        else:
            if args.wait_seconds < 30:
                raise FileNotFoundError(args.source_queue)
            result = runner.wait_and_run(poll_seconds=args.wait_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


__all__ = (
    "ProgressiveShortQATMaterializer",
    "ProgressiveShortQATRunner",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
