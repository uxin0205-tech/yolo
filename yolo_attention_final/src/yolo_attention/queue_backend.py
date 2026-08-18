"""Live queue dispatch, constructed only behind the CLI ``--execute`` gate."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .config import BasisKind, VariantConfig
from .evaluation import (
    EvaluationRecipe,
    EvaluationRequest,
    UltralyticsEvaluationBackend,
)
from .profiling import write_variant_profile
from .queue_model import JobKind, JobStatus, QueueJob, QueueResult, QueueState
from .queue_policy import (
    SelectionDecision,
    SelectionInputError,
    select_architecture,
    select_bias,
    select_d1,
    select_d2,
    select_final,
    select_n0,
    select_normalization,
    select_r_denominator,
    select_recovery,
    select_scale,
)
from .run_config import TrainingRecipe


class EvaluationBackend(Protocol):
    def evaluate_official(self, request: EvaluationRequest) -> QueueResult: ...

    def evaluate_variant(self, request: EvaluationRequest) -> QueueResult: ...


class PWLRunner(Protocol):
    def score_analysis(self, request: EvaluationRequest) -> QueueResult: ...

    def compare(self, request: EvaluationRequest, *, score_run_dir: Path) -> QueueResult: ...


TrainingLauncher = Callable[[object], object]


def completed_training_checkpoint(run_dir: Path, expected_epochs: int) -> Path | None:
    """Return best.pt only when immutable Ultralytics outputs prove training completed."""

    weights = run_dir / "ultralytics" / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    results = run_dir / "ultralytics" / "results.csv"
    completion = run_dir / "training-complete.json"
    if not (best.is_file() and last.is_file() and results.is_file()):
        return None
    with results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "epoch" not in rows[-1]:
        return None
    try:
        completed_epochs = int(float(rows[-1]["epoch"]))
    except (TypeError, ValueError):
        return None
    completed_normally = completion.is_file() or completed_epochs >= expected_epochs
    return best.resolve() if completed_normally else None


P0Runner = Callable[[QueueJob, Path], QueueResult]
P0_REQUIRED_PATHS = (
    "model.10.m.0.attn",
    "model.10",
    "model.22.m.0.1.attn",
    "model.22",
)


class ResearchQueueBackend:
    """Dispatch one already-authorized queue job; it never owns scheduling state."""

    N0_COST_ORDER = (
        "n0-shift",
        "n0-hsig",
        "n0-relu",
        "n0-mk1",
        "n0-mk3",
        "n0-mk5",
        "n0-pwl",
        "n0-lut",
        "n0-exact",
    )

    def __init__(
        self,
        *,
        project_root: str | Path,
        evaluation_backend: EvaluationBackend | None = None,
        training_launcher: TrainingLauncher | None = None,
        p0_runner: P0Runner | None = None,
        pwl_runner: PWLRunner | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runs_root = self.project_root / "artifacts" / "runs"
        self.evaluation = evaluation_backend or UltralyticsEvaluationBackend()
        self._training_launcher = training_launcher
        self._p0_runner = p0_runner or run_p0_equivalence
        self._pwl_runner = pwl_runner

    def execute(self, job: QueueJob, state: QueueState) -> QueueResult | SelectionDecision:
        if job.kind is JobKind.SELECT:
            return self._select(job, state)
        if job.id in {"pwl-score-analysis", "pwl-compare"}:
            return self._run_pwl(job)
        if job.kind is JobKind.EVALUATE:
            return self._evaluate(job)
        if job.kind is JobKind.TRAIN:
            return self._train(job)
        if job.kind is JobKind.VALIDATE:
            return self._p0_runner(job, self.runs_root / job.id)
        raise ValueError(f"unsupported queue job kind: {job.kind.value}")

    def _run_pwl(self, job: QueueJob) -> QueueResult:
        runner = self._pwl_runner
        if runner is None:
            from .pwl_experiment import PWLExperimentRunner

            runner = PWLExperimentRunner(project_root=self.project_root)
        request = self._request(job)
        if job.id == "pwl-score-analysis":
            return runner.score_analysis(request)
        return runner.compare(
            request,
            score_run_dir=self.runs_root / "pwl-score-analysis",
        )

    @staticmethod
    def _require_path(value: str | None, field: str) -> Path:
        if value is None:
            raise ValueError(f"job is missing {field}")
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{field} does not exist: {path}")
        return path

    def _request(self, job: QueueJob, *, checkpoint: Path | None = None) -> EvaluationRequest:
        recipe_path = (
            Path(job.evaluation_path)
            if job.evaluation_path is not None
            else self.project_root / "configs" / "evaluation" / "coco2017.yaml"
        )
        return EvaluationRequest(
            run_id=job.id,
            run_dir=self.runs_root / job.id,
            parent_checkpoint=checkpoint or self._require_path(job.parent_checkpoint, "parent checkpoint"),
            recipe=EvaluationRecipe.from_yaml(recipe_path),
            variant_path=Path(job.variant_path).resolve() if job.variant_path else None,
        )

    def _evaluate(self, job: QueueJob) -> QueueResult:
        request = self._request(job)
        if job.id == "b26-fp":
            return self.evaluation.evaluate_official(request)
        if request.variant_path is None:
            raise ValueError(f"variant evaluation {job.id} has no variant configuration")
        result = self.evaluation.evaluate_variant(request)
        return self._attach_profile(result, request.variant_path, request.run_dir)

    def _train(self, job: QueueJob) -> QueueResult:
        from .runner import TrainingRequest, launch_training

        variant_path = self._require_path(job.variant_path, "variant configuration")
        training_path = self._require_path(job.training_path, "training configuration")
        parent = self._require_path(job.parent_checkpoint, "parent checkpoint")
        recipe = replace(TrainingRecipe.from_yaml(training_path), weights=str(parent))
        request = TrainingRequest(
            variant=VariantConfig.from_yaml(variant_path),
            training=recipe,
            artifacts_root=self.runs_root,
            run_id=job.id,
            project_root=self.project_root,
        )
        run_dir = self.runs_root / job.id
        best = completed_training_checkpoint(run_dir, recipe.epochs)
        if best is None:
            launcher = self._training_launcher or launch_training
            launcher(request)
            best = completed_training_checkpoint(run_dir, recipe.epochs)
        if best is None:
            raise FileNotFoundError(
                f"training did not produce complete best/last/results artifacts in {run_dir}"
            )
        # The checkpoint already contains the converted attention modules and
        # trained parameters, so evaluating it must not convert a second time.
        evaluation_request = self._request(job, checkpoint=best)
        result = self.evaluation.evaluate_official(evaluation_request)
        return self._attach_profile(result, variant_path, evaluation_request.run_dir)

    @staticmethod
    def _attach_profile(result: QueueResult, variant_path: Path, run_dir: Path) -> QueueResult:
        profile = write_variant_profile(run_dir, VariantConfig.from_yaml(variant_path))
        if result.metrics_path is None:
            raise ValueError("evaluation result is missing metrics_path")
        metrics_path = Path(result.metrics_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("queue result metrics must contain an object")
        payload["profile_path"] = str(profile)
        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return replace(result, profile_path=str(profile))

    @staticmethod
    def _job_result(state: QueueState, job_id: str) -> QueueResult:
        parent = state.job(job_id)
        if parent.status is not JobStatus.SUCCEEDED or parent.result is None:
            raise SelectionInputError(f"selection parent {job_id!r} has no successful result")
        if parent.result.map50_95 is None:
            raise SelectionInputError(f"selection parent {job_id!r} is missing map50_95")
        return parent.result

    def _metrics(self, state: QueueState, ids: tuple[str, ...]) -> dict[str, float]:
        return {job_id: self._job_result(state, job_id).map50_95 for job_id in ids}

    @staticmethod
    def _decision_winner(state: QueueState, job_id: str) -> str:
        decision = state.job(job_id).decision
        if not decision or len(decision.get("winners", ())) != 1:
            raise SelectionInputError(f"selection {job_id!r} has no unique winner")
        return decision["winners"][0]

    @staticmethod
    def _profile(result: QueueResult, job_id: str) -> dict[str, object]:
        if result.profile_path is None:
            raise SelectionInputError(f"selection parent {job_id!r} is missing profile_path")
        path = Path(result.profile_path)
        if not path.is_file():
            raise SelectionInputError(f"selection profile does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SelectionInputError(f"profile {path} must contain an object")
        return payload

    def _profiles(self, state: QueueState, ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
        return {job_id: self._profile(self._job_result(state, job_id), job_id) for job_id in ids}

    def _a0_winner(self, state: QueueState) -> str:
        return self._decision_winner(state, "a0-select")

    def _select(self, job: QueueJob, state: QueueState) -> SelectionDecision:
        ids = job.parent_job_ids
        if job.id == "architecture-select":
            return select_architecture(self._metrics(state, ids))
        if job.id == "recovery-select":
            return select_recovery(self._metrics(state, ids))
        if job.id == "scale-select":
            return select_scale(self._metrics(state, ids))
        if job.id == "a0-select":
            return select_bias(self._metrics(state, ids))
        if job.id == "n0-select":
            a0 = self._job_result(state, self._a0_winner(state)).map50_95
            return select_n0(self._metrics(state, ids), a0_map=a0, cost_order=self.N0_COST_ORDER)
        if job.id == "normalization-select":
            candidates = ("n0-exact",) + tuple(name for name in ids if name.startswith("n1-"))
            return select_normalization(
                self._metrics(state, candidates),
                self._profiles(state, candidates),
            )
        if job.id == "d0-select":
            return SelectionDecision((ids[0],), (), "D0 reference accepted")
        if job.id == "d1-select":
            return select_d1(self._metrics(state, ids))
        if job.id == "d1-confirm-select":
            primary = self._decision_winner(state, "d1-select")
            primary_map = self._job_result(state, primary).map50_95
            seed_map = self._job_result(state, "d1-seed1").map50_95
            gap = abs(primary_map - seed_map)
            return SelectionDecision(
                (primary,),
                ("d1-seed1",),
                f"recorded seed-0/seed-1 absolute mAP gap {gap:.6f}; retained seed-0 parent",
            )
        if job.id == "d2-select":
            a0 = self._job_result(state, self._a0_winner(state)).map50_95
            return select_d2(self._metrics(state, ids), a0_map=a0)
        if job.id == "denominator-select":
            results = {name: self._job_result(state, name) for name in ids}
            row_error = results["r1-rlut"].row_sum_max_error
            if row_error is None:
                raise SelectionInputError("R1 result is missing row_sum_max_error")
            return select_r_denominator(
                r0_map=results["r0-div"].map50_95,
                r1_map=results["r1-rlut"].map50_95,
                r1_row_sum_max_error=row_error,
            )
        if job.id == "bdcn-select":
            parent = ids[0]
            winner = self._decision_winner(state, parent) if parent == "denominator-select" else parent
            return SelectionDecision((winner,), (), "accepted BDCN denominator result")
        if job.id == "final-select":
            candidates = (
                self._a0_winner(state),
                self._decision_winner(state, "normalization-select"),
                self._decision_winner(state, "bdcn-select"),
            )
            candidates = tuple(dict.fromkeys(candidates))
            return select_final(self._metrics(state, candidates), self._profiles(state, candidates))
        raise ValueError(f"unsupported selection job: {job.id}")


def _tensor_leaves(value: object):
    import torch

    if isinstance(value, torch.Tensor):
        yield value.detach().float().cpu().reshape(-1)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _tensor_leaves(value[key])


def build_p0_equivalence_report(
    required_pairs: dict[str, tuple[object, object]],
    *,
    final_pairs: tuple[tuple[object, object], ...],
    tolerance: float,
) -> dict[str, object]:
    """Gate research boundaries and record decoded-output amplification separately."""

    import torch

    if tolerance <= 0 or set(required_pairs) != set(P0_REQUIRED_PATHS):
        raise ValueError("P0 report requires every Attention/C2PSA path and positive tolerance")

    def difference(left: object, right: object) -> float:
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError("P0 comparisons require tensors")
        if left.shape != right.shape:
            raise ValueError(f"P0 tensor shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
        return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max().item())

    by_path = {path: difference(left, right) for path, (left, right) in required_pairs.items()}
    required_max = max(by_path.values())
    decoded_errors = [difference(left, right) for left, right in final_pairs]
    return {
        "passed": required_max <= tolerance,
        "tolerance": tolerance,
        "required_max_abs_error": required_max,
        "required_max_abs_error_by_path": by_path,
        "decoded_max_abs_error": max(decoded_errors) if decoded_errors else None,
        "decoded_output_is_diagnostic_only": True,
    }


def run_p0_equivalence(job: QueueJob, run_dir: Path) -> QueueResult:
    """Compare official and P0-converted YOLO26 outputs on deterministic CPU input."""

    import torch
    from ultralytics import YOLO

    from .integration import convert_yolo26_model

    if job.parent_checkpoint is None or not Path(job.parent_checkpoint).is_file():
        raise FileNotFoundError(f"P0 parent checkpoint does not exist: {job.parent_checkpoint}")
    if job.variant_path is None:
        raise ValueError("P0 job is missing variant configuration")
    config = VariantConfig.from_yaml(job.variant_path)
    if config.basis is not BasisKind.FP:
        raise ValueError("P0 validation requires the FP basis")
    official = YOLO(str(Path(job.parent_checkpoint).resolve())).model.float().eval().cpu()
    converted = copy.deepcopy(official)
    paths = convert_yolo26_model(converted, config)
    captures: dict[str, dict[str, object]] = {"official": {}, "converted": {}}

    def capture(bucket: dict[str, object], path: str):
        def hook(_module, _inputs, output):
            bucket[path] = output.detach().clone()

        return hook

    for label, model in (("official", official), ("converted", converted)):
        for path in P0_REQUIRED_PATHS:
            model.get_submodule(path).register_forward_hook(capture(captures[label], path))
    generator = torch.Generator(device="cpu").manual_seed(0)
    sample = torch.randn(1, 3, 64, 64, generator=generator)
    with torch.no_grad():
        expected = list(_tensor_leaves(official(sample)))
        actual = list(_tensor_leaves(converted(sample)))
    if len(expected) != len(actual) or not expected:
        raise RuntimeError("P0 output structure differs from official YOLO26")
    report = build_p0_equivalence_report(
        {path: (captures["official"][path], captures["converted"][path]) for path in P0_REQUIRED_PATHS},
        final_pairs=tuple(zip(expected, actual, strict=True)),
        tolerance=1e-4,
    )
    report["converted_paths"] = paths
    if not report["passed"]:
        raise RuntimeError(
            f"P0 Attention/C2PSA equivalence failed: max_abs_error={report['required_max_abs_error']:.8g}"
        )
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "queue-result.json"
    equivalence_path = metrics_dir / "p0-equivalence.json"
    equivalence_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "map50_95": None,
        "map50": None,
        "map75": None,
        "maps": [],
        "row_sum_max_error": None,
        "checkpoint_path": str(Path(job.parent_checkpoint).resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "profile_path": None,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return QueueResult(
        checkpoint_path=payload["checkpoint_path"],
        metrics_path=payload["metrics_path"],
    )
