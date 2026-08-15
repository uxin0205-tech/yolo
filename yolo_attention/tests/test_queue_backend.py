from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from yolo_attention.config import VariantConfig
from yolo_attention.evaluation import write_standard_result
from yolo_attention.queue_backend import ResearchQueueBackend, build_p0_equivalence_report
from yolo_attention.queue_model import JobKind, JobStatus, QueueJob, QueueResult, QueueState
from yolo_attention.queue_policy import SelectionDecision
from yolo_attention.run_config import TrainingRecipe


def _state(*jobs: QueueJob, project_root: Path) -> QueueState:
    return QueueState.initial(tuple(jobs), project_root=str(project_root))


def _result(tmp_path: Path, name: str, value: float, *, row_error: float | None = None) -> QueueResult:
    checkpoint = tmp_path / name / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    metrics = tmp_path / name / "queue-result.json"
    metrics.write_text("{}\n", encoding="utf-8")
    return QueueResult(
        map50_95=value,
        checkpoint_path=str(checkpoint),
        metrics_path=str(metrics),
        row_sum_max_error=row_error,
    )


def test_selection_dispatch_reads_only_completed_parent_results(tmp_path: Path) -> None:
    jobs = tuple(
        QueueJob(
            id=name,
            run_name=name,
            stage="test",
            kind=JobKind.TRAIN,
            order=index,
            status=JobStatus.SUCCEEDED,
            result=_result(tmp_path, name, value),
        )
        for index, (name, value) in enumerate((("i-scr", 0.4000), ("h-scr", 0.4008), ("t5-scr", 0.4009)))
    )
    select = QueueJob(
        id="architecture-select",
        run_name="ARCHITECTURE-SELECT",
        stage="test",
        kind=JobKind.SELECT,
        order=3,
        status=JobStatus.RUNNING,
        parent_job_ids=("i-scr", "h-scr", "t5-scr"),
    )

    decision = ResearchQueueBackend(project_root=tmp_path).execute(
        select, _state(*jobs, select, project_root=tmp_path)
    )

    assert isinstance(decision, SelectionDecision)
    assert decision.winners == ("i-scr",)


def test_evaluation_dispatch_uses_official_only_for_baseline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.touch()
    recipe_path = tmp_path / "evaluation.yaml"
    recipe_path.write_text(
        "data: coco.yaml\nimgsz: 64\nbatch: 1\ndevice: cpu\nworkers: 0\nsplit: val\nplots: false\n",
        encoding="utf-8",
    )
    variant_path = tmp_path / "variant.yaml"
    VariantConfig(name="I").to_yaml(variant_path)
    calls: list[str] = []

    class FakeEvaluation:
        def evaluate_official(self, request):
            calls.append("official")
            return _result(tmp_path, request.run_id, 0.4)

        def evaluate_variant(self, request):
            calls.append("variant")
            return _result(tmp_path, request.run_id, 0.39)

    backend = ResearchQueueBackend(project_root=tmp_path, evaluation_backend=FakeEvaluation())
    baseline = QueueJob(
        id="b26-fp",
        run_name="B26-FP",
        stage="baseline",
        kind=JobKind.EVALUATE,
        order=0,
        status=JobStatus.RUNNING,
        evaluation_path=str(recipe_path),
        parent_checkpoint=str(checkpoint),
    )
    variant = replace(baseline, id="n0-lut", run_name="N0-LUT", order=1, variant_path=str(variant_path))

    backend.execute(baseline, _state(baseline, project_root=tmp_path))
    variant_result = backend.execute(variant, _state(variant, project_root=tmp_path))

    assert calls == ["official", "variant"]
    assert variant_result.profile_path is not None
    assert Path(variant_result.profile_path).is_file()


def test_training_dispatch_replaces_recipe_weight_with_parent_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.touch()
    variant_path = tmp_path / "variant.yaml"
    VariantConfig(name="H").to_yaml(variant_path)
    training_path = tmp_path / "training.yaml"
    TrainingRecipe(
        stage="screening",
        weights="wrong.pt",
        data="coco.yaml",
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        workers=0,
        seed=0,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        amp=False,
    ).to_yaml(training_path)
    evaluation_path = tmp_path / "configs" / "evaluation" / "coco2017.yaml"
    evaluation_path.parent.mkdir(parents=True)
    evaluation_path.write_text(
        "data: coco.yaml\nimgsz: 64\nbatch: 1\ndevice: cpu\nworkers: 0\nsplit: val\nplots: false\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_launch(request):
        seen["request"] = request
        best = request.artifacts_root / request.run_id / "ultralytics" / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.touch()

    class FakeEvaluation:
        def evaluate_official(self, request):
            return write_standard_result(
                request.run_dir,
                {"map50_95": 0.4, "map50": 0.5, "map75": 0.42, "maps": []},
                checkpoint_path=request.parent_checkpoint,
                profile_path=None,
                row_sum_max_error=None,
            )

    job = QueueJob(
        id="h-scr",
        run_name="H-SCR",
        stage="architecture-screening",
        kind=JobKind.TRAIN,
        order=0,
        status=JobStatus.RUNNING,
        variant_path=str(variant_path),
        training_path=str(training_path),
        parent_checkpoint=str(checkpoint),
    )
    backend = ResearchQueueBackend(
        project_root=tmp_path,
        evaluation_backend=FakeEvaluation(),
        training_launcher=fake_launch,
    )

    result = backend.execute(job, _state(job, project_root=tmp_path))

    assert seen["request"].training.weights == str(checkpoint.resolve())
    assert result.checkpoint_path.endswith("h-scr/ultralytics/weights/best.pt")


def test_validate_dispatch_uses_injected_p0_runner(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.touch()
    metrics = tmp_path / "p0.json"
    metrics.write_text(json.dumps({"ok": True}), encoding="utf-8")
    expected = QueueResult(checkpoint_path=str(checkpoint), metrics_path=str(metrics))
    calls: list[str] = []

    def fake_p0(job, run_dir):
        calls.append(job.id)
        return expected

    job = QueueJob(
        id="p0",
        run_name="P0",
        stage="validation",
        kind=JobKind.VALIDATE,
        order=0,
        status=JobStatus.RUNNING,
        parent_checkpoint=str(checkpoint),
    )

    actual = ResearchQueueBackend(project_root=tmp_path, p0_runner=fake_p0).execute(
        job, _state(job, project_root=tmp_path)
    )

    assert actual == expected
    assert calls == ["p0"]


def test_p0_report_gates_attention_and_c2psa_not_decoded_postprocess() -> None:
    import torch

    required = {
        "model.10.m.0.attn": (torch.zeros(2), torch.tensor([0.0, 1e-6])),
        "model.10": (torch.zeros(2), torch.tensor([0.0, 2e-6])),
        "model.22.m.0.1.attn": (torch.zeros(2), torch.tensor([0.0, 1e-6])),
        "model.22": (torch.zeros(2), torch.tensor([0.0, 2e-6])),
    }

    report = build_p0_equivalence_report(
        required,
        final_pairs=((torch.zeros(2), torch.tensor([0.0, 1e-3])),),
        tolerance=1e-4,
    )

    assert report["passed"] is True
    assert report["required_max_abs_error"] == pytest.approx(2e-6)
    assert report["decoded_max_abs_error"] == pytest.approx(1e-3)
