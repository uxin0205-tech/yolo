from __future__ import annotations

from pathlib import Path

from yolo_attention.queue_backend import ResearchQueueBackend
from yolo_attention.queue_model import JobKind, QueueJob, QueueResult, QueueState

ROOT = Path(__file__).resolve().parents[1]


def test_queue_backend_dispatches_pwl_validation_to_dedicated_runner(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    variant = tmp_path / "exact.yaml"
    variant.write_text((ROOT / "PWL/configs/exact.yaml").read_text())
    evaluation = tmp_path / "eval.yaml"
    evaluation.write_text((ROOT / "configs/evaluation/coco2017.yaml").read_text())
    calls: list[str] = []

    class FakePWLRunner:
        def score_analysis(self, request):
            calls.append(request.run_id)
            metrics = request.run_dir / "metrics.json"
            metrics.parent.mkdir(parents=True, exist_ok=True)
            metrics.write_text("{}")
            return QueueResult(checkpoint_path=str(checkpoint), metrics_path=str(metrics))

        def compare(self, request, *, score_run_dir):
            raise AssertionError("comparison was not requested")

    job = QueueJob(
        id="pwl-score-analysis",
        run_name="PWL-SCORE-ANALYSIS",
        stage="pwl",
        kind=JobKind.VALIDATE,
        order=0,
        variant_path=str(variant),
        evaluation_path=str(evaluation),
        parent_checkpoint=str(checkpoint),
    )
    state = QueueState.initial((job,), project_root=str(tmp_path))

    result = ResearchQueueBackend(project_root=tmp_path, pwl_runner=FakePWLRunner()).execute(job, state)

    assert calls == ["pwl-score-analysis"]
    assert result.checkpoint_path == str(checkpoint)
