"""Attention QueueExecutor：校準、smoke、單候選適應、剝除驗證、交付。"""

import gc
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed
from yolo_attention.queue_model import JobKind, QueueJob, QueueResult, QueueState
from yolo_attention.queue_policy import SelectionDecision
from yolo_attention.queue_store import QueueStore
from yolo_attention.stage3_kd_training import metric_vector

from . import checkpoint
from .common import (
    ARTIFACTS,
    PROJECT,
    ROOT,
    emit,
    read,
    review,
    source_guard,
    write_json,
)
from .models import strip, strip_proof


def initialize(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ARTIFACTS):
        raise ValueError("fresh Stage5 run directory required")
    steps = [
        ("calibration", JobKind.VALIDATE),
        ("smoke", JobKind.VALIDATE),
        ("training", JobKind.TRAIN),
        ("verification", JobKind.EVALUATE),
        ("selection", JobKind.SELECT),
    ]
    jobs = tuple(
        QueueJob(
            id=name,
            run_name=name,
            stage="integrated-stage5-hog",
            kind=kind,
            order=i,
            parent_job_ids=() if i == 0 else (steps[i - 1][0],),
            requires_gpu=kind is not JobKind.SELECT,
            evaluation_path=str(output / name),
        )
        for i, (name, kind) in enumerate(steps)
    )
    QueueStore(output).initialize(QueueState.initial(jobs, project_root=str(PROJECT)))
    source_guard(output, create=True)
    (output / "README.md").write_text(
        "# 第五階段 HOG\n\n單一候選，完整 COCO 適應最多十輪、至少五輪，patience 4；"
        "HOG E1 漸開、E2～E8 啟用、E9～E10 關閉。完成驗證與選型後停止，不啟動融合。\n"
    )


class Backend:
    def execute(self, job, state):
        from binaryqk.data import dataset
        from binaryqk.evaluate import evaluate
        from binaryqk.train import run

        from .diagnostics import calibrate
        from .training import HOGStudy

        folder = Path(job.evaluation_path)
        output = folder.parent
        source_guard(output)
        data = review()
        callback = lambda event: emit(output, {"job": job.id, **event})
        callback({"kind": "starting-job"})
        if job.kind is JobKind.SELECT:
            from .report import summarize

            summarize(output)
            callback({"kind": "completed", "stage": 5, "next_stage_started": False})
            return SelectionDecision(
                winners=("verification",),
                skipped=(),
                reason="第五階段完成，停止供使用者檢視",
            )
        require_idle_gpu()
        seed(data["protocol"]["seed"])
        try:
            if job.id == "calibration":
                folder.mkdir(exist_ok=False)
                calibrate(folder, callback)
                result = QueueResult(
                    checkpoint_path=data["parent"],
                    metrics_path=str(folder / "calibration.json"),
                )
            elif job.id in ("smoke", "training"):
                policy = HOGStudy(output / "calibration/calibration.json")
                run(
                    "H2",
                    folder,
                    ROOT / "yolo_binaryqk/artifacts/stage1/calibration.json",
                    phase="hog-adaptation",
                    parent=data["parent"],
                    smoke=job.id == "smoke",
                    resume=(folder / "resume.pt").exists(),
                    policy=policy,
                )
                summary = read(folder / "result.json")
                best = summary["best"]
                if job.id == "smoke":
                    payload = torch.load(
                        folder / "resume.pt", map_location="cpu", weights_only=True
                    )
                    initial = checkpoint.build(output / "calibration/calibration.json")
                    changed = float(
                        (
                            payload["live_state"]["hog_aux.projection.weight"]
                            - initial.hog_aux.projection.weight.detach()
                        )
                        .abs()
                        .max()
                    )
                    if changed <= 0:
                        raise ValueError("HOG smoke did not update auxiliary weights")
                    trained, _ = checkpoint.load(folder / best["checkpoint"])
                    proof = strip_proof(trained)
                    proof["hog_update_abs_max"] = changed
                    write_json(folder / "strip-proof.json", proof)
                    del payload, initial, trained
                result = QueueResult(
                    map50_95=best["AP50_95"],
                    checkpoint_path=str(folder / best["checkpoint"]),
                    metrics_path=str(folder / "result.json"),
                )
            elif job.id == "verification":
                folder.mkdir(exist_ok=False)
                summary = read(output / "training/result.json")
                best = summary["best"]
                model, _ = checkpoint.load(output / "training" / best["checkpoint"])
                write_json(folder / "strip-proof.json", strip_proof(model))
                deployment = strip(model)
                checkpoint.save(
                    folder / "deployment.pt",
                    deployment,
                    checkpoint.manifest_for(deployment),
                )
                restored, _ = checkpoint.load(folder / "deployment.pt")
                ds, data_yaml, _ = dataset("val", 8)
                metrics = evaluate(
                    restored.to("cuda:0"),
                    ds,
                    data_yaml,
                    folder / "evaluation",
                    progress=callback,
                )
                expected = metric_vector(
                    read(output / "training" / f"metrics-{best['epoch']:02d}.json")
                )
                if any(
                    abs(value - expected[key]) > 1e-8
                    for key, value in metric_vector(metrics).items()
                ):
                    raise ValueError("stripped deployment reload formal AP mismatch")
                write_json(folder / "metrics.json", metrics)
                result = QueueResult(
                    map50_95=metrics["official"]["AP50_95"],
                    checkpoint_path=str(folder / "deployment.pt"),
                    metrics_path=str(folder / "metrics.json"),
                )
                del model, deployment, restored, ds
            else:
                raise ValueError(job.id)
            callback({"kind": "job-complete"})
            return result
        finally:
            gc.collect()
            torch.cuda.empty_cache()
