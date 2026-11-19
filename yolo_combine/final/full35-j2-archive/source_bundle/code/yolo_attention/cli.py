"""單一命令入口；所有會改變狀態的操作都必須明確指定 --execute。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m yolo_attention.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--variant", type=Path, required=True)
    train.add_argument("--training", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    train.add_argument("--execute", action="store_true")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--model", default="yolo26m.yaml")
    smoke.add_argument("--variant", type=Path, default=Path("configs/variants/float-pwl-final.yaml"))
    smoke.add_argument("--imgsz", type=int, default=64)
    queue = commands.add_parser("queue")
    qcommands = queue.add_subparsers(dest="queue_command", required=True)
    for name in (
        "init-pwl-final",
        "init-pwl-recovery",
        "init-pwl-lr-sweep",
        "status",
        "validate",
        "next",
        "run-next",
        "run",
        "retry",
    ):
        item = qcommands.add_parser(name)
        default_root = Path(
            "artifacts/lr-sweep-queue"
            if name == "init-pwl-lr-sweep"
            else "artifacts/recovery-queue"
            if name == "init-pwl-recovery"
            else "artifacts/queue"
        )
        item.add_argument("--queue-root", type=Path, default=default_root)
        if name in {"init-pwl-final", "init-pwl-recovery", "init-pwl-lr-sweep"}:
            item.add_argument("--project-root", type=Path, default=Path.cwd())
        if name == "retry":
            item.add_argument("job_id")
        if name in {"run-next", "run"}:
            item.add_argument("--execute", action="store_true")
    return parser


def _validate_queue(state) -> dict[str, object]:
    from .config import VariantConfig
    from .run_config import TrainingRecipe

    errors: list[str] = []
    warnings = ["canonical COCO API 不是 gate；正式依據為 Ultralytics internal COCO metrics"]
    try:
        state.validate()
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    for job in state.jobs:
        if job.variant_path:
            try:
                VariantConfig.from_yaml(job.variant_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{job.id} 的 variant 無效：{exc}")
        if job.training_path:
            path = Path(job.training_path)
            if path.exists():
                try:
                    TrainingRecipe.from_yaml(path)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{job.id} 的訓練配方無效：{exc}")
            elif "generated" not in path.parts:
                errors.append(f"缺少 {job.id} 的訓練配方：{path}")
    root = Path(state.project_root)
    for required in (root / "weights/v1-br-best.pt", root / "data/coco2017.yaml"):
        if not required.is_file():
            errors.append(f"缺少必要輸入：{required}")
    if not (root / "artifacts/queue/generated").exists():
        warnings.append("seed 配方會在 pilot 選擇後以 immutable 方式產生")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "jobs": len(state.jobs)}


def _queue(args: argparse.Namespace) -> int:
    from .final_workflow import create_pwl_final_state
    from .lr_sweep_workflow import create_lr_sweep_state
    from .queue_executor import QueueExecutor
    from .queue_store import QueueStore
    from .recovery_workflow import create_pwl_recovery_state

    store = QueueStore(args.queue_root)
    if args.queue_command == "init-pwl-final":
        state = create_pwl_final_state(args.project_root)
        store.initialize(state)
        store.append_event("initialized_pwl_final", job_id=None, details={"jobs": len(state.jobs)})
        _print({"queue_root": str(store.root.resolve()), "jobs": len(state.jobs), "will_execute": False})
        return 0
    if args.queue_command == "init-pwl-recovery":
        state = create_pwl_recovery_state(args.project_root)
        store.initialize(state)
        store.append_event("initialized_pwl_recovery", job_id=None, details={"jobs": len(state.jobs)})
        _print({"queue_root": str(store.root.resolve()), "jobs": len(state.jobs), "will_execute": False})
        return 0
    if args.queue_command == "init-pwl-lr-sweep":
        state = create_lr_sweep_state(args.project_root)
        store.initialize(state)
        store.append_event("initialized_pwl_lr_sweep", job_id=None, details={"jobs": len(state.jobs)})
        _print({"queue_root": str(store.root.resolve()), "jobs": len(state.jobs), "will_execute": False})
        return 0
    state = store.load()
    if args.queue_command == "status":
        _print(
            {
                "revision": state.revision,
                "counts": dict(Counter(job.status.value for job in state.jobs)),
                "jobs": [
                    {"id": job.id, "kind": job.kind.value, "status": job.status.value} for job in state.jobs
                ],
            }
        )
        return 0
    if args.queue_command == "validate":
        report = _validate_queue(state)
        _print(report)
        return 0 if report["valid"] else 1
    if args.queue_command == "retry":

        class RetryBackend:
            def execute(self, *_args):
                raise RuntimeError("retry 不會直接執行工作")

        job = QueueExecutor(store, backend=RetryBackend()).retry(args.job_id)
        _print({"job_id": job.id, "status": job.status.value, "will_execute": False})
        return 0
    execute = bool(getattr(args, "execute", False))
    if execute:
        from .final_backend import FinalQueueBackend
        from .final_evaluation import PWLFinalEvaluationBackend

        backend = FinalQueueBackend(
            project_root=state.project_root, evaluation_backend=PWLFinalEvaluationBackend()
        )
    else:

        class DryBackend:
            def execute(self, *_args):
                raise RuntimeError("執行工作必須指定 --execute")

        backend = DryBackend()
    executor = QueueExecutor(store, backend=backend)
    if args.queue_command == "next":
        _print(executor.preview_next().to_dict())
    elif args.queue_command == "run-next":
        _print(executor.run_next(execute=execute).to_dict())
    else:
        previews = executor.run_all(execute=execute)
        _print({"previews": [preview.to_dict() for preview in previews], "will_execute": execute})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "queue":
        return _queue(args)
    if args.command == "train":
        from .runner import TrainingRequest, launch_training

        request = TrainingRequest.from_files(
            variant_path=args.variant,
            training_path=args.training,
            artifacts_root=args.artifacts_root,
            run_id=args.run_id,
        )
        if not args.execute:
            _print(request.summary())
            return 0
        launch_training(request)
        return 0
    if args.command == "smoke":
        import torch
        from ultralytics.nn.tasks import DetectionModel

        from .config import VariantConfig
        from .integration import convert_yolo26_model, fixed_scale_modules

        model = DetectionModel(args.model, ch=3, nc=80, verbose=False).eval()
        paths = convert_yolo26_model(model, VariantConfig.from_yaml(args.variant))
        sample = torch.randn(1, 3, args.imgsz, args.imgsz)
        pending = fixed_scale_modules(model)
        for attention in pending:
            attention.score.begin_calibration()
        with torch.no_grad():
            model(sample)
        for attention in pending:
            attention.score.finish_calibration()
        with torch.no_grad():
            output = model(sample)
        _print({"model": args.model, "nc": 80, "paths": paths, "output": type(output).__name__})
        return 0
    raise AssertionError("未處理的命令")


if __name__ == "__main__":
    raise SystemExit(main())
