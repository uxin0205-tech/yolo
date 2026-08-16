"""CPU-safe command line entry points for configuration and smoke checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .config import VariantConfig
from .experiments import ExperimentRegistry, Stage
from .profiling import AttentionShape, estimate_operations
from .workflow import ResearchWorkflow


def _add_queue_parser(subparsers) -> None:
    queue = subparsers.add_parser("queue", help="Manage the single-worker experiment queue")
    commands = queue.add_subparsers(dest="queue_command", required=True)
    for name in ("init", "status", "validate", "next", "run-next", "run"):
        command = commands.add_parser(name)
        command.add_argument("--queue-root", type=Path, default=Path("artifacts/queue"))
        command.add_argument("--json", action="store_true")
        if name == "init":
            command.add_argument("--project-root", type=Path, default=Path.cwd())
        if name in {"run-next", "run"}:
            command.add_argument("--execute", action="store_true")
    retry = commands.add_parser("retry")
    retry.add_argument("job_id")
    retry.add_argument("--queue-root", type=Path, default=Path("artifacts/queue"))
    retry.add_argument("--json", action="store_true")
    rewind = commands.add_parser("rewind")
    rewind.add_argument("selection_job_id")
    rewind.add_argument("--queue-root", type=Path, default=Path("artifacts/queue"))
    rewind.add_argument("--json", action="store_true")
    append_bdcn = commands.add_parser("append-bdcn-v2")
    append_bdcn.add_argument("--queue-root", type=Path, default=Path("artifacts/queue"))
    append_bdcn.add_argument("--json", action="store_true")
    append_bdcn_v3 = commands.add_parser("append-bdcn-v3")
    append_bdcn_v3.add_argument("--queue-root", type=Path, default=Path("artifacts/queue"))
    append_bdcn_v3.add_argument("--json", action="store_true")


class _ExecutionDisabledBackend:
    def execute(self, job, state):
        raise RuntimeError("queue execution backend is only constructed with --execute")


def _print_payload(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def _run_queue_command(args) -> int:
    from .queue_executor import QueueExecutor
    from .queue_store import QueueStore
    from .queue_workflow import create_initial_state, validate_queue_environment

    store = QueueStore(args.queue_root)
    try:
        if args.queue_command == "init":
            state = create_initial_state(args.project_root)
            store.initialize(state)
            store.append_event("initialized", job_id=None, details={"jobs": len(state.jobs)})
            _print_payload(
                {"queue_root": str(store.root.resolve()), "jobs": len(state.jobs)}, as_json=args.json
            )
            return 0
        if args.queue_command == "status":
            state = store.load()
            counts = Counter(job.status.value for job in state.jobs)
            payload = {
                "queue_root": str(store.root.resolve()),
                "revision": state.revision,
                "counts": dict(sorted(counts.items())),
                "jobs": [
                    {"id": job.id, "status": job.status.value, "kind": job.kind.value, "order": job.order}
                    for job in state.jobs
                ],
            }
            _print_payload(payload, as_json=args.json)
            return 0
        if args.queue_command == "validate":
            report = validate_queue_environment(store.load())
            _print_payload(report, as_json=args.json)
            return 0 if report["valid"] else 1
        execute = bool(getattr(args, "execute", False))
        if execute:
            from .queue_backend import ResearchQueueBackend

            backend = ResearchQueueBackend(project_root=store.load().project_root)
        else:
            backend = _ExecutionDisabledBackend()
        executor = QueueExecutor(store, backend=backend)
        if args.queue_command == "next":
            _print_payload(executor.preview_next().to_dict(), as_json=args.json)
            return 0
        if args.queue_command == "retry":
            job = executor.retry(args.job_id)
            _print_payload({"job_id": job.id, "status": job.status.value}, as_json=args.json)
            return 0
        if args.queue_command == "rewind":
            job = executor.rewind_selection(args.selection_job_id)
            _print_payload({"job_id": job.id, "status": job.status.value}, as_json=args.json)
            return 0
        if args.queue_command == "append-bdcn-v2":
            state = executor.append_bdcn_v2()
            _print_payload(
                {"jobs": len(state.jobs), "next": executor.preview_next().job_id},
                as_json=args.json,
            )
            return 0
        if args.queue_command == "append-bdcn-v3":
            state = executor.append_bdcn_v3()
            _print_payload(
                {"jobs": len(state.jobs), "next": executor.preview_next().job_id},
                as_json=args.json,
            )
            return 0
        if args.queue_command == "run-next":
            payload = executor.run_next(execute=execute).to_dict()
        else:
            previews = executor.run_all(execute=execute)
            payload = (
                previews[-1].to_dict()
                if previews
                else {
                    "job_id": None,
                    "run_name": None,
                    "kind": None,
                    "requires_gpu": False,
                    "will_execute": execute,
                }
            )
        _print_payload(payload, as_json=args.json)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI converts all command failures to a stable exit code.
        _print_payload({"error": str(exc)}, as_json=getattr(args, "json", False))
        return 1


def _registry_payload(registry: ExperimentRegistry) -> dict[str, object]:
    return {
        "screening": [run.variant.name for run in registry.for_stage(Stage.SCREENING)],
        "recovery": [run.variant.name for run in registry.for_stage(Stage.RECOVERY)],
        "conditional": [name for name in registry.names() if registry.get(name).conditional],
        "all": list(registry.names()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo26-attention")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="List the approved experiment registry")
    listing.add_argument("--json", action="store_true")

    workflow = subparsers.add_parser("workflow", help="Show the main funnel and deferred quantization")
    workflow.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate-config", help="Parse and validate a variant YAML")
    validate.add_argument("path", type=Path)

    profile = subparsers.add_parser("profile", help="Print analytical attention operation counts")
    profile.add_argument("--tokens", type=int, default=400)
    profile.add_argument("--heads", type=int, default=4)
    profile.add_argument("--key-dim", type=int, default=32)
    profile.add_argument("--value-dim", type=int, default=64)

    smoke = subparsers.add_parser("smoke", help="Build and forward a converted YOLO26 model on CPU")
    smoke.add_argument("--model", default="yolo26n.yaml")
    smoke.add_argument("--variant", type=Path, required=True)
    smoke.add_argument("--imgsz", type=int, default=64)

    train = subparsers.add_parser("train", help="Preview a run; add --execute to start training")
    train.add_argument("--variant", type=Path, required=True)
    train.add_argument("--training", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    train.add_argument("--execute", action="store_true")
    _add_queue_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "queue":
        return _run_queue_command(args)
    if args.command == "list":
        payload = _registry_payload(ExperimentRegistry.default())
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            for stage in ("screening", "recovery", "conditional"):
                print(f"{stage}: {', '.join(payload[stage])}")
        return 0
    if args.command == "workflow":
        payload = ResearchWorkflow.default().to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            for group in ("main", "optional"):
                print(f"{group}:")
                for step in payload[group]:
                    print(f"  {step['key']}: {', '.join(step['runs'])} ({step['epochs']} ep)")
        return 0
    if args.command == "validate-config":
        config = VariantConfig.from_yaml(args.path)
        print(json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "profile":
        report = estimate_operations(
            AttentionShape(
                tokens=args.tokens,
                heads=args.heads,
                key_dim=args.key_dim,
                value_dim=args.value_dim,
            )
        )
        print(json.dumps(report.to_dict(), sort_keys=True))
        return 0
    if args.command == "smoke":
        import torch
        from ultralytics.nn.tasks import DetectionModel

        from .integration import convert_yolo26_model

        config = VariantConfig.from_yaml(args.variant)
        model = DetectionModel(args.model, ch=3, nc=80, verbose=False).eval()
        paths = convert_yolo26_model(model, config)
        with torch.no_grad():
            output = model(torch.randn(1, 3, args.imgsz, args.imgsz))
        summary = {
            "variant": config.name,
            "converted_paths": paths,
            "output_type": type(output).__name__,
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "train":
        from .runner import TrainingRequest, launch_training

        request = TrainingRequest.from_files(
            variant_path=args.variant,
            training_path=args.training,
            artifacts_root=args.artifacts_root,
            run_id=args.run_id,
        )
        if not args.execute:
            print(json.dumps(request.summary(), ensure_ascii=False, sort_keys=True))
            return 0
        launch_training(request)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
