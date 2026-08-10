"""Operator CLI for MASF-YOLO Phase 1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from masf_yolo.contracts import Phase1Config


def load_config(path: Path) -> Phase1Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return Phase1Config.from_mapping(raw)


def ensure_tracked_clean(status_output: str) -> None:
    tracked = [line for line in status_output.splitlines() if line.strip() and not line.startswith("??")]
    if tracked:
        raise RuntimeError(f"tracked worktree must be clean before pipeline start: {tracked}")


def build_systemd_command(*, config_path: Path, unit: str, python: Path) -> list[str]:
    workdir = config_path.parent.parent
    return [
        "systemd-run",
        "--user",
        "--unit",
        unit,
        "--property=Restart=on-failure",
        "--property=RestartSec=30s",
        "--property=StartLimitBurst=3",
        "--property=StartLimitIntervalSec=1h",
        "--property=OOMPolicy=continue",
        "--property=MemoryAccounting=yes",
        "--collect",
        f"--working-directory={workdir}",
        str(python),
        "-m",
        "masf_yolo.cli",
        "pipeline",
        "execute",
        f"--config={config_path}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--config", type=Path, required=True)
    cleanup.add_argument("--apply", action="store_true")
    pipeline = commands.add_parser("pipeline")
    pipeline_commands = pipeline.add_subparsers(dest="pipeline_command", required=True)
    for name in ("start", "execute", "status"):
        command = pipeline_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    return parser


def _run_audit(config_path: Path) -> dict[str, object]:
    from masf_yolo.data.audit import audit_dataset

    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    values = config.values
    manifest = audit_dataset(
        root / values["dataset"]["source"],
        root / values["artifacts_root"] / "dataset",
        seed=values["dataset"]["seed"],
        minimum_ball_count=values["dataset"]["minimum_ball_count"],
    )
    return manifest.to_dict()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    if args.command == "audit":
        print(json.dumps(_run_audit(config_path), indent=2, sort_keys=True))
        return
    if args.command == "verify":
        from masf_yolo.runtime import verify_environment

        print(json.dumps(verify_environment(config_path, require_cuda=True).to_dict(), indent=2, sort_keys=True))
        return
    if args.command == "report":
        from masf_yolo.reporting import rebuild_report

        print(rebuild_report(config_path))
        return
    if args.command == "cleanup":
        from masf_yolo.cleanup import apply_cleanup, build_cleanup_plan
        from masf_yolo.runtime import query_systemd_unit_active

        config = load_config(config_path)
        repo_root = config_path.parent.parent
        artifact_root = repo_root / config.values["artifacts_root"]
        plan = build_cleanup_plan(
            repo_root,
            artifact_root,
            service_active=query_systemd_unit_active,
        )
        payload = (
            apply_cleanup(plan, artifact_root / "cleanup_manifest.json")
            if args.apply
            else plan.to_dict()
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    from masf_yolo.runtime import execute_pipeline, pipeline_start, pipeline_status

    if args.pipeline_command == "start":
        print(json.dumps(pipeline_start(config_path), indent=2, sort_keys=True))
    elif args.pipeline_command == "execute":
        execute_pipeline(config_path)
    else:
        print(json.dumps(pipeline_status(config_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
