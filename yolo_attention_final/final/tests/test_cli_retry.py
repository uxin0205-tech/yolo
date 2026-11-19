from __future__ import annotations

from pathlib import Path

from yolo_attention.cli import build_parser


def test_queue_retry_is_part_of_single_cli_surface() -> None:
    args = build_parser().parse_args(["queue", "retry", "pilot-1e-5"])
    assert args.command == "queue"
    assert args.queue_command == "retry"
    assert args.job_id == "pilot-1e-5"


def test_recovery_queue_has_an_independent_default_root() -> None:
    args = build_parser().parse_args(["queue", "init-pwl-recovery"])
    assert args.queue_command == "init-pwl-recovery"
    assert args.queue_root == Path("artifacts/recovery-queue")
