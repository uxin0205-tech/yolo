from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full35_queue as queue_module
from full35_queue import _command, _job_metrics, load_queue

QUEUE = PROJECT_ROOT / "training/full35/finalist-seed1-queue.yaml"


def test_finalist_seed1_queue_is_matched_serial_and_uses_best_joint() -> None:
    config = load_queue(QUEUE)

    assert [job.activation for job in config.jobs] == ["silu", "qsilu_pq"]
    assert all(job.phase == "finalist_seed1" for job in config.jobs)
    assert all(job.metric_selection == "best_joint" for job in config.jobs)
    assert config.jobs[1].depends_on == (config.jobs[0].job_id,)
    assert config.jobs[1].detect_microbatch == 16
    command = _command(config.jobs[1], device="0")
    assert command[command.index("--phase") + 1] == "finalist_seed1"


def test_best_joint_metrics_require_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue_module, "RUN_ROOT", tmp_path)
    job = replace(load_queue(QUEUE).jobs[0], run_name="candidate")
    run_dir = tmp_path / job.run_name
    best_metrics_path = run_dir / "validation/epoch-0001/bittrue/metrics.json"
    best_metrics_path.parent.mkdir(parents=True)
    best_metrics_path.write_text(
        json.dumps({"metrics": {"coco/box/map50_95": 0.5}}),
        encoding="utf-8",
    )
    other_metrics_path = run_dir / "validation/epoch-0002/bittrue/metrics.json"
    other_metrics_path.parent.mkdir(parents=True)
    other_metrics_path.write_text(
        json.dumps({"metrics": {"coco/box/map50_95": 0.4}}),
        encoding="utf-8",
    )
    events_path = run_dir / "logs/events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        json.dumps(
            {
                "kind": "gate",
                "step": 1,
                "values": {"score/best_joint": 0.7, "passed": 0.0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "gate",
                "step": 2,
                "values": {"score/best_joint": 0.6, "passed": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _job_metrics(job) is None
    (run_dir / "activation-experiment.json").write_text("{}\n", encoding="utf-8")
    assert _job_metrics(job) == {"coco/box/map50_95": 0.4}


def test_best_joint_metrics_fall_back_to_best_score_when_all_gates_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue_module, "RUN_ROOT", tmp_path)
    job = replace(load_queue(QUEUE).jobs[0], run_name="all-failed")
    run_dir = tmp_path / job.run_name
    for step, metric in ((1, 0.5), (2, 0.4)):
        metrics_path = run_dir / f"validation/epoch-{step:04d}/bittrue/metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"metrics": {"coco/box/map50_95": metric}}),
            encoding="utf-8",
        )
    events_path = run_dir / "logs/events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "gate",
                    "step": step,
                    "values": {"score/best_joint": score, "passed": 0.0},
                }
            )
            for step, score in ((1, 0.7), (2, 0.6))
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "activation-experiment.json").write_text("{}\n", encoding="utf-8")

    assert _job_metrics(job) == {"coco/box/map50_95": 0.5}
