"""Deterministic multi-seed job plan without launching any process."""

from __future__ import annotations

from typing import Any

from .contracts import CLEAN_EXPERIMENTS, CleanStudyConfig


def build_clean_plan(config: CleanStudyConfig) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in config.values["seeds"]:
        for name, spec in CLEAN_EXPERIMENTS.items():
            dependency = None
            if spec.parent_required:
                dependency = f"P2-Control-Clean-Head:seed{seed}"
            jobs.append({
                "job_id": f"{name}:seed{seed}",
                "experiment": name,
                "seed": seed,
                "comparison_tier": spec.comparison_tier,
                "schedule": spec.schedule,
                "depends_on": dependency,
                "status": "prepared_not_queued",
            })
    return jobs
