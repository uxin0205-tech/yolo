"""Validation for the immutable attention-parent study provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import file_sha256


def load_parent_study(study_path: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a completed parent study and verify its formal winner checkpoint."""
    path = Path(study_path)
    checkpoint = Path(checkpoint_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    queue = payload.get("queue", {})
    succeeded = int(queue.get("jobs_succeeded", -1))
    total = int(queue.get("jobs_total", -1))
    if total <= 0 or succeeded != total:
        raise ValueError(f"parent training queue is incomplete: {succeeded}/{total}")

    formal = payload.get("formal_winner", {})
    expected_sha256 = formal.get("sha256")
    actual_sha256 = file_sha256(checkpoint)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "parent formal winner SHA256 does not match inputs/parent/best.pt: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    best_observed = payload.get("best_observed", {})
    if best_observed.get("included_in_final") is not False:
        raise ValueError("parent best-observed checkpoint must not be marked as the formal winner")

    return {
        "completed": True,
        "completed_at": payload.get("completed_at"),
        "queue": {
            "revision": int(queue.get("revision", -1)),
            "succeeded": succeeded,
            "total": total,
        },
        "formal_winner": {
            "selected": True,
            "sha256": actual_sha256,
            "map50_95": formal.get("map50_95"),
            "reason": formal.get("reason"),
        },
        "best_observed": {
            "selected": False,
            "sha256": best_observed.get("sha256"),
            "map50_95": best_observed.get("map50_95"),
            "delta_vs_formal_winner": best_observed.get("delta_vs_formal_winner"),
        },
    }
