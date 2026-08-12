"""Final integrity audit for unified retest post-processing."""

from __future__ import annotations

import json
from pathlib import Path

from .postprocess import ART


def audit() -> dict:
    metrics = list((ART / "evaluation").glob("*/**/metrics.json"))
    profiles = list((ART / "profiles").glob("*.json"))
    lineage = ART / "lineage" / "checkpoints.json"
    state = json.loads((ART / "queue_state.json").read_text(encoding="utf-8"))
    errors = []
    if len(metrics) != 28:
        errors.append(f"expected 28 evaluation metrics, found {len(metrics)}")
    if len(profiles) != 15:
        errors.append(f"expected 15 profile json files including summary, found {len(profiles)}")
    if not lineage.is_file():
        errors.append("missing checkpoint lineage")
    if state.get("status") != "formal_complete":
        errors.append("queue is not formal_complete")
    result = {"ok": not errors, "errors": errors, "metrics": len(metrics), "profiles": len(profiles)}
    (ART / "final_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
