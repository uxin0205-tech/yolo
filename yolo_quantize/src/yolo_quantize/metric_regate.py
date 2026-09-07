"""Re-gate immutable Full35 validation evidence under a new metric contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metric_gate import (
    Full35MetricCandidate,
    Full35MetricGate,
    Full35MetricGateSpec,
    Full35MetricSnapshot,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _verified_role_metrics(
    source_report: Path,
    roles: Mapping[str, Any],
    role: str,
    spec: Full35MetricGateSpec,
) -> tuple[Mapping[str, Any], dict[str, float], Path, str]:
    record = _mapping(roles.get(role), f"{role} role")
    if record.get("status") != "completed":
        raise RuntimeError(f"{role} source role is incomplete")
    raw_path = Path(str(record.get("metrics_report", ""))).expanduser()
    if not raw_path.is_absolute():
        raw_path = source_report.parent / raw_path
    raw_path = raw_path.resolve()
    expected = str(record.get("metrics_report_sha256", ""))
    actual = _sha256(raw_path)
    if not expected or actual != expected:
        raise RuntimeError(f"{role} metrics SHA-256 does not match source record")
    raw = _mapping(json.loads(raw_path.read_text(encoding="utf-8")), "raw metrics")
    values = _mapping(raw.get("metrics"), "raw metrics values")
    missing = tuple(key for key in spec.metric_keys if key not in values)
    if missing:
        raise RuntimeError(
            f"{role} raw metrics omit {spec.metric_family}: " + ", ".join(missing)
        )
    metrics = {key: float(values[key]) for key in spec.metric_keys}
    return record, metrics, raw_path, actual


def regate_search_report(
    *,
    source_report: str | Path,
    expected_source_sha256: str,
    metric_contract_id: str,
    spec: Full35MetricGateSpec | None = None,
) -> dict[str, Any]:
    """Verify and re-evaluate saved raw metrics without validation or GPU work."""

    path = Path(source_report).expanduser().resolve()
    actual_source_sha256 = _sha256(path)
    if actual_source_sha256 != expected_source_sha256:
        raise RuntimeError("source report SHA-256 does not match reviewed contract")
    if not metric_contract_id.strip():
        raise ValueError("metric_contract_id must not be empty")
    active_spec = (
        Full35MetricGateSpec(
            metric_family="map50",
            total_max_drop=0.015,
            w8_incremental_max_drop=0.01,
            sham_max_absolute_drift=0.01,
            recovery_floor=0.04,
        )
        if spec is None
        else spec
    )
    if active_spec.metric_family != "map50":
        raise ValueError("active re-gate contract must use map50")

    source = _mapping(json.loads(path.read_text(encoding="utf-8")), "source report")
    if source.get("status") != "completed":
        raise RuntimeError("source report is incomplete")
    source_contract = _mapping(source.get("contract"), "source contract")
    roles = _mapping(source.get("roles"), "source roles")

    extracted: dict[str, dict[str, Any]] = {}
    source_records: dict[str, Mapping[str, Any]] = {}
    for role in ("accepted", "matched", "candidate"):
        record, metrics, raw_path, raw_sha256 = _verified_role_metrics(
            path,
            roles,
            role,
            active_spec,
        )
        source_records[role] = record
        extracted[role] = {
            "policy_id": str(record.get("policy_id", "")),
            "metrics": metrics,
            "source_metrics_report": str(raw_path),
            "source_metrics_report_sha256": raw_sha256,
        }

    accepted_policy = extracted["accepted"]["policy_id"]
    matched_policy = extracted["matched"]["policy_id"]
    candidate_policy = extracted["candidate"]["policy_id"]
    if not accepted_policy or not matched_policy or candidate_policy != matched_policy:
        raise RuntimeError("source report does not preserve a matched policy triplet")
    weight = _mapping(
        source_records["candidate"].get("weight_quantization"),
        "candidate weight quantization",
    )
    format_id = str(weight.get("format_id", "")).strip()
    if not format_id:
        raise RuntimeError("candidate source role has no weight format identity")

    source_plan_id = str(source_contract.get("plan_id", "legacy-search"))
    accepted = Full35MetricSnapshot(
        run_id=f"{source_plan_id}:accepted:regate-map50",
        policy_id=accepted_policy,
        metric_contract_id=metric_contract_id,
        metrics=extracted["accepted"]["metrics"],
    )
    matched = Full35MetricSnapshot(
        run_id=f"{source_plan_id}:matched:regate-map50",
        policy_id=matched_policy,
        metric_contract_id=metric_contract_id,
        metrics=extracted["matched"]["metrics"],
    )
    candidate = Full35MetricCandidate(
        run_id=f"{source_plan_id}:candidate:regate-map50",
        format_id=format_id,
        stage="ptq",
        policy_id=candidate_policy,
        metric_contract_id=metric_contract_id,
        metrics=extracted["candidate"]["metrics"],
    )
    gate = Full35MetricGate(active_spec).evaluate(candidate, accepted, matched)
    return {
        "schema_version": 1,
        "kind": "full35_search_metric_regate",
        "status": "completed",
        "gpu_used": False,
        "validation_rerun": False,
        "formal_training": False,
        "formal_validation": False,
        "metric_contract_id": metric_contract_id,
        "source": {
            "report": str(path),
            "report_sha256": actual_source_sha256,
            "plan_id": source_plan_id,
        },
        "roles": extracted,
        "gate": gate.to_dict(),
        "selection_claim": "single_cell_gate_only",
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-gate saved Full35 metrics as mAP50"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--metric-contract-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = regate_search_report(
        source_report=args.source,
        expected_source_sha256=args.source_sha256,
        metric_contract_id=args.metric_contract_id,
    )
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload["gate"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
