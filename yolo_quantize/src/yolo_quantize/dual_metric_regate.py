"""CPU-only dual mAP50/mAP50-95 re-gate for immutable search artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS
from .qat_metrics import Map50AccuracyGate


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


def _resolve(source: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def _raw_metrics(source: Path, record: Mapping[str, Any], label: str) -> dict[str, float]:
    path_key = (
        "source_metrics_report"
        if "source_metrics_report" in record
        else "metrics_report"
    )
    digest_key = f"{path_key}_sha256"
    path = _resolve(source, record.get(path_key, ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(record.get(digest_key, ""))
    actual = _sha256(path)
    if not expected or actual != expected:
        raise RuntimeError(f"{label} raw metrics SHA-256 differs")
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    values = _mapping(payload.get("metrics"), f"{label} metric values")
    required = FULL35_MAP50_KEYS + FULL35_MAP50_95_KEYS
    missing = tuple(name for name in required if name not in values)
    if missing:
        raise RuntimeError(f"{label} omits dual-gate metrics: {missing}")
    return {name: float(values[name]) for name in required}


def _reference_roles(
    source_path: Path,
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(source.get("reference"), dict):
        return _mapping(source["reference"].get("roles"), "reference roles")
    if isinstance(source.get("roles"), dict):
        return _mapping(source.get("roles"), "source roles")
    contract = source.get("contract")
    if isinstance(contract, dict):
        reuse = contract.get("reference_reuse")
        if isinstance(reuse, dict):
            reference_path = _resolve(source_path, reuse.get("report", ""))
            if reference_path == source_path:
                raise RuntimeError("reference report cannot point to itself")
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            expected = str(reuse.get("sha256", ""))
            actual = _sha256(reference_path)
            if not expected or actual != expected:
                raise RuntimeError("reference report SHA-256 differs")
            reference = _mapping(
                json.loads(reference_path.read_text(encoding="utf-8")),
                "reference report",
            )
            if reference.get("status") != "completed":
                raise RuntimeError("reference report is incomplete")
            return _reference_roles(reference_path, reference)
    raise TypeError("source report has no reference roles")


def _candidate_records(source: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source.get("results"), dict):
        return _mapping(source["results"], "candidate results")
    roles = _mapping(source.get("roles"), "source roles")
    return {"candidate": _mapping(roles.get("candidate"), "candidate role")}


def _worst(deltas: Mapping[str, float], keys: tuple[str, ...]) -> tuple[str, float]:
    return min(((name, float(deltas[name])) for name in keys), key=lambda item: item[1])


def _metric_summary(metrics: Mapping[str, float]) -> dict[str, float]:
    """Return transparent cross-task summaries without calling them official mAP."""

    return {
        "mean_required_map50": sum(metrics[name] for name in FULL35_MAP50_KEYS)
        / len(FULL35_MAP50_KEYS),
        "mean_required_map50_95": sum(
            metrics[name] for name in FULL35_MAP50_95_KEYS
        )
        / len(FULL35_MAP50_95_KEYS),
        "joint_priority_map50": (
            0.2 * metrics["coco/box/map50"]
            + 0.2 * metrics["coco/person/box/map50"]
            + 0.2 * metrics["bbat/box/map50"]
            + 0.4 * metrics["bbat/pose/map50"]
        ),
        "joint_priority_map50_95": (
            0.2 * metrics["coco/box/map50_95"]
            + 0.2 * metrics["coco/person/box/map50_95"]
            + 0.2 * metrics["bbat/box/map50_95"]
            + 0.4 * metrics["bbat/pose/map50_95"]
        ),
    }


def _summary_deltas(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
) -> dict[str, float]:
    candidate_summary = _metric_summary(candidate)
    reference_summary = _metric_summary(reference)
    return {
        name: candidate_summary[name] - reference_summary[name]
        for name in candidate_summary
    }


def regate_candidate_report(
    *,
    source_report: str | Path,
    expected_source_sha256: str,
    metric_contract_id: str,
    map50_max_drop: float = 0.015,
    map50_95_max_drop: float = 0.04,
    map50_95_recovery_floor: float = 0.08,
) -> dict[str, Any]:
    """Re-read hash-pinned raw metrics and add a dual-family decision."""

    path = Path(source_report).expanduser().resolve()
    actual = _sha256(path)
    if actual != expected_source_sha256:
        raise RuntimeError("source report SHA-256 does not match reviewed contract")
    if not metric_contract_id.strip():
        raise ValueError("metric_contract_id must not be empty")
    if map50_95_recovery_floor < map50_95_max_drop:
        raise ValueError("mAP50-95 recovery floor cannot be stricter than its gate")
    source = _mapping(json.loads(path.read_text(encoding="utf-8")), "source report")
    if source.get("status") != "completed":
        raise RuntimeError("source report is incomplete")
    roles = _reference_roles(path, source)
    accepted = _raw_metrics(path, _mapping(roles.get("accepted"), "accepted"), "accepted")
    matched = _raw_metrics(path, _mapping(roles.get("matched"), "matched"), "matched")
    evaluator = Map50AccuracyGate(
        accepted,
        maximum_drop=map50_max_drop,
        maximum_map50_95_drop=map50_95_max_drop,
    )
    results: dict[str, Any] = {}
    for candidate_id, raw_record in _candidate_records(source).items():
        record = _mapping(raw_record, f"candidate {candidate_id}")
        if record.get("status") != "completed":
            results[str(candidate_id)] = {
                "status": str(record.get("status", "unknown")),
                "decision": "not_evaluated",
            }
            continue
        metrics = _raw_metrics(path, record, f"candidate {candidate_id}")
        gate = evaluator.evaluate(metrics)
        source_gate = _mapping(record.get("gate"), f"candidate {candidate_id} gate")
        source_passed = bool(source_gate.get("passed"))
        if source_passed and gate.primary_failed_metrics:
            raise RuntimeError(
                f"candidate {candidate_id} saved a green mAP50 gate despite raw failure"
            )
        incremental = {name: metrics[name] - matched[name] for name in metrics}
        worst_map50_metric, worst_map50_delta = _worst(
            gate.deltas, FULL35_MAP50_KEYS
        )
        worst_quality_metric, worst_quality_delta = _worst(
            gate.deltas, FULL35_MAP50_95_KEYS
        )
        worst_inc50_metric, worst_inc50_delta = _worst(
            incremental, FULL35_MAP50_KEYS
        )
        worst_inc_quality_metric, worst_inc_quality_delta = _worst(
            incremental, FULL35_MAP50_95_KEYS
        )
        dual_passed = source_passed and not gate.companion_failed_metrics
        if dual_passed:
            decision = "green"
        elif source_passed and worst_quality_delta >= -map50_95_recovery_floor:
            decision = "recover"
        else:
            decision = str(source_gate.get("decision", "reject"))
            if decision == "green":
                decision = "reject"
        results[str(candidate_id)] = {
            "status": "completed",
            "source_map50_gate_passed": source_passed,
            "source_map50_decision": str(source_gate.get("decision", "unknown")),
            "dual_gate_passed": dual_passed,
            "decision": decision,
            "total_deltas": gate.deltas,
            "incremental_deltas": incremental,
            "failed_map50_metrics": list(gate.primary_failed_metrics),
            "failed_map50_95_metrics": list(gate.companion_failed_metrics),
            "worst_map50_metric": worst_map50_metric,
            "worst_map50_delta": worst_map50_delta,
            "worst_map50_95_metric": worst_quality_metric,
            "worst_map50_95_delta": worst_quality_delta,
            "worst_incremental_map50_metric": worst_inc50_metric,
            "worst_incremental_map50_delta": worst_inc50_delta,
            "worst_incremental_map50_95_metric": worst_inc_quality_metric,
            "worst_incremental_map50_95_delta": worst_inc_quality_delta,
            "metric_summaries": {
                "semantics": {
                    "mean_required": "diagnostic_mean_not_official_map",
                    "joint_priority": (
                        "fixed_0.2_0.2_0.2_0.4_selection_score_not_official_map"
                    ),
                },
                "accepted": _metric_summary(accepted),
                "matched": _metric_summary(matched),
                "candidate": _metric_summary(metrics),
                "total_deltas": _summary_deltas(metrics, accepted),
                "incremental_deltas": _summary_deltas(metrics, matched),
            },
        }
    return {
        "schema_version": 1,
        "kind": "full35_dual_metric_candidate_regate",
        "status": "completed",
        "gpu_used": False,
        "validation_rerun": False,
        "formal_training": False,
        "formal_validation": False,
        "metric_contract_id": metric_contract_id,
        "thresholds": {
            "map50_total_max_drop": map50_max_drop,
            "map50_95_total_max_drop": map50_95_max_drop,
            "map50_95_recovery_floor": map50_95_recovery_floor,
            "activation_replacement_included": True,
            "all_eight_metrics_per_family_required": True,
        },
        "source": {"report": str(path), "report_sha256": actual},
        "candidates": results,
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
    parser = argparse.ArgumentParser(description="Re-gate saved search metrics as mAP50 plus mAP50-95")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--metric-contract-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = regate_candidate_report(
        source_report=args.source,
        expected_source_sha256=args.source_sha256,
        metric_contract_id=args.metric_contract_id,
    )
    _atomic_json(args.output.resolve(), payload)
    summary = {
        identifier: record["decision"]
        for identifier, record in payload["candidates"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("regate_candidate_report",)
