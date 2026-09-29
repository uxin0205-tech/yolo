from __future__ import annotations

import json
from pathlib import Path

import pytest

from masf_yolo.cleanup import apply_cleanup, build_cleanup_plan


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_bytes(path: Path, value: bytes = b"checkpoint") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _seed_completed_repo(root: Path) -> Path:
    artifact_root = root / "artifacts" / "static-phase1"
    _write_json(
        artifact_root / "pipeline.json",
        {"pipeline_id": "pipeline-123", "unit": "masf-yolo-phase1.service"},
    )
    _write_json(artifact_root / "stages" / "report.json", {"status": "completed"})
    _write_json(artifact_root / "final_audit.json", {"ok": True, "errors": []})

    _write_bytes(artifact_root / "runs" / "m2" / "weights" / "best.pt", b"formal-best")
    _write_bytes(artifact_root / "runs" / "m2" / "weights" / "last.pt", b"formal-last")
    _write_bytes(artifact_root / "training" / "formal_m2" / "canonical.pt", b"formal-canonical")
    _write_bytes(artifact_root / "evaluation" / "test" / "m2" / "metrics.json", b"metrics")
    _write_bytes(artifact_root / "profiles" / "m2" / "profile.json", b"profile")

    _write_bytes(artifact_root / "smoke_runs" / "m2-smoke" / "weights" / "best.pt", b"smoke-best")
    _write_bytes(artifact_root / "smoke_runs" / "m2-smoke" / "weights" / "last.pt", b"smoke-last")
    _write_bytes(artifact_root / "training" / "smoke_m2" / "canonical.pt", b"smoke-canonical")
    _write_bytes(artifact_root / "preflight" / "m2.pt", b"preflight")
    _write_bytes(artifact_root / "m7_gate" / "m7.pt", b"gate")
    _write_bytes(artifact_root / "runs" / "m2" / "train_batch0.jpg", b"preview")
    _write_bytes(artifact_root / "runs" / "m2" / "labels.jpg", b"labels")
    _write_bytes(artifact_root / "smoke_runs" / "m2-smoke" / "train_batch0.jpg", b"smoke-preview")
    _write_bytes(artifact_root / "smoke_runs" / "m2-smoke" / "labels.jpg", b"smoke-labels")
    _write_bytes(root / ".pytest_cache" / "v" / "cache" / "nodeids", b"cache")
    _write_bytes(root / "masf_yolo" / "__pycache__" / "module.pyc", b"bytecode")
    _write_bytes(root / "yolo26n.pt", b"unused-model")
    return artifact_root


def test_cleanup_plan_selects_only_disposable_artifacts(tmp_path: Path) -> None:
    artifact_root = _seed_completed_repo(tmp_path)

    plan = build_cleanup_plan(tmp_path, artifact_root, service_active=lambda _unit: False)

    selected = {target.relative_path for target in plan.targets}
    assert "artifacts/static-phase1/smoke_runs/m2-smoke/weights/best.pt" in selected
    assert "artifacts/static-phase1/training/smoke_m2/canonical.pt" in selected
    assert "artifacts/static-phase1/preflight/m2.pt" in selected
    assert "artifacts/static-phase1/m7_gate/m7.pt" in selected
    assert "artifacts/static-phase1/runs/m2/train_batch0.jpg" in selected
    assert ".pytest_cache/v/cache/nodeids" in selected
    assert "masf_yolo/__pycache__/module.pyc" in selected
    assert "yolo26n.pt" in selected
    assert "artifacts/static-phase1/runs/m2/weights/best.pt" not in selected
    assert "artifacts/static-phase1/runs/m2/weights/last.pt" not in selected
    assert "artifacts/static-phase1/training/formal_m2/canonical.pt" not in selected
    assert "artifacts/static-phase1/evaluation/test/m2/metrics.json" not in selected
    assert plan.pipeline_id == "pipeline-123"
    assert set(plan.preserved_checkpoint_hashes) == {
        "artifacts/static-phase1/runs/m2/weights/best.pt",
        "artifacts/static-phase1/runs/m2/weights/last.pt",
        "artifacts/static-phase1/training/formal_m2/canonical.pt",
    }


def test_cleanup_plan_refuses_active_service(tmp_path: Path) -> None:
    artifact_root = _seed_completed_repo(tmp_path)

    with pytest.raises(RuntimeError, match="service is active"):
        build_cleanup_plan(tmp_path, artifact_root, service_active=lambda _unit: True)


@pytest.mark.parametrize(
    ("report_status", "audit", "message"),
    [
        ("running", {"ok": True, "errors": []}, "report stage is not completed"),
        ("completed", {"ok": False, "errors": ["broken"]}, "final audit did not pass"),
    ],
)
def test_cleanup_plan_requires_completed_audited_pipeline(
    tmp_path: Path,
    report_status: str,
    audit: dict[str, object],
    message: str,
) -> None:
    artifact_root = _seed_completed_repo(tmp_path)
    _write_json(artifact_root / "stages" / "report.json", {"status": report_status})
    _write_json(artifact_root / "final_audit.json", audit)

    with pytest.raises(RuntimeError, match=message):
        build_cleanup_plan(tmp_path, artifact_root, service_active=lambda _unit: False)


def test_cleanup_plan_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    artifact_root = _seed_completed_repo(root)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    link = artifact_root / "preflight" / "outside.pt"
    link.symlink_to(outside)

    with pytest.raises(RuntimeError, match="outside repo root"):
        build_cleanup_plan(root, artifact_root, service_active=lambda _unit: False)


def test_runtime_reference_preserves_yolo26n(tmp_path: Path) -> None:
    artifact_root = _seed_completed_repo(tmp_path)
    _write_bytes(tmp_path / "configs" / "model.yaml", b"model: yolo26n.pt\n")

    plan = build_cleanup_plan(tmp_path, artifact_root, service_active=lambda _unit: False)

    assert "yolo26n.pt" not in {target.relative_path for target in plan.targets}


def test_apply_cleanup_writes_manifest_and_preserves_formal_hashes(tmp_path: Path) -> None:
    artifact_root = _seed_completed_repo(tmp_path)
    plan = build_cleanup_plan(tmp_path, artifact_root, service_active=lambda _unit: False)
    manifest_path = artifact_root / "cleanup_manifest.json"
    before = dict(plan.preserved_checkpoint_hashes)

    result = apply_cleanup(plan, manifest_path)

    assert manifest_path.is_file()
    assert result["total_deleted_bytes"] == result["total_planned_bytes"]
    assert {target["status"] for target in result["targets"]} == {"deleted"}
    assert result["preserved_checkpoint_hashes"] == before
    assert all(not (tmp_path / target["relative_path"]).exists() for target in result["targets"])
    assert (artifact_root / "runs" / "m2" / "weights" / "best.pt").read_bytes() == b"formal-best"
    assert (artifact_root / "runs" / "m2" / "weights" / "last.pt").read_bytes() == b"formal-last"
    assert (artifact_root / "training" / "formal_m2" / "canonical.pt").read_bytes() == b"formal-canonical"


def test_cleanup_dry_run_has_no_side_effects(tmp_path: Path) -> None:
    artifact_root = _seed_completed_repo(tmp_path)

    plan = build_cleanup_plan(tmp_path, artifact_root, service_active=lambda _unit: False)

    assert not (artifact_root / "cleanup_manifest.json").exists()
    assert all((tmp_path / target.relative_path).exists() for target in plan.targets)
