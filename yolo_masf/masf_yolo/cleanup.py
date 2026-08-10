"""Fail-closed cleanup for disposable experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from masf_yolo.artifacts.io import atomic_write_json


@dataclass(frozen=True, slots=True)
class CleanupTarget:
    relative_path: str
    category: str
    reason: str
    size_bytes: int
    sha256: str
    status: str = "planned"


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    repo_root: Path
    artifact_root: Path
    pipeline_id: str
    targets: tuple[CleanupTarget, ...]
    preserved_checkpoint_hashes: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": "dry-run",
            "pipeline_id": self.pipeline_id,
            "target_count": len(self.targets),
            "total_planned_bytes": sum(target.size_bytes for target in self.targets),
            "preserved_checkpoint_hashes": dict(self.preserved_checkpoint_hashes),
            "targets": [asdict(target) for target in self.targets],
        }


_DELETE_PATTERNS = (
    (".pytest_cache/**/*", "cache", "可重建的 pytest cache"),
    ("**/__pycache__/**/*", "cache", "可重建的 Python bytecode cache"),
    ("**/*.pyc", "cache", "可重建的 Python bytecode"),
    ("**/*.pyo", "cache", "可重建的 optimized Python bytecode"),
    (
        "artifacts/static-phase1/smoke_runs/*/weights/*.pt",
        "smoke_checkpoint",
        "smoke acceptance 完成後不再保留的 native checkpoint",
    ),
    (
        "artifacts/static-phase1/training/smoke_*/canonical.pt",
        "smoke_checkpoint",
        "smoke acceptance 完成後不再保留的 canonical checkpoint",
    ),
    (
        "artifacts/static-phase1/preflight/*.pt",
        "preflight_checkpoint",
        "正式 pipeline 完成後可重建的 preflight checkpoint",
    ),
    (
        "artifacts/static-phase1/m7_gate/m7.pt",
        "gate_checkpoint",
        "M7 acceptance gate 完成後可重建的 checkpoint",
    ),
    (
        "artifacts/static-phase1/runs/*/train_batch*.jpg",
        "training_preview",
        "正式訓練的可重建 batch preview",
    ),
    (
        "artifacts/static-phase1/runs/*/labels.jpg",
        "training_preview",
        "正式訓練的可重建 label preview",
    ),
    (
        "artifacts/static-phase1/smoke_runs/*/train_batch*.jpg",
        "training_preview",
        "smoke 訓練的可重建 batch preview",
    ),
    (
        "artifacts/static-phase1/smoke_runs/*/labels.jpg",
        "training_preview",
        "smoke 訓練的可重建 label preview",
    ),
)

_PRESERVED_CHECKPOINT_PATTERNS = (
    "artifacts/static-phase1/runs/*/weights/best.pt",
    "artifacts/static-phase1/runs/*/weights/last.pt",
    "artifacts/static-phase1/training/b1_a/canonical.pt",
    "artifacts/static-phase1/training/b1_b/canonical.pt",
    "artifacts/static-phase1/training/formal_*/canonical.pt",
    "artifacts/static-phase1/training/sp2_a/canonical.pt",
    "artifacts/static-phase1/training/sp2_b/canonical.pt",
    "artifacts/static-phase1/training/sp2p_a/canonical.pt",
    "artifacts/static-phase1/training/sp2p_b/canonical.pt",
)

_RUNTIME_REFERENCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".toml", ".sh"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required cleanup gate is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"cleanup gate must be a JSON object: {path}")
    return value


def _inside_repo(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(repo_root):
        raise RuntimeError(f"cleanup target resolves outside repo root: {path}")
    return resolved


def _runtime_references_yolo26n(repo_root: Path) -> bool:
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _RUNTIME_REFERENCE_SUFFIXES:
            continue
        relative = path.relative_to(repo_root)
        if relative.parts and relative.parts[0] == "artifacts":
            continue
        try:
            if "yolo26n.pt" in path.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _preserved_checkpoint_hashes(repo_root: Path) -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in _PRESERVED_CHECKPOINT_PATTERNS:
        paths.update(path for path in repo_root.glob(pattern) if path.is_file())
    return {
        path.relative_to(repo_root).as_posix(): _sha256(_inside_repo(path, repo_root))
        for path in sorted(paths)
    }


def build_cleanup_plan(
    repo_root: Path,
    artifact_root: Path,
    *,
    service_active: Callable[[str], bool],
) -> CleanupPlan:
    """Build an immutable cleanup plan after all terminal gates pass."""

    repo_root = repo_root.resolve()
    artifact_root = _inside_repo(artifact_root, repo_root)
    pipeline = _read_json(artifact_root / "pipeline.json")
    unit = pipeline.get("unit")
    pipeline_id = pipeline.get("pipeline_id")
    if not isinstance(unit, str) or not isinstance(pipeline_id, str):
        raise RuntimeError("pipeline metadata is incomplete")
    if service_active(unit):
        raise RuntimeError(f"cleanup refused because service is active: {unit}")
    report = _read_json(artifact_root / "stages" / "report.json")
    if report.get("status") != "completed":
        raise RuntimeError("report stage is not completed")
    audit = _read_json(artifact_root / "final_audit.json")
    if audit.get("ok") is not True or audit.get("errors") != []:
        raise RuntimeError("final audit did not pass")

    preserved = _preserved_checkpoint_hashes(repo_root)
    preserved_resolved = {_inside_repo(repo_root / relative, repo_root) for relative in preserved}
    candidates: dict[str, CleanupTarget] = {}
    for pattern, category, reason in _DELETE_PATTERNS:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            resolved = _inside_repo(path, repo_root)
            if resolved in preserved_resolved:
                raise RuntimeError(f"cleanup target conflicts with preserved checkpoint: {path}")
            relative = path.relative_to(repo_root).as_posix()
            candidates[relative] = CleanupTarget(
                relative_path=relative,
                category=category,
                reason=reason,
                size_bytes=resolved.stat().st_size,
                sha256=_sha256(resolved),
            )

    yolo26n = repo_root / "yolo26n.pt"
    if yolo26n.is_file() and not _runtime_references_yolo26n(repo_root):
        resolved = _inside_repo(yolo26n, repo_root)
        candidates["yolo26n.pt"] = CleanupTarget(
            relative_path="yolo26n.pt",
            category="unreferenced_model",
            reason="runtime 程式與設定均未引用的下載模型",
            size_bytes=resolved.stat().st_size,
            sha256=_sha256(resolved),
        )

    return CleanupPlan(
        repo_root=repo_root,
        artifact_root=artifact_root,
        pipeline_id=pipeline_id,
        targets=tuple(candidates[key] for key in sorted(candidates)),
        preserved_checkpoint_hashes=preserved,
    )


def _verify_preserved(plan: CleanupPlan) -> None:
    actual = _preserved_checkpoint_hashes(plan.repo_root)
    if actual != plan.preserved_checkpoint_hashes:
        raise RuntimeError("preserved checkpoint hashes changed during cleanup")


def apply_cleanup(plan: CleanupPlan, manifest_path: Path) -> dict[str, object]:
    """Apply a verified plan and atomically update its deletion manifest."""

    manifest_path = _inside_repo(manifest_path, plan.repo_root)
    _verify_preserved(plan)
    targets = list(plan.targets)
    payload: dict[str, object] = {
        "schema_version": 1,
        "pipeline_id": plan.pipeline_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_planned_bytes": sum(target.size_bytes for target in targets),
        "total_deleted_bytes": 0,
        "preserved_checkpoint_hashes": dict(plan.preserved_checkpoint_hashes),
        "targets": [asdict(target) for target in targets],
    }
    atomic_write_json(manifest_path, payload)

    deleted_bytes = 0
    for index, target in enumerate(targets):
        path = plan.repo_root / target.relative_path
        resolved = _inside_repo(path, plan.repo_root)
        if not path.is_file():
            raise RuntimeError(f"cleanup target disappeared before deletion: {target.relative_path}")
        if resolved.stat().st_size != target.size_bytes or _sha256(resolved) != target.sha256:
            raise RuntimeError(f"cleanup target changed before deletion: {target.relative_path}")
        path.unlink()
        deleted_bytes += target.size_bytes
        targets[index] = replace(target, status="deleted")
        payload["targets"] = [asdict(item) for item in targets]
        payload["total_deleted_bytes"] = deleted_bytes
        atomic_write_json(manifest_path, payload)

    _verify_preserved(plan)
    return payload
