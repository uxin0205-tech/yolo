"""Formal achitechure_1 handoff contract and fail-closed intake validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics

from .graph import GraphReport, inspect_graph

VARIANTS = frozenset({"full35", "partial75"})


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactDeclaration:
    path: Path
    sha256: str

    @classmethod
    def from_mapping(cls, value: Any, *, base: Path, label: str) -> ArtifactDeclaration:
        if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
            raise ValueError(f"{label} must declare path and sha256")
        path = Path(value["path"])
        path = path if path.is_absolute() else base / path
        sha256 = str(value["sha256"]).lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"{label}.sha256 must be a lowercase SHA256 digest")
        return cls(path.resolve(), sha256)


@dataclass(frozen=True)
class CheckpointDeclaration:
    path: Path
    sha256: str
    map50_95: float

    @classmethod
    def from_mapping(cls, value: Any, *, base: Path, label: str) -> CheckpointDeclaration:
        if not isinstance(value, dict):
            raise TypeError(f"{label} must be an object")
        required = {"path", "sha256", "map50_95"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"{label} missing fields: {sorted(missing)}")
        path = Path(value["path"])
        path = path if path.is_absolute() else (base / path)
        score = float(value["map50_95"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{label}.map50_95 must be in [0, 1]")
        sha256 = str(value["sha256"]).lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"{label}.sha256 must be a lowercase SHA256 digest")
        return cls(path.resolve(), sha256, score)


@dataclass(frozen=True)
class HandoffManifest:
    variant: str
    float_checkpoint: CheckpointDeclaration
    bittrue_checkpoint: CheckpointDeclaration
    torch_version: str
    ultralytics_version: str
    selection_manifest: ArtifactDeclaration
    model_selection: dict[str, Any]
    source_manifest: Path

    @classmethod
    def load(cls, path: str | Path) -> HandoffManifest:
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("handoff schema_version must be 1")
        variant = str(payload.get("variant", "")).lower()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        environment = payload.get("environment")
        if (
            not isinstance(environment, dict)
            or not environment.get("torch")
            or not environment.get("ultralytics")
        ):
            raise ValueError("environment must declare torch and ultralytics versions")
        selection = payload.get("model_selection")
        if (
            not isinstance(selection, dict)
            or not selection.get("selected_variant")
            or not selection.get("basis")
        ):
            raise ValueError("model_selection must declare selected_variant and basis")
        if str(selection["selected_variant"]).lower() != variant:
            raise ValueError("model_selection.selected_variant does not match variant")
        return cls(
            variant=variant,
            float_checkpoint=CheckpointDeclaration.from_mapping(
                payload.get("float_checkpoint"), base=source.parent, label="float_checkpoint"
            ),
            bittrue_checkpoint=CheckpointDeclaration.from_mapping(
                payload.get("bittrue_checkpoint"), base=source.parent, label="bittrue_checkpoint"
            ),
            torch_version=str(environment["torch"]),
            ultralytics_version=str(environment["ultralytics"]),
            selection_manifest=ArtifactDeclaration.from_mapping(
                payload.get("selection_manifest"), base=source.parent, label="selection_manifest"
            ),
            model_selection=selection,
            source_manifest=source,
        )


@dataclass(frozen=True)
class IntakeReport:
    accepted: bool
    accepted_at: str
    handoff_manifest: str
    manifest_sha256: str
    variant: str
    float_checkpoint: dict[str, Any]
    bittrue_checkpoint: dict[str, Any]
    selection_manifest: dict[str, str]
    float_graph: dict[str, Any]
    bittrue_graph: dict[str, Any]
    fresh_process: dict[str, Any]
    environment: dict[str, str]
    model_selection: dict[str, Any]


def _verify_artifact(declaration: ArtifactDeclaration, label: str) -> None:
    if not declaration.path.is_file():
        raise FileNotFoundError(declaration.path)
    actual = file_sha256(declaration.path)
    if actual != declaration.sha256:
        raise ValueError(f"{label} SHA256 mismatch: declared {declaration.sha256}, got {actual}")


def _verify_checkpoint(declaration: CheckpointDeclaration, label: str) -> None:
    if not declaration.path.is_file():
        raise FileNotFoundError(declaration.path)
    actual = file_sha256(declaration.path)
    if actual != declaration.sha256:
        raise ValueError(f"{label} SHA256 mismatch: declared {declaration.sha256}, got {actual}")


def _default_loader(path: Path) -> torch.nn.Module:
    from ultralytics import YOLO

    return YOLO(str(path)).model


def _assert_backend(graph: GraphReport, expected: str, label: str) -> None:
    actual = graph.attention_normalizations
    aliases = {"float": {"piecewise_linear", "float_pwl"}, "bittrue": {"bit_true_pwl"}}
    if any(name not in aliases[expected] for name in actual):
        raise ValueError(f"{label} attention normalization must be {expected}, got {actual}")


def fresh_process_reload(project_root: Path, checkpoints: tuple[Path, Path]) -> dict[str, Any]:
    """Reload both checkpoints in a separate interpreter and repeat graph assertions."""

    repository = project_root.parents[1]
    extra = (
        project_root / "src",
        project_root.parent / "achitechure_1" / "src",
        repository / "yolo_attention_final" / "src",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in extra) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import json; from ultralytics import YOLO; from achitechure_2.graph import inspect_graph; "
        f"paths={tuple(str(path) for path in checkpoints)!r}; "
        "reports=[inspect_graph(YOLO(p).model).to_dict() for p in paths]; print(json.dumps(reports))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"fresh-process reload failed: {result.stderr.strip()}")
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("[")]
    if not lines:
        raise RuntimeError(f"fresh-process reload produced no graph report: {result.stdout.strip()}")
    return {"ok": True, "reports": json.loads(lines[-1])}


def validate_handoff(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    loader: Callable[[Path], torch.nn.Module] = _default_loader,
    require_fresh_process: bool = True,
) -> IntakeReport:
    """Validate provenance, both graph modes, and reloadability before accepting C0."""

    manifest = HandoffManifest.load(manifest_path)
    _verify_checkpoint(manifest.float_checkpoint, "Float-PWL checkpoint")
    _verify_checkpoint(manifest.bittrue_checkpoint, "Bit-True checkpoint")
    _verify_artifact(manifest.selection_manifest, "selection manifest")
    if manifest.torch_version != torch.__version__:
        raise ValueError(
            f"PyTorch version mismatch: handoff {manifest.torch_version}, runtime {torch.__version__}"
        )
    if manifest.ultralytics_version != ultralytics.__version__:
        raise ValueError(
            f"Ultralytics version mismatch: handoff {manifest.ultralytics_version}, runtime {ultralytics.__version__}"
        )
    float_graph = inspect_graph(loader(manifest.float_checkpoint.path))
    bittrue_graph = inspect_graph(loader(manifest.bittrue_checkpoint.path))
    for label, graph in (("Float-PWL", float_graph), ("Bit-True", bittrue_graph)):
        if graph.masf_variant != manifest.variant:
            raise ValueError(f"{label} MASF variant {graph.masf_variant} does not match {manifest.variant}")
    _assert_backend(float_graph, "float", "Float-PWL checkpoint")
    _assert_backend(bittrue_graph, "bittrue", "Bit-True checkpoint")
    if (
        float_graph.detect_inputs != bittrue_graph.detect_inputs
        or float_graph.strides != bittrue_graph.strides
    ):
        raise ValueError("Float-PWL and Bit-True graphs disagree")
    fresh = (
        fresh_process_reload(
            Path(project_root).resolve(),
            (manifest.float_checkpoint.path, manifest.bittrue_checkpoint.path),
        )
        if require_fresh_process
        else {"ok": False, "skipped_for_test": True}
    )
    checkpoint_payload = lambda declaration: {
        "path": str(declaration.path),
        "sha256": declaration.sha256,
        "map50_95": declaration.map50_95,
    }
    return IntakeReport(
        accepted=True,
        accepted_at=datetime.now(timezone.utc).isoformat(),
        handoff_manifest=str(manifest.source_manifest),
        manifest_sha256=file_sha256(manifest.source_manifest),
        variant=manifest.variant,
        float_checkpoint=checkpoint_payload(manifest.float_checkpoint),
        bittrue_checkpoint=checkpoint_payload(manifest.bittrue_checkpoint),
        selection_manifest={
            "path": str(manifest.selection_manifest.path),
            "sha256": manifest.selection_manifest.sha256,
        },
        float_graph=float_graph.to_dict(),
        bittrue_graph=bittrue_graph.to_dict(),
        fresh_process=fresh,
        environment={"torch": torch.__version__, "ultralytics": ultralytics.__version__},
        model_selection=manifest.model_selection,
    )


def write_intake(report: IntakeReport, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def require_accepted_intake(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root).resolve() / "artifacts/intake/accepted.json"
    if not path.is_file():
        raise RuntimeError("formal handoff is not accepted; run intake --execute first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("accepted") is not True:
        raise RuntimeError("intake artifact is not accepted")
    artifacts = (
        (payload.get("handoff_manifest"), payload.get("manifest_sha256"), "handoff manifest"),
        (
            payload.get("float_checkpoint", {}).get("path"),
            payload.get("float_checkpoint", {}).get("sha256"),
            "Float checkpoint",
        ),
        (
            payload.get("bittrue_checkpoint", {}).get("path"),
            payload.get("bittrue_checkpoint", {}).get("sha256"),
            "Bit-True checkpoint",
        ),
        (
            payload.get("selection_manifest", {}).get("path"),
            payload.get("selection_manifest", {}).get("sha256"),
            "selection manifest",
        ),
    )
    for artifact_path, expected, label in artifacts:
        if not artifact_path or not Path(artifact_path).is_file():
            raise RuntimeError(f"accepted {label} is missing: {artifact_path}")
        actual = file_sha256(artifact_path)
        if actual != expected:
            raise RuntimeError(f"accepted {label} changed after intake")
    return payload
