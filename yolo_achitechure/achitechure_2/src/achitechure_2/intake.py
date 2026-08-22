"""yolo_combine winner 的不可變 Handoff Revision 契約。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from torch import nn

from .config import EXPECTED_ULTRALYTICS_VERSION
from .graph import inspect_c3k2_layer

FUSION_KINDS = frozenset({"shared_dual_head", "routed_dual", "partial_shared"})
REGION_ROLES = frozenset({"shared", "detect_specific", "pose_specific"})
REQUIRED_TRAINING_FIELDS = frozenset(
    {
        "batch",
        "fraction",
        "scale",
        "cache",
        "imgsz",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "nbs",
        "losses",
        "augmentation",
        "freeze_policy",
        "amp",
        "deterministic",
        "seed",
        "optimizer_steps",
        "validation_interval",
        "task_ratio",
        "epochs",
        "patience",
    }
)


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ArtifactDeclaration:
    path: Path
    sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        base: Path,
        label: str,
    ) -> ArtifactDeclaration:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"{label} must declare exactly path and sha256")
        path = Path(str(value["path"]))
        if not path.is_absolute():
            path = base / path
        if not _valid_sha256(value["sha256"]):
            raise ValueError(f"{label}.sha256 must be a lowercase SHA256 digest")
        return cls(path.resolve(), str(value["sha256"]))

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class CandidateRegion:
    region_id: str
    role: str
    tasks: tuple[str, ...]
    module_paths: tuple[str, ...]
    head_paths: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> CandidateRegion:
        expected = {"region_id", "role", "tasks", "module_paths", "head_paths"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(f"{label} fields must be exactly {sorted(expected)}")
        region_id = str(value["region_id"])
        role = str(value["role"])
        tasks = tuple(str(task) for task in value["tasks"])
        module_paths = tuple(str(path) for path in value["module_paths"])
        head_paths = tuple(str(path) for path in value["head_paths"])
        if not region_id or role not in REGION_ROLES:
            raise ValueError(f"{label}: invalid region_id/role")
        expected_tasks = {
            "shared": ("detect", "pose"),
            "detect_specific": ("detect",),
            "pose_specific": ("pose",),
        }[role]
        if tasks != expected_tasks:
            raise ValueError(f"{label}: role={role} requires tasks={expected_tasks}")
        if not module_paths or not head_paths:
            raise ValueError(f"{label}: module_paths/head_paths cannot be empty")
        all_paths = (*module_paths, *head_paths)
        if len(set(all_paths)) != len(all_paths) or any(
            not path or path.startswith(".") or path.endswith(".") for path in all_paths
        ):
            raise ValueError(f"{label}: paths must be non-empty and unique")
        return cls(region_id, role, tasks, module_paths, head_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "role": self.role,
            "tasks": list(self.tasks),
            "module_paths": list(self.module_paths),
            "head_paths": list(self.head_paths),
        }


def _normalize_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {int(index): str(name) for index, name in value.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("model names keys must be integer-like") from error
    raise ValueError("model names must be a list or mapping")


def _normalize_contract(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key: Any = key
            if isinstance(key, str) and key.isdigit():
                normalized_key = int(key)
            normalized[normalized_key] = _normalize_contract(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_contract(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_contract(item) for item in value]
    return value


def _validate_model_contract(contract: Any, fusion_kind: str) -> dict[str, Any]:
    required = {
        "interface",
        "model_kind",
        "head_inputs",
        "detect_nc",
        "pose_nc",
        "kpt_shape",
        "detect_names",
        "pose_names",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError(f"model_contract fields must be exactly {sorted(required)}")
    normalized = _normalize_contract(contract)
    if normalized["interface"] != "model(images, tasks=detect|pose|both)":
        raise ValueError("model_contract interface must support detect/pose/both")
    if normalized["model_kind"] != fusion_kind:
        raise ValueError("model_contract.model_kind must match fusion_kind")
    if normalized["detect_nc"] != 80:
        raise ValueError("main Detect contract requires detect_nc=80")
    detect_names = _normalize_names(normalized["detect_names"])
    if len(detect_names) != 80 or detect_names.get(0) != "person":
        raise ValueError("main Detect contract requires all COCO80 names and class 0 person")
    if normalized["pose_nc"] != 2:
        raise ValueError("main Pose contract requires pose_nc=2")
    if _normalize_names(normalized["pose_names"]) != {0: "ball", 1: "bat"}:
        raise ValueError("main Pose contract requires names ball/bat")
    if normalized["kpt_shape"] != [2, 3]:
        raise ValueError("main Pose contract requires kpt_shape=[2,3]")
    head_inputs = normalized["head_inputs"]
    if not isinstance(head_inputs, list) or len(head_inputs) != 3:
        raise ValueError("model_contract.head_inputs must contain P3/P4/P5 inputs")
    normalized["detect_names"] = detect_names
    normalized["pose_names"] = {0: "ball", 1: "bat"}
    return normalized


def _validate_region_topology(fusion_kind: str, regions: tuple[CandidateRegion, ...]) -> None:
    roles = tuple(region.role for region in regions)
    if len({region.region_id for region in regions}) != len(regions):
        raise ValueError("candidate region_id values must be unique")
    module_paths = [path for region in regions for path in region.module_paths]
    if len(set(module_paths)) != len(module_paths):
        raise ValueError("candidate module paths cannot occur in multiple regions")
    if fusion_kind == "shared_dual_head" and roles != ("shared",):
        raise ValueError("shared_dual_head requires exactly one shared candidate region")
    if fusion_kind == "routed_dual" and set(roles) != {"detect_specific", "pose_specific"}:
        raise ValueError("routed_dual requires detect_specific and pose_specific regions")
    if fusion_kind == "partial_shared" and "shared" not in roles:
        raise ValueError("partial_shared requires at least one shared region")


@dataclass(frozen=True)
class HandoffManifest:
    revision_id: str
    winner_id: str
    fusion_kind: str
    checkpoint: ArtifactDeclaration
    builder: ArtifactDeclaration
    architecture: ArtifactDeclaration
    training_recipe: ArtifactDeclaration
    datasets: Mapping[str, ArtifactDeclaration]
    selection: ArtifactDeclaration
    fresh_process_report: ArtifactDeclaration
    environment: Mapping[str, str]
    model_contract: Mapping[str, Any]
    candidate_regions: tuple[CandidateRegion, ...]
    protected_module_paths: tuple[str, ...]
    frozen_module_paths: tuple[str, ...]
    source_manifest: Path

    @classmethod
    def load(cls, path: str | Path) -> HandoffManifest:
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "producer",
            "revision_id",
            "winner_id",
            "fusion_kind",
            "checkpoint_format",
            "checkpoint",
            "builder",
            "architecture",
            "training_recipe",
            "datasets",
            "selection",
            "fresh_process_report",
            "environment",
            "model_contract",
            "candidate_regions",
            "protected_module_paths",
            "frozen_module_paths",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError(
                f"handoff fields incomplete: missing={sorted(required - set(payload))} "
                f"unknown={sorted(set(payload) - required)}"
            )
        if payload["schema_version"] != 2:
            raise ValueError("handoff schema_version must be 2")
        if payload["producer"] != "yolo_combine":
            raise ValueError("handoff producer must be yolo_combine")
        if payload["checkpoint_format"] != "state_dict_only":
            raise ValueError("handoff checkpoint_format must be state_dict_only")
        revision_id = str(payload["revision_id"])
        winner_id = str(payload["winner_id"])
        fusion_kind = str(payload["fusion_kind"])
        if not revision_id or not winner_id:
            raise ValueError("handoff revision_id/winner_id cannot be empty")
        if fusion_kind not in FUSION_KINDS:
            raise ValueError(f"unsupported fusion_kind {fusion_kind!r}")
        base = source.parent
        datasets = payload["datasets"]
        if not isinstance(datasets, dict) or set(datasets) != {
            "coco_detect",
            "bbat5_pose",
            "bbat5_detect",
        }:
            raise ValueError("datasets must declare coco_detect/bbat5_pose/bbat5_detect")
        environment = payload["environment"]
        if not isinstance(environment, dict) or set(environment) != {
            "torch",
            "ultralytics",
            "git_revision",
        }:
            raise ValueError("environment must declare torch/ultralytics/git_revision")
        if not all(str(value) for value in environment.values()):
            raise ValueError("environment values cannot be empty")
        regions_raw = payload["candidate_regions"]
        if not isinstance(regions_raw, list) or not regions_raw:
            raise ValueError("candidate_regions cannot be empty")
        regions = tuple(
            CandidateRegion.from_mapping(value, label=f"candidate_regions[{index}]")
            for index, value in enumerate(regions_raw)
        )
        _validate_region_topology(fusion_kind, regions)
        protected = tuple(str(item) for item in payload["protected_module_paths"])
        frozen = tuple(str(item) for item in payload["frozen_module_paths"])
        if (
            not protected
            or not frozen
            or len(set(protected)) != len(protected)
            or len(set(frozen)) != len(frozen)
            or not set(frozen).issubset(protected)
        ):
            raise ValueError("frozen_module_paths must be a non-empty subset of protected_module_paths")
        candidate_paths = {path for region in regions for path in region.module_paths}
        if candidate_paths & set(protected):
            raise ValueError("candidate module paths overlap protected_module_paths")
        head_paths = {path for region in regions for path in region.head_paths}
        if not head_paths.issubset(protected):
            raise ValueError("all task heads must be listed in protected_module_paths")
        return cls(
            revision_id=revision_id,
            winner_id=winner_id,
            fusion_kind=fusion_kind,
            checkpoint=ArtifactDeclaration.from_mapping(
                payload["checkpoint"], base=base, label="checkpoint"
            ),
            builder=ArtifactDeclaration.from_mapping(payload["builder"], base=base, label="builder"),
            architecture=ArtifactDeclaration.from_mapping(
                payload["architecture"], base=base, label="architecture"
            ),
            training_recipe=ArtifactDeclaration.from_mapping(
                payload["training_recipe"], base=base, label="training_recipe"
            ),
            datasets={
                name: ArtifactDeclaration.from_mapping(value, base=base, label=f"datasets.{name}")
                for name, value in datasets.items()
            },
            selection=ArtifactDeclaration.from_mapping(
                payload["selection"], base=base, label="selection"
            ),
            fresh_process_report=ArtifactDeclaration.from_mapping(
                payload["fresh_process_report"], base=base, label="fresh_process_report"
            ),
            environment={name: str(value) for name, value in environment.items()},
            model_contract=_validate_model_contract(payload["model_contract"], fusion_kind),
            candidate_regions=regions,
            protected_module_paths=protected,
            frozen_module_paths=frozen,
            source_manifest=source,
        )

    def artifacts(self) -> dict[str, ArtifactDeclaration]:
        return {
            "checkpoint": self.checkpoint,
            "builder": self.builder,
            "architecture": self.architecture,
            "training_recipe": self.training_recipe,
            "selection": self.selection,
            "fresh_process_report": self.fresh_process_report,
            **{f"datasets.{name}": value for name, value in self.datasets.items()},
        }


@dataclass(frozen=True)
class IntakeReport:
    accepted: bool
    accepted_at: str
    revision_id: str
    winner_id: str
    fusion_kind: str
    handoff_manifest: dict[str, str]
    artifacts: Mapping[str, Mapping[str, str]]
    environment: Mapping[str, str]
    model_contract: Mapping[str, Any]
    candidate_regions: tuple[Mapping[str, Any], ...]
    protected_module_paths: tuple[str, ...]
    frozen_module_paths: tuple[str, ...]
    graph_audit: Mapping[str, Any]
    training_recipe_fields: tuple[str, ...]
    fresh_process: Mapping[str, Any]


def _verify_artifact(declaration: ArtifactDeclaration, label: str) -> None:
    if not declaration.path.is_file():
        raise FileNotFoundError(declaration.path)
    actual = file_sha256(declaration.path)
    if actual != declaration.sha256:
        raise ValueError(f"{label} SHA256 mismatch: declared {declaration.sha256}, got {actual}")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _load_training_recipe(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("training_recipe must contain a YAML mapping")
    missing = REQUIRED_TRAINING_FIELDS - set(payload)
    if missing:
        raise ValueError(f"training_recipe missing required fields: {sorted(missing)}")
    if payload["seed"] != 0:
        raise ValueError("training_recipe seed must be 0 for first-round comparison")
    return payload


def _model_contract(model: nn.Module) -> dict[str, Any]:
    method = getattr(model, "contract", None)
    if not callable(method):
        raise TypeError("winner model must expose contract()")
    value = method()
    if not isinstance(value, dict):
        raise TypeError("winner model contract() must return a mapping")
    return _normalize_contract(value)


def _audit_model_paths(model: nn.Module, manifest: HandoffManifest) -> dict[str, Any]:
    candidate_reports: list[dict[str, Any]] = []
    for region in manifest.candidate_regions:
        for path in region.module_paths:
            try:
                module = model.get_submodule(path)
            except AttributeError as error:
                raise ValueError(f"candidate module path does not exist: {path}") from error
            try:
                report = inspect_c3k2_layer(-1, module)
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError(f"candidate module is not C3k2-compatible: {path}") from error
            if (
                report.e != 0.5
                or report.inner_n != 2
                or report.kernel_mode != "3x3_3x3"
                or report.use_rep
            ):
                raise ValueError(f"candidate module baseline factors drift at {path}")
            candidate_reports.append(
                {
                    "path": path,
                    "region_id": region.region_id,
                    "role": region.role,
                    "e": report.e,
                    "inner_n": report.inner_n,
                    "kernel_mode": report.kernel_mode,
                    "use_rep": report.use_rep,
                }
            )
    for path in (*manifest.protected_module_paths, *manifest.frozen_module_paths):
        try:
            model.get_submodule(path)
        except AttributeError as error:
            raise ValueError(f"protected/frozen module path does not exist: {path}") from error
    return {
        "candidate_module_paths": [item["path"] for item in candidate_reports],
        "candidate_modules": candidate_reports,
        "protected_module_paths": list(manifest.protected_module_paths),
        "frozen_module_paths": list(manifest.frozen_module_paths),
    }


def validate_handoff(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    loader: Callable[[Path], nn.Module] | None = None,
) -> IntakeReport:
    """驗證 provenance、主任務語意、region graph 與 fresh-process 證據。"""

    del project_root  # 保留穩定公開介面；所有路徑都相對 manifest 解析。
    manifest = HandoffManifest.load(manifest_path)
    for label, artifact in manifest.artifacts().items():
        _verify_artifact(artifact, label)
    if manifest.environment["torch"] != torch.__version__:
        raise ValueError(
            f"PyTorch version mismatch: handoff {manifest.environment['torch']}, runtime {torch.__version__}"
        )
    if manifest.environment["ultralytics"] != ultralytics.__version__:
        raise ValueError(
            "Ultralytics version mismatch: "
            f"handoff {manifest.environment['ultralytics']}, runtime {ultralytics.__version__}"
        )
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise ValueError(
            f"architecture_2 requires Ultralytics {EXPECTED_ULTRALYTICS_VERSION}"
        )
    recipe = _load_training_recipe(manifest.training_recipe.path)
    fresh = _load_json_object(manifest.fresh_process_report.path, "fresh-process report")
    if (
        fresh.get("passed") is not True
        or fresh.get("state_dict_reload") is not True
        or fresh.get("tasks") != ["detect", "pose", "both"]
    ):
        raise ValueError("fresh-process report must pass state_dict reload for detect/pose/both")
    graph_audit: dict[str, Any] = {
        "candidate_module_paths": [
            path for region in manifest.candidate_regions for path in region.module_paths
        ],
        "not_materialized": loader is None,
    }
    if loader is not None:
        model = loader(manifest.checkpoint.path)
        actual_contract = _model_contract(model)
        if actual_contract != _normalize_contract(manifest.model_contract):
            raise ValueError(f"model contract mismatch: {actual_contract} != {manifest.model_contract}")
        graph_audit = _audit_model_paths(model, manifest)
        graph_audit["not_materialized"] = False
    return IntakeReport(
        accepted=loader is not None,
        accepted_at=datetime.now(timezone.utc).isoformat(),
        revision_id=manifest.revision_id,
        winner_id=manifest.winner_id,
        fusion_kind=manifest.fusion_kind,
        handoff_manifest={
            "path": str(manifest.source_manifest),
            "sha256": file_sha256(manifest.source_manifest),
        },
        artifacts={
            label: artifact.to_dict() for label, artifact in manifest.artifacts().items()
        },
        environment=dict(manifest.environment),
        model_contract=dict(manifest.model_contract),
        candidate_regions=tuple(region.to_dict() for region in manifest.candidate_regions),
        protected_module_paths=manifest.protected_module_paths,
        frozen_module_paths=manifest.frozen_module_paths,
        graph_audit=graph_audit,
        training_recipe_fields=tuple(sorted(recipe)),
        fresh_process=fresh,
    )


def write_intake(report: IntakeReport, destination: str | Path) -> Path:
    if not report.accepted or report.graph_audit.get("not_materialized") is not False:
        raise ValueError("未 materialize 並驗證 winner graph，不得寫入 accepted intake")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def require_accepted_intake(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root).resolve() / "artifacts/intake/accepted.json"
    if not path.is_file():
        raise RuntimeError("尚未驗收 yolo_combine winner handoff")
    payload = _load_json_object(path, "accepted intake")
    if payload.get("accepted") is not True or not payload.get("revision_id"):
        raise RuntimeError("intake artifact 未被接受或缺少 revision_id")
    manifest = payload.get("handoff_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("accepted intake 缺少 handoff_manifest")  # noqa: TRY004
    references: dict[str, Mapping[str, str]] = {"handoff_manifest": manifest}
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("accepted intake 缺少 artifacts")  # noqa: TRY004
    references.update(artifacts)
    for label, declaration in references.items():
        artifact_path = Path(str(declaration.get("path", "")))
        expected = declaration.get("sha256")
        if not artifact_path.is_file():
            raise RuntimeError(f"{label} missing after intake: {artifact_path}")
        actual = file_sha256(artifact_path)
        if actual != expected:
            raise RuntimeError(f"{label} changed after intake: {actual} != {expected}")
    return payload
