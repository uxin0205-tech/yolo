"""Load and audit isolated Full35 and Partial75 experiment workspaces."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .data import CanonicalBBAT5, DatasetContractError

Architecture = Literal["full35", "partial75"]
VariantRole = Literal["primary", "fallback"]

_TOP_LEVEL_KEYS = {
    "schema_version",
    "architecture",
    "role",
    "source",
    "dataset_contract",
    "datasets",
    "runs",
    "documents",
}
_SOURCE_KEYS = {
    "bundle",
    "float_checkpoint",
    "float_sha256",
    "bittrue_checkpoint",
    "bittrue_sha256",
}
_DATASET_CONTRACT_KEYS = {"registry", "version"}
_DATASET_KEYS = {"coco_detect", "bbt5_pose", "bbt5_detect"}
_RUN_KEYS = {"root"}
_DOCUMENT_KEYS = {"baseline", "training", "fusion"}


class VariantConfigError(ValueError):
    """Raised when a variant workspace violates its fail-closed schema."""


@dataclass(frozen=True)
class VariantAudit:
    missing_paths: tuple[str, ...]
    hash_mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_paths and not self.hash_mismatches


@dataclass(frozen=True)
class VariantWorkspace:
    """One architecture's sources, evidence, settings, and isolated run root."""

    root: Path
    project_root: Path
    architecture: Architecture
    role: VariantRole
    source_bundle: Path
    float_checkpoint: Path
    float_sha256: str
    bittrue_checkpoint: Path
    bittrue_sha256: str
    bbat5_registry: Path
    bbat5_version: str
    datasets: Mapping[str, Path]
    run_root: Path
    documents: Mapping[str, Path]

    @classmethod
    def load(cls, directory: str | Path) -> VariantWorkspace:
        root = Path(directory).expanduser().resolve()
        config_path = root / "variant.yaml"
        payload = _load_yaml(config_path)
        _require_keys(payload, _TOP_LEVEL_KEYS, "variant")
        if payload["schema_version"] != 2:
            raise VariantConfigError("variant.schema_version must be 2")

        architecture = payload["architecture"]
        if architecture not in {"full35", "partial75"}:
            raise VariantConfigError("architecture must be full35 or partial75")
        if root.name != architecture:
            raise VariantConfigError(
                f"variant folder {root.name!r} does not match architecture {architecture!r}"
            )
        role = payload["role"]
        if role not in {"primary", "fallback"}:
            raise VariantConfigError("role must be primary or fallback")

        project_root = _find_project_root(root)
        source = _mapping(payload, "source")
        dataset_contract = _mapping(payload, "dataset_contract")
        datasets = _mapping(payload, "datasets")
        runs = _mapping(payload, "runs")
        documents = _mapping(payload, "documents")
        _require_keys(source, _SOURCE_KEYS, "source")
        _require_keys(dataset_contract, _DATASET_CONTRACT_KEYS, "dataset_contract")
        _require_keys(datasets, _DATASET_KEYS, "datasets")
        _require_keys(runs, _RUN_KEYS, "runs")
        _require_keys(documents, _DOCUMENT_KEYS, "documents")

        # Mutable outputs belong to the selected architecture folder. Resolving
        # from root prevents one experiment from writing into the other run tree.
        run_root = _resolve_relative(root, runs["root"], "runs.root")
        document_paths = {
            name: _resolve_relative(root, value, f"documents.{name}")
            for name, value in documents.items()
        }
        return cls(
            root=root,
            project_root=project_root,
            architecture=architecture,
            role=role,
            source_bundle=_absolute_path(source["bundle"], "source.bundle"),
            float_checkpoint=_absolute_path(
                source["float_checkpoint"], "source.float_checkpoint"
            ),
            float_sha256=_sha_value(source["float_sha256"], "source.float_sha256"),
            bittrue_checkpoint=_absolute_path(
                source["bittrue_checkpoint"], "source.bittrue_checkpoint"
            ),
            bittrue_sha256=_sha_value(
                source["bittrue_sha256"], "source.bittrue_sha256"
            ),
            bbat5_registry=_absolute_path(
                dataset_contract["registry"], "dataset_contract.registry"
            ),
            bbat5_version=_text(
                dataset_contract["version"], "dataset_contract.version"
            ),
            datasets={
                name: _absolute_path(value, f"datasets.{name}")
                for name, value in datasets.items()
            },
            run_root=run_root,
            documents=document_paths,
        )

    def audit(self, *, verify_hashes: bool = False) -> VariantAudit:
        """Check referenced files without importing Torch or constructing a model."""

        missing: list[str] = []
        if not self.source_bundle.is_dir():
            missing.append(str(self.source_bundle))
        for path in (
            self.root / "README.md",
            self.root / "variant.yaml",
            self.float_checkpoint,
            self.bittrue_checkpoint,
            self.bbat5_registry,
            *self.datasets.values(),
            *self.documents.values(),
        ):
            if not path.is_file():
                missing.append(str(path))

        mismatches: list[str] = []
        if self.bbat5_registry.is_file():
            try:
                contract = CanonicalBBAT5.load(self.bbat5_registry)
            except (DatasetContractError, FileNotFoundError) as error:
                mismatches.append(f"bbat5-contract:{error}")
            else:
                dataset_audit = contract.audit()
                if (
                    not dataset_audit.derivation_exact
                    or dataset_audit.broken_image_links
                    or dataset_audit.source_group_overlap
                    or dataset_audit.train.negative_keypoint_rows
                    or dataset_audit.valid.negative_keypoint_rows
                ):
                    mismatches.append(f"bbat5-content:{dataset_audit}")
                if contract.dataset_id != self.bbat5_version:
                    mismatches.append(
                        f"bbat5-version:{contract.dataset_id}!={self.bbat5_version}"
                    )
                expected = {
                    "bbt5_pose": contract.pose.yaml,
                    "bbt5_detect": contract.detect.yaml,
                }
                for name, path in expected.items():
                    if self.datasets[name] != path:
                        mismatches.append(f"{name}:{self.datasets[name]}!={path}")
        if verify_hashes:
            for kind, path, expected in (
                ("float", self.float_checkpoint, self.float_sha256),
                ("bittrue", self.bittrue_checkpoint, self.bittrue_sha256),
            ):
                if path.is_file():
                    actual = _file_sha256(path)
                    if actual != expected:
                        mismatches.append(f"{kind}:{actual}!={expected}")
        return VariantAudit(tuple(sorted(missing)), tuple(sorted(mismatches)))

    @property
    def pose_view_root(self) -> Path:
        """Symlink-only cache view derived from the canonical BBAT5 asset root."""

        return self.run_root / "cache-views" / "bbat5-v1"

    @property
    def pose_run_root(self) -> Path:
        """Independent Pose run root owned only by this workspace."""

        return self.run_root / "pose"

    @property
    def fusion_run_root(self) -> Path:
        """Shared-model run root owned only by this workspace."""

        return self.run_root / "fusion"

    @property
    def cpu_report_path(self) -> Path:
        """Latest generated CPU acceptance report for this workspace."""

        return self.run_root / "cpu-validation.json"

    @property
    def joint_smoke_report_path(self) -> Path:
        """Latest real-data joint smoke report for this workspace."""

        return self.run_root / "joint-smoke.json"


def load_variant(
    name: Architecture, *, project_root: str | Path | None = None
) -> VariantWorkspace:
    """Load a named workspace through one stable project-level seam."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    return VariantWorkspace.load(root / "variants" / name)


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VariantConfigError(f"{path} must contain a mapping")
    return payload


def _mapping(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload[name]
    if not isinstance(value, dict):
        raise VariantConfigError(f"{name} must be a mapping")
    return value


def _require_keys(
    payload: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise VariantConfigError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _find_project_root(root: Path) -> Path:
    for candidate in (root, *root.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise VariantConfigError(f"cannot find project root above {root}")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VariantConfigError(f"{label} must be a non-empty string")
    return value


def _absolute_path(value: object, label: str) -> Path:
    path = Path(_text(value, label)).expanduser()
    if not path.is_absolute():
        raise VariantConfigError(f"{label} must be absolute")
    return path.resolve()


def _resolve_relative(parent: Path, value: object, label: str) -> Path:
    path = Path(_text(value, label))
    if path.is_absolute():
        raise VariantConfigError(f"{label} must be relative")
    resolved = (parent / path).resolve()
    if not resolved.is_relative_to(parent):
        raise VariantConfigError(f"{label} escapes {parent}")
    return resolved


def _sha_value(value: object, label: str) -> str:
    text = _text(value, label).lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise VariantConfigError(f"{label} must be a SHA256 hex digest")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
