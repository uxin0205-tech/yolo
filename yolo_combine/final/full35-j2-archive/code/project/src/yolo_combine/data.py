"""Read-only lineage checks for the paired BBT5 Pose and Detect views."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import Task

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
DEFAULT_BBAT5_REGISTRY = Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml")


class DatasetContractError(ValueError):
    """Raised when the canonical dataset registry or one of its files drifts."""


@dataclass(frozen=True)
class SplitStats:
    images: int
    labels: int
    instances: int
    empty_labels: int
    ball_instances: int
    bat_instances: int
    negative_keypoint_rows: int


@dataclass(frozen=True)
class BBT5AuditReport:
    train: SplitStats
    valid: SplitStats
    derivation_mismatches: int
    broken_image_links: int
    source_group_overlap: tuple[str, ...]

    @property
    def derivation_exact(self) -> bool:
        return self.derivation_mismatches == 0


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    root: Path
    yaml: Path
    manifest: Path
    source_yaml: Path
    images: int
    labels: int
    source_patched_coordinates: int


@dataclass(frozen=True)
class PreparedDetectSubset:
    root: Path
    yaml: Path
    manifest: Path
    images: int
    labels: int
    backgrounds: int
    source_images: int


@dataclass(frozen=True)
class _DatasetLayout:
    yaml: Path
    root: Path
    train_images: Path
    train_labels: Path
    val_images: Path
    val_labels: Path
    names: Mapping[int, str]
    kpt_shape: tuple[int, int] | None
    flip_idx: tuple[int, ...] | None

    def images(self, split: str) -> Path:
        return self.train_images if split == "train" else self.val_images

    def labels(self, split: str) -> Path:
        return self.train_labels if split == "train" else self.val_labels


@dataclass(frozen=True)
class CanonicalBBAT5:
    """Validated immutable BBAT5 contract loaded through one registry seam."""

    registry: Path
    dataset_id: str
    root: Path
    pose: _DatasetLayout
    detect: _DatasetLayout
    images: int
    train_images: int
    val_images: int
    source_groups: int
    patched_coordinates: int
    spec_version: str
    spec_sha256: str

    @classmethod
    def load(cls, registry: str | Path = DEFAULT_BBAT5_REGISTRY) -> CanonicalBBAT5:
        registry_path = Path(registry).expanduser().resolve()
        payload = _load_yaml_mapping(registry_path)
        required = {
            "schema_version",
            "dataset_id",
            "aliases",
            "status",
            "root",
            "spec",
            "source_archives",
            "tasks",
            "search_tasks",
            "manifests",
            "counts",
            "splits",
            "policy",
        }
        if set(payload) != required:
            raise DatasetContractError(
                f"registry keys drifted; missing={sorted(required - set(payload))}, "
                f"extra={sorted(set(payload) - required)}"
            )
        if payload["schema_version"] != 1:
            raise DatasetContractError("registry.schema_version must be 1")
        if payload["dataset_id"] != "bbat5-v1" or payload["status"] != "canonical":
            raise DatasetContractError("registry must select canonical bbat5-v1")

        root = _absolute_path(payload["root"], "registry.root")
        if not root.is_dir():
            raise FileNotFoundError(root)
        spec = _mapping(payload, "spec")
        spec_version = _text(spec.get("version"), "spec.version")
        spec_sha256 = _sha_value(spec.get("sha256"), "spec.sha256")

        tasks = _mapping(payload, "tasks")
        if set(tasks) != {"pose", "detect_2class"}:
            raise DatasetContractError(
                "registry.tasks must contain pose and detect_2class"
            )
        pose_yaml = _verified_reference(tasks["pose"], "tasks.pose")
        detect_yaml = _verified_reference(tasks["detect_2class"], "tasks.detect_2class")

        search_tasks = _mapping(payload, "search_tasks")
        if set(search_tasks) != {"pose", "detect_2class"}:
            raise DatasetContractError(
                "registry.search_tasks must contain pose and detect_2class"
            )
        _verified_reference(search_tasks["pose"], "search_tasks.pose")
        _verified_reference(search_tasks["detect_2class"], "search_tasks.detect_2class")

        manifests = _mapping(payload, "manifests")
        expected_manifests = {
            "source_audit",
            "split",
            "patch",
            "coco_exclusion",
            "rebuild",
        }
        if set(manifests) != expected_manifests:
            raise DatasetContractError("registry.manifests keys drifted")
        manifest_paths = {
            name: _verified_reference(manifests[name], f"manifests.{name}")
            for name in sorted(expected_manifests)
        }

        pose = _load_dataset_layout(pose_yaml)
        detect = _load_dataset_layout(detect_yaml)
        if pose.root != root / "pose" or detect.root != root / "detect":
            raise DatasetContractError("task YAML roots do not belong to registry.root")
        if dict(pose.names) != {0: "ball", 1: "bat"}:
            raise DatasetContractError("Pose names must be ball=0 and bat=1")
        if pose.names != detect.names:
            raise DatasetContractError("Pose and Detect class mappings differ")
        if pose.kpt_shape != (2, 3):
            raise DatasetContractError("Pose kpt_shape must be [2, 3]")

        counts = _mapping(payload, "counts")
        splits = _mapping(payload, "splits")
        formal = _mapping(splits, "formal")
        if splits.get("test", object()) is not None:
            raise DatasetContractError("bbat5-v1 must not declare a test split")
        images = _positive_int(counts.get("images"), "counts.images")
        source_groups = _positive_int(
            counts.get("source_groups"), "counts.source_groups"
        )
        patched = _positive_int(
            counts.get("patched_coordinates"), "counts.patched_coordinates"
        )
        train_images = _positive_int(
            formal.get("train_images"), "splits.formal.train_images"
        )
        val_images = _positive_int(formal.get("val_images"), "splits.formal.val_images")
        if formal.get("source_group_overlap") != 0:
            raise DatasetContractError("formal source_group_overlap must be zero")
        if train_images + val_images != images:
            raise DatasetContractError(
                "formal split counts do not sum to counts.images"
            )

        for task_name, layout in (("pose", pose), ("detect", detect)):
            actual_train = len(_indexed_files(layout.train_images, images=True))
            actual_val = len(_indexed_files(layout.val_images, images=True))
            train_labels = len(_indexed_files(layout.train_labels, images=False))
            val_labels = len(_indexed_files(layout.val_labels, images=False))
            if (actual_train, actual_val) != (train_images, val_images):
                raise DatasetContractError(
                    f"{task_name} split counts drifted: "
                    f"{actual_train}/{actual_val} != {train_images}/{val_images}"
                )
            if (train_labels, val_labels) != (train_images, val_images):
                raise DatasetContractError(f"{task_name} image/label pairing drifted")

        split_payload = _load_json_mapping(manifest_paths["split"])
        formal_manifest = _mapping(split_payload, "formal")
        assignment = _mapping(formal_manifest, "assignment")
        actual_assignment: dict[str, str] = {}
        for split, directory in (
            ("train", pose.train_images),
            ("val", pose.val_images),
        ):
            for stem in _indexed_files(directory, images=True):
                group = _source_group(stem)
                previous = actual_assignment.setdefault(group, split)
                if previous != split:
                    raise DatasetContractError(
                        f"formal split leaks source groups: {group!r} is in train and val"
                    )
        if len(actual_assignment) != source_groups:
            raise DatasetContractError(
                f"source group count drifted: {len(actual_assignment)} != {source_groups}"
            )
        expected_assignment = {
            str(group): str(split) for group, split in assignment.items()
        }
        if actual_assignment != expected_assignment:
            missing = sorted(set(expected_assignment) - set(actual_assignment))[:5]
            unexpected = sorted(set(actual_assignment) - set(expected_assignment))[:5]
            moved = sorted(
                group
                for group in set(actual_assignment) & set(expected_assignment)
                if actual_assignment[group] != expected_assignment[group]
            )[:5]
            raise DatasetContractError(
                "formal assignment drifted; "
                f"missing={missing}, unexpected={unexpected}, moved={moved}"
            )

        policy = _mapping(payload, "policy")
        if policy.get("training_default") != "formal":
            raise DatasetContractError("registry training_default must be formal")
        return cls(
            registry=registry_path,
            dataset_id="bbat5-v1",
            root=root,
            pose=pose,
            detect=detect,
            images=images,
            train_images=train_images,
            val_images=val_images,
            source_groups=source_groups,
            patched_coordinates=patched,
            spec_version=spec_version,
            spec_sha256=spec_sha256,
        )

    def layout(self, task: Task | str) -> _DatasetLayout:
        selected = Task(task)
        return self.pose if selected is Task.POSE else self.detect

    def audit(self) -> BBT5AuditReport:
        """Prove paired task labels, links, coordinates, and zero leakage."""

        return _audit_contract(self)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetContractError(f"{path} must contain a mapping")
    return payload


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetContractError(f"{path} must contain a JSON object")
    return payload


def _mapping(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DatasetContractError(f"{name} must be a mapping")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetContractError(f"{label} must be a non-empty string")
    return value


def _absolute_path(value: object, label: str) -> Path:
    path = Path(_text(value, label)).expanduser()
    if not path.is_absolute():
        raise DatasetContractError(f"{label} must be absolute")
    return path.resolve()


def _sha_value(value: object, label: str) -> str:
    digest = _text(value, label).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DatasetContractError(f"{label} must be a SHA256 digest")
    return digest


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DatasetContractError(f"{label} must be a positive integer")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_reference(value: object, label: str) -> Path:
    if not isinstance(value, dict) or set(value) not in (
        {"data_yaml", "data_yaml_sha256"},
        {"path", "sha256"},
    ):
        raise DatasetContractError(f"{label} must contain one path and one SHA256")
    path_key = "data_yaml" if "data_yaml" in value else "path"
    hash_key = "data_yaml_sha256" if "data_yaml_sha256" in value else "sha256"
    path = _absolute_path(value[path_key], f"{label}.{path_key}")
    expected = _sha_value(value[hash_key], f"{label}.{hash_key}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _file_sha256(path)
    if actual != expected:
        raise DatasetContractError(f"{label} SHA256 drifted: {actual} != {expected}")
    return path


def _resolve_dataset_path(
    yaml_path: Path, root: Path, value: object, label: str
) -> Path:
    text = _text(value, label)
    if text.endswith(".txt"):
        raise DatasetContractError(
            f"{label} must reference a direct formal image directory"
        )
    configured = Path(text).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (root / configured).resolve()
    )


def _label_directory(root: Path, image_directory: Path, label: str) -> Path:
    try:
        relative = image_directory.relative_to(root)
    except ValueError as error:
        raise DatasetContractError(f"{label} escapes dataset root") from error
    parts = list(relative.parts)
    if "images" not in parts:
        raise DatasetContractError(f"{label} has no images path segment")
    parts[parts.index("images")] = "labels"
    return root.joinpath(*parts)


def _load_dataset_layout(data_yaml: Path) -> _DatasetLayout:
    payload = _load_yaml_mapping(data_yaml)
    configured_root = Path(_text(payload.get("path"), f"{data_yaml}:path")).expanduser()
    root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (data_yaml.parent / configured_root).resolve()
    )
    train_images = _resolve_dataset_path(
        data_yaml, root, payload.get("train"), f"{data_yaml}:train"
    )
    val_images = _resolve_dataset_path(
        data_yaml, root, payload.get("val"), f"{data_yaml}:val"
    )
    train_labels = _label_directory(root, train_images, f"{data_yaml}:train")
    val_labels = _label_directory(root, val_images, f"{data_yaml}:val")
    for directory in (train_images, val_images, train_labels, val_labels):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    raw_names = payload.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise DatasetContractError(f"{data_yaml}:names must be a list or mapping")
    raw_shape = payload.get("kpt_shape")
    kpt_shape = (
        tuple(int(value) for value in raw_shape)
        if isinstance(raw_shape, list) and len(raw_shape) == 2
        else None
    )
    raw_flip = payload.get("flip_idx")
    flip_idx = (
        tuple(int(value) for value in raw_flip) if isinstance(raw_flip, list) else None
    )
    return _DatasetLayout(
        yaml=data_yaml,
        root=root,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        names=names,
        kpt_shape=kpt_shape,
        flip_idx=flip_idx,
    )


def _indexed_files(directory: Path, *, images: bool) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = (
        (path for path in directory.iterdir() if path.is_file() or path.is_symlink())
        if images
        else directory.glob("*.txt")
    )
    selected: dict[str, Path] = {}
    for path in paths:
        if images and path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in selected:
            raise ValueError(f"duplicate stem {path.stem!r} in {directory}")
        selected[path.stem] = path
    return selected


def _read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw.split()
        if not fields:
            continue
        if any(not _is_finite_number(value) for value in fields):
            raise ValueError(f"non-numeric label at {path}:{line_number}")
        rows.append(fields)
    return rows


def _is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _source_group(stem: str) -> str:
    return stem.split(".rf.", maxsplit=1)[0]


def _audit_split(
    pose: _DatasetLayout,
    detect: _DatasetLayout,
    split: str,
) -> tuple[SplitStats, int, int, set[str]]:
    pose_images = _indexed_files(pose.images(split), images=True)
    pose_labels = _indexed_files(pose.labels(split), images=False)
    detect_images = _indexed_files(detect.images(split), images=True)
    detect_labels = _indexed_files(detect.labels(split), images=False)
    expected = set(pose_images)
    for label, actual in (
        ("Pose labels", set(pose_labels)),
        ("Detect images", set(detect_images)),
        ("Detect labels", set(detect_labels)),
    ):
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(
                f"{split} {label} mismatch; missing={missing}, extra={extra}"
            )

    mismatches = 0
    broken_links = 0
    instances = 0
    empty = 0
    class_counts = {0: 0, 1: 0}
    negative_keypoints = 0
    for stem in sorted(expected):
        if (
            not pose_images[stem].resolve().is_file()
            or not detect_images[stem].resolve().is_file()
        ):
            broken_links += 1
        pose_rows = _read_rows(pose_labels[stem])
        detect_rows = _read_rows(detect_labels[stem])
        if len(pose_rows) != len(detect_rows):
            mismatches += 1
            continue
        if not pose_rows:
            empty += 1
        instances += len(pose_rows)
        for pose_row, detect_row in zip(pose_rows, detect_rows, strict=True):
            if len(pose_row) != 11:
                raise ValueError(
                    f"expected 11-column Pose label in {pose_labels[stem]}"
                )
            if len(detect_row) != 5:
                raise ValueError(
                    f"expected 5-column Detect label in {detect_labels[stem]}"
                )
            class_id = int(float(pose_row[0]))
            if class_id not in class_counts:
                raise ValueError(f"unexpected class {class_id} in {pose_labels[stem]}")
            class_counts[class_id] += 1
            if pose_row[:5] != detect_row:
                mismatches += 1
            keypoint_coordinates = (pose_row[5], pose_row[6], pose_row[8], pose_row[9])
            if any(float(value) < 0 for value in keypoint_coordinates):
                negative_keypoints += 1
    stats = SplitStats(
        images=len(pose_images),
        labels=len(pose_labels),
        instances=instances,
        empty_labels=empty,
        ball_instances=class_counts[0],
        bat_instances=class_counts[1],
        negative_keypoint_rows=negative_keypoints,
    )
    return stats, mismatches, broken_links, {_source_group(stem) for stem in expected}


def _audit_contract(contract: CanonicalBBAT5) -> BBT5AuditReport:
    train, train_mismatches, train_broken, train_groups = _audit_split(
        contract.pose, contract.detect, "train"
    )
    valid, valid_mismatches, valid_broken, valid_groups = _audit_split(
        contract.pose, contract.detect, "val"
    )
    return BBT5AuditReport(
        train=train,
        valid=valid,
        derivation_mismatches=train_mismatches + valid_mismatches,
        broken_image_links=train_broken + valid_broken,
        source_group_overlap=tuple(sorted(train_groups & valid_groups)),
    )


def audit_bbt5(
    registry: str | Path = DEFAULT_BBAT5_REGISTRY,
) -> BBT5AuditReport:
    """Validate registry lineage and prove paired Pose/Detect task views."""

    return CanonicalBBAT5.load(registry).audit()


def _write_generated(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"generated file differs from requested content: {path}"
            )
        return
    path.write_text(content, encoding="utf-8")


def prepare_bbt5_view(
    registry: str | Path,
    destination: str | Path,
    *,
    task: Task | str = Task.POSE,
) -> PreparedDataset:
    """Create a symlink-only runtime view from the canonical registry."""

    selected = Task(task)
    contract = CanonicalBBAT5.load(registry)
    source = contract.layout(selected)
    audit = _audit_contract(contract)
    if not audit.derivation_exact or audit.broken_image_links:
        raise DatasetContractError(f"canonical task views failed audit: {audit}")
    if audit.source_group_overlap:
        raise DatasetContractError(
            f"canonical split leaks source groups: {audit.source_group_overlap}"
        )
    if audit.train.negative_keypoint_rows or audit.valid.negative_keypoint_rows:
        raise DatasetContractError(
            "canonical Pose labels still contain negative coordinates"
        )

    target = Path(destination).expanduser().resolve()
    if target == contract.root or contract.root in target.parents:
        raise ValueError("destination must not be the source or a child of the source")
    total_images = 0
    total_labels = 0
    split_groups: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        source_images = _indexed_files(source.images(split), images=True)
        source_labels = _indexed_files(source.labels(split), images=False)
        if set(source_images) != set(source_labels):
            raise ValueError(f"{split} source images and labels are not paired")
        image_dir = target / split / "images"
        label_dir = target / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for stem in sorted(source_images):
            image_source = source_images[stem].resolve()
            image_target = image_dir / source_images[stem].name
            if image_target.exists() or image_target.is_symlink():
                if (
                    not image_target.is_symlink()
                    or image_target.resolve() != image_source
                ):
                    raise FileExistsError(
                        f"unexpected generated image entry: {image_target}"
                    )
            else:
                image_target.symlink_to(image_source)
            label_source = source_labels[stem].resolve()
            label_target = label_dir / label_source.name
            if label_target.exists() or label_target.is_symlink():
                if (
                    not label_target.is_symlink()
                    or label_target.resolve() != label_source
                ):
                    raise FileExistsError(
                        f"unexpected generated label entry: {label_target}"
                    )
            else:
                label_target.symlink_to(label_source)
        total_images += len(source_images)
        total_labels += len(source_labels)
        split_counts[split] = len(source_images)
        split_groups[split] = {_source_group(stem) for stem in source_images}

    data = {
        "path": str(target),
        "train": "train/images",
        "val": "val/images",
        "names": dict(source.names),
        "license": "CC BY 4.0",
        "dataset_id": contract.dataset_id,
        "source_registry": str(contract.registry),
        "source_yaml": str(source.yaml),
    }
    if selected is Task.POSE:
        data.update(
            {
                "kpt_shape": list(source.kpt_shape or ()),
                "flip_idx": list(source.flip_idx or ()),
            }
        )
    yaml_path = target / "data.yaml"
    _write_generated(
        yaml_path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    )
    manifest_payload = {
        "schema_version": 2,
        "dataset_id": contract.dataset_id,
        "task": selected.value,
        "source_registry": str(contract.registry),
        "source_yaml": str(source.yaml),
        "source_yaml_sha256": _file_sha256(source.yaml),
        "images": total_images,
        "labels": total_labels,
        "split_counts": split_counts,
        "source_patched_coordinates": contract.patched_coordinates,
        "source_group_overlap": sorted(split_groups["train"] & split_groups["val"]),
        "storage": "symlink-only-runtime-view",
    }
    manifest_path = target / "manifest.json"
    _write_generated(
        manifest_path, json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    )
    return PreparedDataset(
        dataset_id=contract.dataset_id,
        root=target,
        yaml=yaml_path,
        manifest=manifest_path,
        source_yaml=source.yaml,
        images=total_images,
        labels=total_labels,
        source_patched_coordinates=contract.patched_coordinates,
    )


def prepare_coco_detect_subset(
    source_yaml: str | Path,
    destination: str | Path,
    *,
    limit: int = 128,
) -> PreparedDetectSubset:
    """Create an idempotent, local-cache COCO subset without writing to its source."""

    if limit < 1:
        raise ValueError("limit must be positive")
    yaml_path = Path(source_yaml).expanduser().resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{yaml_path} must contain a mapping")
    configured_root = Path(str(payload.get("path", ""))).expanduser()
    source_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (yaml_path.parent / configured_root).resolve()
    )
    train_value = payload.get("train")
    if not isinstance(train_value, str) or not train_value.endswith(".txt"):
        raise ValueError("COCO subset source must use a train text manifest")
    train_list = (source_root / train_value).resolve()
    raw_entries = [
        line.strip()
        for line in train_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit > len(raw_entries):
        raise ValueError(f"limit {limit} exceeds source image count {len(raw_entries)}")

    target = Path(destination).expanduser().resolve()
    if target == source_root or source_root in target.parents:
        raise ValueError("destination must not be the source or a child of the source")
    image_dir = target / "train" / "images"
    label_dir = target / "train" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = 0
    selected_stems: list[str] = []
    for raw in raw_entries[:limit]:
        image_source = Path(raw).expanduser()
        if not image_source.is_absolute():
            image_source = (source_root / image_source).resolve()
        if not image_source.is_file():
            raise FileNotFoundError(image_source)
        if image_source.stem in selected_stems:
            raise ValueError(f"duplicate image stem in subset: {image_source.stem}")
        selected_stems.append(image_source.stem)
        try:
            images_index = image_source.parts.index("images")
        except ValueError as error:
            raise ValueError(
                f"image path has no images segment: {image_source}"
            ) from error
        label_source = Path(
            *image_source.parts[:images_index],
            "labels",
            *image_source.parts[images_index + 1 :],
        ).with_suffix(".txt")

        image_target = image_dir / image_source.name
        if image_target.exists() or image_target.is_symlink():
            if not image_target.is_symlink() or image_target.resolve() != image_source:
                raise FileExistsError(
                    f"unexpected generated image entry: {image_target}"
                )
        else:
            image_target.symlink_to(image_source)

        label_target = label_dir / f"{image_source.stem}.txt"
        if label_source.is_file():
            if label_target.exists() or label_target.is_symlink():
                if (
                    not label_target.is_symlink()
                    or label_target.resolve() != label_source
                ):
                    raise FileExistsError(
                        f"unexpected generated label entry: {label_target}"
                    )
            else:
                label_target.symlink_to(label_source)
        else:
            backgrounds += 1
            _write_generated(label_target, "")

    data = {
        "path": str(target),
        "train": "train/images",
        "val": "train/images",
        "names": payload["names"],
        "license": "COCO",
        "source": str(yaml_path),
    }
    subset_yaml = target / "data.yaml"
    _write_generated(
        subset_yaml,
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )
    manifest_payload = {
        "schema_version": 1,
        "task": Task.DETECT.value,
        "source": str(yaml_path),
        "source_images": len(raw_entries),
        "selected_images": len(selected_stems),
        "labels": len(selected_stems),
        "backgrounds": backgrounds,
        "first_stem": selected_stems[0],
        "last_stem": selected_stems[-1],
    }
    manifest = target / "manifest.json"
    _write_generated(
        manifest,
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
    )
    return PreparedDetectSubset(
        root=target,
        yaml=subset_yaml,
        manifest=manifest,
        images=len(selected_stems),
        labels=len(selected_stems),
        backgrounds=backgrounds,
        source_images=len(raw_entries),
    )
