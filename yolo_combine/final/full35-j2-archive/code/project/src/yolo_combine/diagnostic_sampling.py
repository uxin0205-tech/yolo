"""Reproducible training-only subsets that cannot become formal evidence.

The canonical BBAT5 assignment remains immutable.  This module only writes a
cache-isolated symlink runtime view, selection manifest, and audit list.
Validation always keeps the complete canonical validation split.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .data import IMAGE_SUFFIXES, PreparedDataset


DIAGNOSTIC_MARKER_NAME = "diagnostic-run.json"


@dataclass(frozen=True)
class DiagnosticSamplingPolicy:
    """A fixed, reproducible policy for non-formal training diagnostics."""

    fraction: float = 0.3
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("diagnostic fraction must be in (0, 1)")
        if self.seed < 0:
            raise ValueError("diagnostic seed cannot be negative")

    @property
    def tag(self) -> str:
        fraction = f"{self.fraction:.6f}".rstrip("0").rstrip(".")
        return f"f{fraction.replace('.', 'p')}-seed{self.seed}"


@dataclass(frozen=True)
class DiagnosticDatasetView:
    """The complete provenance of one training-only diagnostic view."""

    root: Path
    yaml: Path
    train_list: Path
    manifest: Path
    source_runtime_root: Path
    source_runtime_manifest: Path
    fraction: float
    seed: int
    strategy: str
    full_train_images: int
    selected_train_images: int
    full_train_groups: int
    selected_train_groups: int
    full_validation_images: int
    full_train_instances: int
    selected_train_instances: int
    full_ball_instances: int
    selected_ball_instances: int
    full_bat_instances: int
    selected_bat_instances: int
    full_empty_labels: int
    selected_empty_labels: int
    selected_train_sha256: str
    formal_eligible: bool = False


def _source_group(path: Path) -> str:
    return path.stem.split(".rf.", maxsplit=1)[0]


def _image_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = tuple(
        sorted(
            (
                path.absolute()
                for path in directory.iterdir()
                if (path.is_file() or path.is_symlink())
                and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name,
        )
    )
    if not paths:
        raise ValueError(f"no images found in {directory}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError(f"duplicate image stems in {directory}")
    return paths


def _select_source_groups(
    paths: tuple[Path, ...], policy: DiagnosticSamplingPolicy
) -> tuple[tuple[Path, ...], int, int]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(_source_group(path), []).append(path)
    group_names = sorted(groups)
    selected_group_count = max(1, round(len(group_names) * policy.fraction))
    shuffled = list(group_names)
    random.Random(policy.seed).shuffle(shuffled)
    chosen = set(shuffled[:selected_group_count])
    selected = tuple(
        sorted(
            (path for group in chosen for path in groups[group]),
            key=lambda path: path.name,
        )
    )
    if not selected:
        raise AssertionError("diagnostic group selection produced no images")
    if {_source_group(path) for path in selected} != chosen:
        raise AssertionError("diagnostic selection split a BBAT5 source group")
    return selected, len(groups), len(chosen)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _pose_label_stats(paths: tuple[Path, ...], label_dir: Path) -> dict[str, int]:
    counts = {"ball": 0, "bat": 0, "empty": 0}
    for image_path in paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        rows = [
            line.split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            counts["empty"] += 1
        for row in rows:
            if len(row) != 11:
                raise ValueError(f"expected 11-column Pose label in {label_path}")
            class_id = int(float(row[0]))
            if class_id == 0:
                counts["ball"] += 1
            elif class_id == 1:
                counts["bat"] += 1
            else:
                raise ValueError(f"unexpected Pose class {class_id} in {label_path}")
    return counts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_generated(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"generated diagnostic file differs from requested content: {path}"
            )
        return
    path.write_text(content, encoding="utf-8")


def _link_pose_records(
    paths: tuple[Path, ...],
    *,
    source_root: Path,
    target_root: Path,
    split: str,
) -> tuple[Path, ...]:
    image_dir = target_root / split / "images"
    label_dir = target_root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    expected_images = {path.name for path in paths}
    expected_labels = {f"{path.stem}.txt" for path in paths}
    actual_images = {
        path.name
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    actual_labels = {path.name for path in label_dir.glob("*.txt")}
    unexpected_images = sorted(actual_images - expected_images)
    unexpected_labels = sorted(actual_labels - expected_labels)
    if unexpected_images or unexpected_labels:
        raise FileExistsError(
            "diagnostic runtime view has unexpected records; "
            f"images={unexpected_images[:5]}, labels={unexpected_labels[:5]}"
        )

    linked: list[Path] = []
    for source_image in paths:
        source_label = (
            source_root / split / "labels" / f"{source_image.stem}.txt"
        )
        if not source_label.is_file():
            raise FileNotFoundError(source_label)
        target_image = image_dir / source_image.name
        target_label = label_dir / source_label.name
        for source, target in (
            (source_image.resolve(), target_image),
            (source_label.resolve(), target_label),
        ):
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != source:
                    raise FileExistsError(
                        f"unexpected diagnostic runtime entry: {target}"
                    )
            else:
                target.symlink_to(source)
        linked.append(target_image.absolute())
    return tuple(linked)


def prepare_pose_diagnostic_view(
    prepared: PreparedDataset,
    destination: str | Path,
    *,
    policy: DiagnosticSamplingPolicy,
) -> DiagnosticDatasetView:
    """Materialize a fixed BBAT5 training list while retaining full validation.

    Images and labels remain owned by the canonical runtime view.  Selection is
    made by source group so brightness siblings are never split across the
    selected and unselected training records.
    """

    if prepared.dataset_id != "bbat5-v1":
        raise ValueError("diagnostic Pose sampling only accepts canonical bbat5-v1")
    source_payload = yaml.safe_load(prepared.yaml.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise TypeError(f"{prepared.yaml} must contain a mapping")
    runtime_manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    if not isinstance(runtime_manifest, dict):
        raise TypeError(f"{prepared.manifest} must contain a mapping")
    split_counts = runtime_manifest.get("split_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("runtime manifest has no split_counts mapping")

    train_paths = _image_paths(prepared.root / "train" / "images")
    validation_paths = _image_paths(prepared.root / "val" / "images")
    if int(split_counts.get("train", -1)) != len(train_paths):
        raise ValueError("runtime train count differs from its canonical manifest")
    if int(split_counts.get("val", -1)) != len(validation_paths):
        raise ValueError("runtime validation count differs from its canonical manifest")

    selected, full_groups, selected_groups = _select_source_groups(
        train_paths, policy
    )
    label_dir = prepared.root / "train" / "labels"
    full_label_stats = _pose_label_stats(train_paths, label_dir)
    selected_label_stats = _pose_label_stats(selected, label_dir)
    target = Path(destination).expanduser().resolve()
    if target == prepared.root or prepared.root in target.parents:
        raise ValueError(
            "diagnostic metadata must not be written inside the runtime view"
        )

    linked_train = _link_pose_records(
        selected,
        source_root=prepared.root,
        target_root=target,
        split="train",
    )
    _link_pose_records(
        validation_paths,
        source_root=prepared.root,
        target_root=target,
        split="val",
    )
    train_content = "".join(f"{path}\n" for path in linked_train)
    train_list = target / "train-selected.txt"
    _write_generated(train_list, train_content)

    data: dict[str, Any] = {
        "path": str(target),
        "train": "train/images",
        "val": "val/images",
        "names": source_payload.get("names"),
        "kpt_shape": source_payload.get("kpt_shape"),
        "flip_idx": source_payload.get("flip_idx"),
        "license": source_payload.get("license", "CC BY 4.0"),
        "dataset_id": prepared.dataset_id,
        "diagnostic_only": True,
        "formal_eligible": False,
        "source_runtime_manifest": str(prepared.manifest),
    }
    if (
        data["names"] is None
        or data["kpt_shape"] is None
        or data["flip_idx"] is None
    ):
        raise ValueError("runtime Pose YAML is missing names/kpt_shape/flip_idx")
    yaml_path = target / "pose-diagnostic.yaml"
    _write_generated(
        yaml_path,
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )

    manifest_payload = {
        "schema_version": 1,
        "dataset_id": prepared.dataset_id,
        "task": "pose",
        "purpose": "training_diagnostic_only",
        "formal_eligible": False,
        "selection": {
            "fraction": policy.fraction,
            "seed": policy.seed,
            "strategy": "seeded-random-source-group",
            "full_train_images": len(train_paths),
            "selected_train_images": len(selected),
            "full_train_groups": full_groups,
            "selected_train_groups": selected_groups,
            "target_group_fraction": policy.fraction,
            "full_train_instances": (
                full_label_stats["ball"] + full_label_stats["bat"]
            ),
            "selected_train_instances": (
                selected_label_stats["ball"] + selected_label_stats["bat"]
            ),
            "full_class_instances": {
                "ball": full_label_stats["ball"],
                "bat": full_label_stats["bat"],
            },
            "selected_class_instances": {
                "ball": selected_label_stats["ball"],
                "bat": selected_label_stats["bat"],
            },
            "full_empty_labels": full_label_stats["empty"],
            "selected_empty_labels": selected_label_stats["empty"],
        },
        "validation": {
            "fraction": 1.0,
            "images": len(validation_paths),
            "formal_split_unchanged": True,
        },
        "source_runtime_root": str(prepared.root),
        "source_runtime_manifest": str(prepared.manifest),
        "source_runtime_manifest_sha256": _file_sha256(prepared.manifest),
        "selected_train_list": str(train_list),
        "selected_train_sha256": _sha256_text(train_content),
        "storage": (
            "symlink-only-diagnostic-runtime-view; "
            "images-and-labels-remain-canonical"
        ),
    }
    manifest_path = target / "manifest.json"
    _write_generated(
        manifest_path,
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
    )
    return DiagnosticDatasetView(
        root=target,
        yaml=yaml_path,
        train_list=train_list,
        manifest=manifest_path,
        source_runtime_root=prepared.root,
        source_runtime_manifest=prepared.manifest,
        fraction=policy.fraction,
        seed=policy.seed,
        strategy="seeded-random-source-group",
        full_train_images=len(train_paths),
        selected_train_images=len(selected),
        full_train_groups=full_groups,
        selected_train_groups=selected_groups,
        full_validation_images=len(validation_paths),
        full_train_instances=full_label_stats["ball"] + full_label_stats["bat"],
        selected_train_instances=(
            selected_label_stats["ball"] + selected_label_stats["bat"]
        ),
        full_ball_instances=full_label_stats["ball"],
        selected_ball_instances=selected_label_stats["ball"],
        full_bat_instances=full_label_stats["bat"],
        selected_bat_instances=selected_label_stats["bat"],
        full_empty_labels=full_label_stats["empty"],
        selected_empty_labels=selected_label_stats["empty"],
        selected_train_sha256=_sha256_text(train_content),
    )


def mark_diagnostic_run(
    run_dir: str | Path,
    *,
    view: DiagnosticDatasetView,
    stage: str,
) -> Path:
    """Write the marker used to prevent a diagnostic run becoming formal input."""

    target = Path(run_dir).expanduser().resolve() / DIAGNOSTIC_MARKER_NAME
    payload = {
        "schema_version": 1,
        "purpose": "training_diagnostic_only",
        "formal_eligible": False,
        "stage": stage,
        "dataset": {
            **asdict(view),
            "root": str(view.root),
            "yaml": str(view.yaml),
            "train_list": str(view.train_list),
            "manifest": str(view.manifest),
            "source_runtime_root": str(view.source_runtime_root),
            "source_runtime_manifest": str(view.source_runtime_manifest),
        },
    }
    _write_generated(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def diagnostic_marker_for_checkpoint(checkpoint: str | Path) -> Path | None:
    """Return an adjacent diagnostic marker when a checkpoint is non-formal."""

    path = Path(checkpoint).expanduser().resolve()
    for parent in (path.parent, *path.parents):
        marker = parent / DIAGNOSTIC_MARKER_NAME
        if marker.is_file():
            return marker
    return None


def checkpoint_is_diagnostic(checkpoint: str | Path) -> bool:
    """Fail closed for marked runs and interrupted runs in a diagnostic tree."""

    path = Path(checkpoint).expanduser().resolve()
    return (
        "diagnostic" in path.parts
        or diagnostic_marker_for_checkpoint(path) is not None
    )
