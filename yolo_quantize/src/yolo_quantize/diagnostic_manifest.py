"""Immutable CPU-only calibration and probe manifest construction."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_WORKSPACE = Path("/home/uxin/yolo")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_shuffle(values: list[Any], *, seed: int, namespace: str) -> None:
    material = f"{seed}:{namespace}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    random.Random(derived).shuffle(values)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"dataset YAML must be a mapping: {path}")
    return payload


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"image path has no images segment: {image}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _label_summary(path: Path) -> tuple[tuple[int, ...], int]:
    if not path.is_file():
        return (), 0
    classes: set[int] = set()
    pose_rows = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        classes.add(int(float(fields[0])))
        if len(fields) >= 11:
            pose_rows += 1
    return tuple(sorted(classes)), pose_rows


def _sample_evidence(image: Path, *, source_group: str | None = None) -> dict[str, Any]:
    label = _label_path(image)
    classes, pose_rows = _label_summary(label)
    return {
        "image": str(image),
        "image_sha256": _sha256(image),
        "label": str(label),
        "label_sha256": _sha256(label) if label.is_file() else None,
        "class_ids": list(classes),
        "pose_rows": pose_rows,
        "source_group": source_group,
    }


def _selection_digest(samples: list[dict[str, Any]]) -> str:
    return _json_sha256({"samples": samples})


def _coverage(samples: list[dict[str, Any]], *, person: bool) -> dict[str, Any]:
    class_ids = sorted(
        {class_id for sample in samples for class_id in sample["class_ids"]}
    )
    payload: dict[str, Any] = {
        "class_ids": class_ids,
        "labeled_images": sum(bool(sample["class_ids"]) for sample in samples),
        "pose_rows": sum(int(sample["pose_rows"]) for sample in samples),
    }
    if person:
        payload["person_images"] = sum(0 in sample["class_ids"] for sample in samples)
    return payload


@dataclass(frozen=True)
class DiagnosticManifestSpec:
    """Fixed diagnostic-only sampling contract confirmed for Full35."""

    seed: int = 20260830
    calibration_per_task: int = 32
    probe_per_task: int = 64
    coco_yaml: Path = _WORKSPACE / "coco2017.yaml"
    bbat_registry: Path = _WORKSPACE / "configs/datasets/bbat5-v1.yaml"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("diagnostic seed must be non-negative")
        if self.calibration_per_task != 32 or self.probe_per_task != 64:
            raise ValueError(
                "diagnostic contract is fixed at 32 calibration and 64 probe"
            )


@dataclass(frozen=True)
class DiagnosticManifest:
    """Manifest payload and canonical serialization digest."""

    payload: dict[str, Any]
    sha256: str
    source_path: Path | None = None

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        verify_files: bool,
    ) -> DiagnosticManifest:
        """Load a manifest only when its identity and fixed sample contract match."""

        source_path = Path(path).expanduser().resolve()
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("diagnostic manifest must be a mapping")
        payload = dict(raw)
        claimed_sha256 = payload.pop("manifest_sha256", None)
        actual_sha256 = _json_sha256(payload)
        if claimed_sha256 != actual_sha256:
            raise ValueError(
                "diagnostic manifest SHA-256 mismatch: "
                f"{claimed_sha256} != {actual_sha256}"
            )
        if (
            payload.get("kind") != "full35_quantization_diagnostic_manifest"
            or payload.get("diagnostic_only") is not True
            or payload.get("formal_training") is not False
        ):
            raise ValueError("diagnostic manifest identity contract is invalid")
        counts = payload.get("counts")
        if counts != {"calibration_per_task": 32, "probe_per_task": 64}:
            raise ValueError("diagnostic manifest sample counts changed")
        sources = payload.get("sources")
        bbat5_source = sources.get("bbat5") if isinstance(sources, dict) else None
        if (
            not isinstance(sources, dict)
            or not isinstance(bbat5_source, dict)
            or bbat5_source.get("dataset_id") != "bbat5-v1"
            or bbat5_source.get("assignment_changed") is not False
        ):
            raise ValueError("diagnostic manifest BBAT5 contract is invalid")
        selections = payload.get("selections")
        if not isinstance(selections, dict):
            raise TypeError("diagnostic manifest selections must be a mapping")
        for selection, expected_count in (("calibration", 32), ("probe", 64)):
            group = selections.get(selection)
            if not isinstance(group, dict):
                raise TypeError(f"diagnostic {selection} selection must be a mapping")
            for task in ("coco_detect", "bbat_pose"):
                task_payload = group.get(task)
                samples = (
                    task_payload.get("samples")
                    if isinstance(task_payload, dict)
                    else None
                )
                if not isinstance(samples, list) or len(samples) != expected_count:
                    raise ValueError(
                        f"diagnostic {selection}/{task} sample count changed"
                    )
                for sample in samples:
                    if not isinstance(sample, dict) or not isinstance(
                        sample.get("image"), str
                    ):
                        raise TypeError(
                            f"diagnostic {selection}/{task} sample is invalid"
                        )
                    if verify_files:
                        image = Path(sample["image"])
                        if not image.is_file() or _sha256(image) != sample.get(
                            "image_sha256"
                        ):
                            raise RuntimeError(
                                f"diagnostic image evidence changed: {image}"
                            )
                        label_value = sample.get("label")
                        label_sha256 = sample.get("label_sha256")
                        if label_sha256 is not None:
                            label = Path(str(label_value))
                            if not label.is_file() or _sha256(label) != label_sha256:
                                raise RuntimeError(
                                    f"diagnostic label evidence changed: {label}"
                                )
        return cls(
            payload=payload,
            sha256=actual_sha256,
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload, "manifest_sha256": self.sha256}

    def paths(self, selection: str, task: str) -> tuple[Path, ...]:
        """Return the immutable ordered image paths for one manifest stratum."""

        if selection not in {"calibration", "probe"}:
            raise ValueError("diagnostic selection must be calibration or probe")
        if task not in {"coco_detect", "bbat_pose"}:
            raise ValueError("diagnostic task must be coco_detect or bbat_pose")
        samples = self.payload["selections"][selection][task]["samples"]
        return tuple(Path(sample["image"]) for sample in samples)


class DiagnosticManifestBuilder:
    """Build deterministic views without changing a canonical assignment."""

    def _coco_selection(
        self,
        *,
        yaml_path: Path,
        split_key: str,
        count: int,
        seed: int,
    ) -> dict[str, Any]:
        config = _read_yaml(yaml_path)
        names = config.get("names")
        if not isinstance(names, dict) or names.get(0) != "person":
            raise ValueError("COCO class 0 must be person")
        root_value = Path(str(config["path"])).expanduser()
        root = root_value if root_value.is_absolute() else yaml_path.parent / root_value
        split_value = Path(str(config[split_key]))
        list_path = split_value if split_value.is_absolute() else root / split_value
        candidates: list[Path] = []
        for raw in list_path.read_text(encoding="utf-8").splitlines():
            relative = raw.strip().removeprefix("./")
            if not relative:
                continue
            image = Path(relative)
            image = image if image.is_absolute() else root / image
            if image.is_file():
                candidates.append(image.resolve())
        _stable_shuffle(candidates, seed=seed, namespace=f"coco:{split_key}")
        if len(candidates) < count:
            raise RuntimeError(f"COCO {split_key} has fewer than {count} images")
        selected = candidates[:count]
        evidence = [_sample_evidence(image) for image in selected]
        if not any(0 in sample["class_ids"] for sample in evidence):
            replacement = next(
                (
                    image
                    for image in candidates[count:]
                    if 0 in _label_summary(_label_path(image))[0]
                ),
                None,
            )
            if replacement is None:
                raise RuntimeError(f"COCO {split_key} provides no person coverage")
            evidence[-1] = _sample_evidence(replacement)
        return {
            "split": "train2017" if split_key == "train" else "val2017",
            "source_list": str(list_path.resolve()),
            "source_list_sha256": _sha256(list_path),
            "selection_sha256": _selection_digest(evidence),
            "samples": evidence,
            "coverage": _coverage(evidence, person=True),
        }

    def _bbat_selection(
        self,
        *,
        pose_root: Path,
        split_name: str,
        count: int,
        seed: int,
    ) -> dict[str, Any]:
        short_split = "train" if split_name == "formal-train" else "val"
        split_path = pose_root / "splits" / f"formal-{short_split}.txt"
        grouped: dict[str, list[Path]] = defaultdict(list)
        for raw in split_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            image = Path(raw.strip()).expanduser()
            expected_parent = pose_root / "images" / short_split
            if not image.is_file() or image.parent != expected_parent:
                raise RuntimeError(f"non-canonical BBAT5 path in {split_path}: {image}")
            source_group = image.name.split(".rf.", maxsplit=1)[0]
            grouped[source_group].append(image)
        group_ids = sorted(grouped)
        _stable_shuffle(group_ids, seed=seed, namespace=f"bbat5:{split_name}")
        if len(group_ids) < count:
            raise RuntimeError(f"BBAT5 {split_name} has fewer than {count} groups")
        selected_groups = group_ids[:count]
        evidence = [
            _sample_evidence(
                min(grouped[source_group]),
                source_group=source_group,
            )
            for source_group in selected_groups
        ]
        coverage = _coverage(evidence, person=False)
        if coverage["class_ids"] != [0, 1] or coverage["pose_rows"] <= 0:
            raise RuntimeError(f"BBAT5 {split_name} diagnostic coverage is incomplete")
        return {
            "split": split_name,
            "source_list": str(split_path.resolve()),
            "source_list_sha256": _sha256(split_path),
            "source_group_policy": "one_canonical_exemplar_per_rf_group",
            "selection_sha256": _selection_digest(evidence),
            "samples": evidence,
            "coverage": coverage,
        }

    def build(self, spec: DiagnosticManifestSpec) -> DiagnosticManifest:
        coco_yaml = spec.coco_yaml.expanduser().resolve()
        registry_path = spec.bbat_registry.expanduser().resolve()
        registry = _read_yaml(registry_path)
        if registry.get("dataset_id") != "bbat5-v1":
            raise ValueError("diagnostic BBAT5 registry must be bbat5-v1")
        if registry.get("status") != "canonical":
            raise ValueError("diagnostic BBAT5 registry is not canonical")
        pose_yaml = Path(registry["tasks"]["pose"]["data_yaml"]).resolve()
        expected_pose_sha = str(registry["tasks"]["pose"]["data_yaml_sha256"])
        if _sha256(pose_yaml) != expected_pose_sha:
            raise RuntimeError("Canonical BBAT5 Pose YAML SHA-256 mismatch")
        pose_config = _read_yaml(pose_yaml)
        pose_root = Path(str(pose_config["path"])).resolve()

        calibration_coco = self._coco_selection(
            yaml_path=coco_yaml,
            split_key="train",
            count=spec.calibration_per_task,
            seed=spec.seed,
        )
        probe_coco = self._coco_selection(
            yaml_path=coco_yaml,
            split_key="val",
            count=spec.probe_per_task,
            seed=spec.seed,
        )
        calibration_bbat = self._bbat_selection(
            pose_root=pose_root,
            split_name="formal-train",
            count=spec.calibration_per_task,
            seed=spec.seed,
        )
        probe_bbat = self._bbat_selection(
            pose_root=pose_root,
            split_name="formal-val",
            count=spec.probe_per_task,
            seed=spec.seed,
        )
        train_groups = {item["source_group"] for item in calibration_bbat["samples"]}
        val_groups = {item["source_group"] for item in probe_bbat["samples"]}
        if train_groups & val_groups:
            raise RuntimeError("BBAT5 diagnostic source groups cross formal splits")

        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "full35_quantization_diagnostic_manifest",
            "diagnostic_only": True,
            "formal_training": False,
            "gpu_used": False,
            "seed": spec.seed,
            "counts": {
                "calibration_per_task": spec.calibration_per_task,
                "probe_per_task": spec.probe_per_task,
            },
            "sources": {
                "coco2017": {
                    "yaml": str(coco_yaml),
                    "yaml_sha256": _sha256(coco_yaml),
                    "person_class_id": 0,
                },
                "bbat5": {
                    "dataset_id": "bbat5-v1",
                    "registry": str(registry_path),
                    "registry_sha256": _sha256(registry_path),
                    "pose_yaml": str(pose_yaml),
                    "pose_yaml_sha256": expected_pose_sha,
                    "assignment_changed": False,
                },
            },
            "selections": {
                "calibration": {
                    "coco_detect": calibration_coco,
                    "bbat_pose": calibration_bbat,
                },
                "probe": {
                    "coco_detect": probe_coco,
                    "bbat_pose": probe_bbat,
                },
            },
        }
        return DiagnosticManifest(payload=payload, sha256=_json_sha256(payload))
