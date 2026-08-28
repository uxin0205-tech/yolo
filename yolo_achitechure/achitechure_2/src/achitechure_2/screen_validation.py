"""Float20 的官方 Detect/Pose 驗證與固定閾值 F1 匯出。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator

from .candidate import ResolvedCandidate
from .full35_adapter import Full35Release
from .runtime_dataset import build_runtime_yolo_dataset


@dataclass(frozen=True)
class ThresholdSet:
    """由 C0-Control search-val 唯一決定的三個 confidence thresholds。"""

    detect_box: float
    pose_box: float
    pose_keypoints: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ThresholdSet:
        required = {"detect_box", "pose_box", "pose_keypoints"}
        if set(value) != required:
            raise ValueError(f"F1 thresholds 必須恰好包含 {sorted(required)}")
        result = cls(
            detect_box=float(value["detect_box"]),
            pose_box=float(value["pose_box"]),
            pose_keypoints=float(value["pose_keypoints"]),
        )
        if any(not 0.0 <= item <= 1.0 for item in asdict(result).values()):
            raise ValueError("F1 confidence threshold 必須介於 0 與 1")
        return result


@dataclass(frozen=True)
class ScreenValidationResult:
    epoch: int
    metrics: dict[str, Any]
    flat_metrics: dict[str, float]
    thresholds: ThresholdSet
    output_dir: Path


def _class_indices(metric: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(metric.ap_class_index).reshape(-1))


def _fixed_operating_point(
    metric: Any,
    *,
    names: Mapping[int, str],
    supports: np.ndarray,
    threshold: float | None,
) -> tuple[dict[str, Any], float]:
    px = np.asarray(metric.px, dtype=np.float64)
    p_curve = np.asarray(metric.p_curve, dtype=np.float64)
    r_curve = np.asarray(metric.r_curve, dtype=np.float64)
    f1_curve = np.asarray(metric.f1_curve, dtype=np.float64)
    indices = _class_indices(metric)
    if (
        px.ndim != 1
        or p_curve.shape != r_curve.shape
        or p_curve.shape != f1_curve.shape
        or p_curve.shape != (len(indices), len(px))
    ):
        raise ValueError("Ultralytics F1 curve shape 不符合鎖定的 8.4.90 契約")
    derived_from_c0 = threshold is None
    if threshold is None:
        position = int(np.nanargmax(np.nanmean(f1_curve, axis=0)))
        threshold = float(px[position])
    else:
        position = int(np.abs(px - float(threshold)).argmin())
        threshold = float(px[position])

    precision = p_curve[:, position]
    recall = r_curve[:, position]
    f1 = f1_curve[:, position]
    per_class: dict[str, dict[str, float | int]] = {}
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    eps = 1e-16
    for row, class_id in enumerate(indices):
        support = int(supports[class_id]) if class_id < len(supports) else 0
        tp = float(recall[row] * support)
        fp = float(tp / max(float(precision[row]), eps) - tp) if tp else 0.0
        fn = float(max(support - tp, 0.0))
        true_positive += tp
        false_positive += max(fp, 0.0)
        false_negative += fn
        per_class[str(class_id)] = {
            "name": str(names[class_id]),
            "support": support,
            "precision": float(precision[row]),
            "recall": float(recall[row]),
            "f1": float(f1[row]),
        }
    macro = float(np.nanmean(f1)) if len(f1) else 0.0
    denominator = 2.0 * true_positive + false_positive + false_negative
    micro = 2.0 * true_positive / denominator if denominator else 0.0
    return (
        {
            "confidence_threshold": threshold,
            "threshold_source": "c0_control_search_val",
            "threshold_derived_in_this_event": derived_from_c0,
            "macro_f1": macro,
            "micro_f1": float(micro),
            "micro_f1_method": "estimated_from_precision_recall_curves_and_supports",
            "per_class": per_class,
            "micro_counts_estimated_from_curves": {
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
            },
        },
        threshold,
    )


def _ap_payload(
    metric: Any,
    *,
    names: Mapping[int, str],
) -> dict[str, Any]:
    indices = _class_indices(metric)
    all_ap = np.asarray(metric.all_ap, dtype=np.float64)
    if all_ap.shape != (len(indices), 10):
        raise ValueError(f"AP table shape 漂移：{all_ap.shape}")
    per_class: dict[str, dict[str, float | str]] = {}
    for row, class_id in enumerate(indices):
        per_class[str(class_id)] = {
            "name": str(names[class_id]),
            "ap50": float(all_ap[row, 0]),
            "ap50_95": float(all_ap[row].mean()),
        }
    return {
        "map50": float(metric.map50),
        "map50_95": float(metric.map),
        "precision_at_ultralytics_best_f1": float(metric.mp),
        "recall_at_ultralytics_best_f1": float(metric.mr),
        "per_class": per_class,
    }


def _metric_payload(
    metric: Any,
    *,
    names: Mapping[int, str],
    supports: np.ndarray,
    threshold: float | None,
) -> tuple[dict[str, Any], float]:
    f1, resolved = _fixed_operating_point(
        metric,
        names=names,
        supports=supports,
        threshold=threshold,
    )
    return {"ap": _ap_payload(metric, names=names), "f1": f1}, resolved


def _flat(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in payload.items():
        key = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(value, Mapping):
            values.update(_flat(value, key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if np.isfinite(number):
                values[key] = number
    return values


class _RuntimeCacheValidatorMixin:
    """讓官方 validator 使用相同 graph/metrics，只改 label cache 位置。"""

    def __init__(self, *args: Any, label_cache_path: str | Path, **kwargs: Any) -> None:
        self._label_cache_path = Path(label_cache_path).expanduser().resolve()
        super().__init__(*args, **kwargs)

    def build_dataset(
        self,
        img_path: str,
        mode: str = "val",
        batch: int | None = None,
    ) -> Any:
        return build_runtime_yolo_dataset(
            self.args,
            img_path,
            batch or int(self.args.batch),
            self.data,
            label_cache_path=self._label_cache_path,
            mode=mode,
            rect=False,
            stride=self.stride,
        )


class _RuntimeDetectionValidator(_RuntimeCacheValidatorMixin, DetectionValidator):
    pass


class _RuntimePoseValidator(_RuntimeCacheValidatorMixin, PoseValidator):
    pass


class ScreenValidator:
    """以 graft 後的官方 graph 驗證共享候選；不把 search-val 當 formal val。"""

    def __init__(
        self,
        release: Full35Release,
        source: Any,
        resolved: ResolvedCandidate,
        *,
        detect_data: str | Path,
        pose_data: str | Path,
        output_root: str | Path,
        imgsz: int = 640,
        detect_batch: int = 16,
        pose_batch: int = 16,
        detect_workers: int = 4,
        pose_workers: int = 8,
        device: str = "0",
        pose_enabled: bool = True,
        runtime_cache_root: str | Path | None = None,
        scope: str = "screening_train_only_search_val",
        formal_split_used: bool = False,
    ) -> None:
        self.release = release
        self.source = source
        self.resolved = resolved
        self.detect_data = Path(detect_data).resolve()
        self.pose_data = Path(pose_data).resolve()
        self.output_root = Path(output_root).resolve()
        self.runtime_cache_root = (
            Path(runtime_cache_root).expanduser().resolve()
            if runtime_cache_root is not None
            else self.output_root / "runtime-cache"
        )
        self.imgsz = int(imgsz)
        self.detect_batch = int(detect_batch)
        self.pose_batch = int(pose_batch)
        self.detect_workers = int(detect_workers)
        self.pose_workers = int(pose_workers)
        self.device = str(device)
        self.pose_enabled = bool(pose_enabled)
        self.scope = str(scope)
        self.formal_split_used = bool(formal_split_used)
        if not self.scope:
            raise ValueError("validation scope 不得為空")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("validation imgsz 必須是 32 的倍數")
        if min(self.detect_batch, self.pose_batch) < 1:
            raise ValueError("validation batch 必須為正整數")

    def _args(
        self,
        *,
        task: str,
        data: Path,
        batch: int,
        workers: int,
    ) -> dict[str, Any]:
        return {
            "task": task,
            "data": str(data),
            "imgsz": self.imgsz,
            "batch": batch,
            "workers": workers,
            "device": self.device,
            "plots": False,
            "save_json": False,
            "compile": False,
            "rect": True,
            "split": "val",
            "mode": "val",
            "verbose": False,
        }

    def validate(
        self,
        shared_ema: torch.nn.Module,
        *,
        epoch: int,
        fixed_thresholds: ThresholdSet | None,
    ) -> ScreenValidationResult:
        if epoch < 0:
            raise ValueError("validation epoch 不得為負")
        materialized = self.release.materialize_validation_models(
            shared_ema,
            self.source,
            self.resolved,
            kind="float",
        )
        root = self.output_root / f"epoch-{epoch:04d}" / "float"
        detect_validator = _RuntimeDetectionValidator(
            save_dir=root / "detect",
            label_cache_path=self.runtime_cache_root / "detect-val.cache",
            args=self._args(
                task="detect",
                data=self.detect_data,
                batch=self.detect_batch,
                workers=self.detect_workers,
            ),
        )
        detect_raw = detect_validator(model=materialized.detect)
        if not isinstance(detect_raw, dict):
            raise TypeError("官方 Detect validator 未回傳 metrics mapping")

        detect_names = {int(key): str(value) for key, value in materialized.detect.names.items()}
        detect_support = np.asarray(detect_validator.metrics.nt_per_class)
        detect, detect_threshold = _metric_payload(
            detect_validator.metrics.box,
            names=detect_names,
            supports=detect_support,
            threshold=(fixed_thresholds.detect_box if fixed_thresholds else None),
        )
        pose_thresholds = (
            (fixed_thresholds.pose_box, fixed_thresholds.pose_keypoints) if fixed_thresholds else (0.0, 0.0)
        )
        pose_raw: dict[str, float] = {}
        pose_payload: dict[str, Any] = {
            "status": "not_run",
            "reason": "pose_not_enabled_for_this_run",
        }
        if self.pose_enabled:
            pose_validator = _RuntimePoseValidator(
                save_dir=root / "pose",
                label_cache_path=self.runtime_cache_root / "pose-val.cache",
                args=self._args(
                    task="pose",
                    data=self.pose_data,
                    batch=self.pose_batch,
                    workers=self.pose_workers,
                ),
            )
            raw = pose_validator(model=materialized.pose)
            if not isinstance(raw, dict):
                raise TypeError("官方 Pose validator 未回傳 metrics mapping")
            pose_raw = {str(key): float(value) for key, value in raw.items()}
            pose_names = {int(key): str(value) for key, value in materialized.pose.names.items()}
            pose_support = np.asarray(pose_validator.metrics.nt_per_class)
            pose_box, pose_box_threshold = _metric_payload(
                pose_validator.metrics.box,
                names=pose_names,
                supports=pose_support,
                threshold=(fixed_thresholds.pose_box if fixed_thresholds else None),
            )
            pose_keypoints, pose_keypoint_threshold = _metric_payload(
                pose_validator.metrics.pose,
                names=pose_names,
                supports=pose_support,
                threshold=(fixed_thresholds.pose_keypoints if fixed_thresholds else None),
            )
            pose_thresholds = (pose_box_threshold, pose_keypoint_threshold)
            pose_payload = {
                "status": "measured",
                "box": pose_box,
                "keypoints": pose_keypoints,
                "official_combined_fitness": float(pose_validator.metrics.fitness),
            }
        thresholds = ThresholdSet(
            detect_box=detect_threshold,
            pose_box=pose_thresholds[0],
            pose_keypoints=pose_thresholds[1],
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "scope": self.scope,
            "formal_split_used": self.formal_split_used,
            "epoch": epoch,
            "backend": "float",
            "detect": {"box": detect},
            "pose": pose_payload,
            "thresholds": thresholds.to_dict(),
            "detect_raw": {str(key): float(value) for key, value in detect_raw.items()},
            "pose_raw": pose_raw,
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        flat = _flat(
            {
                "detect": payload["detect"],
                "pose": payload["pose"],
            }
        )
        return ScreenValidationResult(
            epoch=epoch,
            metrics=payload,
            flat_metrics=flat,
            thresholds=thresholds,
            output_dir=root,
        )
