"""Explicit COCO evaluation requests and standardized queue result artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import VariantConfig
from .integration import convert_yolo26_model, fixed_scale_modules
from .queue_model import QueueResult


class ResultContractError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationRecipe:
    data: str
    imgsz: int
    batch: int
    device: str
    workers: int
    split: str
    plots: bool

    def __post_init__(self) -> None:
        if not self.data.strip():
            raise ValueError("evaluation data path cannot be empty")
        if min(self.imgsz, self.batch) < 1 or self.workers < 0:
            raise ValueError("evaluation imgsz/batch must be positive and workers non-negative")
        if self.split not in {"val", "test"}:
            raise ValueError("evaluation split must be val or test")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationRecipe:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("evaluation YAML must contain a mapping")
        return cls(**data)

    def to_ultralytics_args(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRequest:
    run_id: str
    run_dir: Path
    parent_checkpoint: Path
    recipe: EvaluationRecipe
    variant_path: Path | None = None


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ResultContractError(f"{field} must be finite")
    return number


def standardize_metrics(result: object) -> dict[str, object]:
    box = getattr(result, "box", None)
    if box is None or not hasattr(box, "map"):
        raise ResultContractError("Ultralytics result is missing box.map")
    maps_source = getattr(box, "maps", ())
    maps = [_finite(value, "box.maps") for value in maps_source]
    return {
        "map50_95": _finite(box.map, "box.map"),
        "map50": _finite(getattr(box, "map50", None), "box.map50"),
        "map75": _finite(getattr(box, "map75", None), "box.map75"),
        "maps": maps,
    }


def collect_row_sum_max_error(model: object) -> float | None:
    """Collect the worst available row-normalization diagnostic after validation."""

    import torch

    modules = getattr(model, "modules", None)
    if modules is None:
        return None
    errors: list[float] = []
    for module in modules():
        rows = getattr(module, "last_row_sums", None)
        if not isinstance(rows, torch.Tensor) or rows.numel() == 0:
            continue
        if torch.is_floating_point(rows):
            error = (rows.float() - 1.0).abs().max().item()
        else:
            error = (rows.float() - 255.0).abs().max().item() / 255.0
        errors.append(float(error))
    return max(errors) if errors else None


def write_standard_result(
    run_dir: str | Path,
    metrics: dict[str, object],
    *,
    checkpoint_path: str | Path | None,
    profile_path: str | Path | None,
    row_sum_max_error: float | None,
) -> QueueResult:
    run = Path(run_dir)
    metrics_dir = run / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
    profile = str(Path(profile_path).resolve()) if profile_path is not None else None
    payload = {
        "map50_95": _finite(metrics.get("map50_95"), "map50_95"),
        "map50": _finite(metrics.get("map50"), "map50"),
        "map75": _finite(metrics.get("map75"), "map75"),
        "maps": [_finite(value, "maps") for value in metrics.get("maps", ())],
        "row_sum_max_error": (
            _finite(row_sum_max_error, "row_sum_max_error") if row_sum_max_error is not None else None
        ),
        "checkpoint_path": checkpoint,
        "profile_path": profile,
    }
    result_path = metrics_dir / "queue-result.json"
    payload["metrics_path"] = str(result_path.resolve())
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return QueueResult.from_dict(payload)


def resolve_run_outputs(run_dir: str | Path, *, require_checkpoint: bool) -> QueueResult:
    path = Path(run_dir) / "metrics" / "queue-result.json"
    if not path.is_file():
        raise ResultContractError(f"missing standard metrics artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ResultContractError("queue-result.json must contain an object")
    result = QueueResult.from_dict(payload)
    _finite(result.map50_95, "map50_95")
    if require_checkpoint and (result.checkpoint_path is None or not Path(result.checkpoint_path).is_file()):
        raise ResultContractError("required checkpoint does not exist")
    if result.profile_path is not None and not Path(result.profile_path).is_file():
        raise ResultContractError("referenced profile does not exist")
    return result


class UltralyticsEvaluationBackend:
    """The only evaluation object allowed to call Ultralytics val()."""

    def __init__(self, model_factory: Callable[[str], Any] | None = None) -> None:
        self._model_factory = model_factory

    def _make_model(self, checkpoint: Path) -> Any:
        if self._model_factory is None:
            from ultralytics import YOLO

            return YOLO(str(checkpoint.resolve()))
        return self._model_factory(str(checkpoint.resolve()))

    @staticmethod
    def _validation_args(request: EvaluationRequest) -> dict[str, object]:
        args = request.recipe.to_ultralytics_args()
        args.update(project=str(request.run_dir.resolve()), name="ultralytics", exist_ok=True)
        return args

    def evaluate_official(self, request: EvaluationRequest) -> QueueResult:
        model = self._make_model(request.parent_checkpoint)
        metrics = standardize_metrics(model.val(**self._validation_args(request)))
        row_error = collect_row_sum_max_error(getattr(model, "model", model))
        return write_standard_result(
            request.run_dir,
            metrics,
            checkpoint_path=request.parent_checkpoint,
            profile_path=None,
            row_sum_max_error=row_error,
        )

    def evaluate_variant(self, request: EvaluationRequest) -> QueueResult:
        if request.variant_path is None:
            raise ValueError("variant evaluation requires variant_path")
        model = self._make_model(request.parent_checkpoint)
        convert_yolo26_model(model.model, VariantConfig.from_yaml(request.variant_path))
        validation_args = self._validation_args(request)
        checkpoint_path = request.parent_checkpoint
        pending = fixed_scale_modules(model.model)
        if pending:
            # Ultralytics validation fuses Conv+BN in place. Preserve the
            # trainable structure before calibration, then copy only the
            # observed fixed-scale state back into it for the parent checkpoint.
            training_model = deepcopy(model.model)
            for attention in pending:
                attention.score.begin_calibration()
            model.val(**validation_args)
            for attention in pending:
                attention.score.finish_calibration()
            training_pending = fixed_scale_modules(training_model)
            if len(training_pending) != len(pending):
                raise RuntimeError("fixed-scale Attention count changed during calibration")
            for calibrated, trainable in zip(pending, training_pending, strict=True):
                trainable.score.set_fixed_coefficients(calibrated.score.fixed_coefficients)
            checkpoint_path = request.run_dir / "checkpoints" / "calibrated.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            inference_model = model.model
            try:
                model.model = training_model
                model.save(str(checkpoint_path))
            finally:
                model.model = inference_model
        metrics = standardize_metrics(model.val(**validation_args))
        row_error = collect_row_sum_max_error(model.model)
        return write_standard_result(
            request.run_dir,
            metrics,
            checkpoint_path=checkpoint_path,
            profile_path=None,
            row_sum_max_error=row_error,
        )
