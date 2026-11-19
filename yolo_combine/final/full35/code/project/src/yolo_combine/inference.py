"""Single-trunk Detect/Pose inference and strict combined-weight loading."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops

from .contracts import Task, normalize_tasks

InferenceTask = Literal["detect", "pose", "both"]


@dataclass(frozen=True)
class LoadedCombinedWeights:
    path: Path
    checkpoint_kind: str
    state_source: str
    tensors: int


def load_combined_weights(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    prefer_ema: bool = True,
) -> LoadedCombinedWeights:
    """Strictly load either full-resume or inference-only combined weights."""

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("combined checkpoint must be a mapping")
    contract_method = getattr(model, "contract", None)
    if not callable(contract_method):
        raise TypeError("combined model exposes no contract()")
    if payload.get("contract") != contract_method():
        raise ValueError("combined checkpoint model contract mismatch")
    kind = str(payload.get("checkpoint_kind"))
    if kind == "inference_only":
        state_source = str(payload.get("source", "unknown"))
        state = payload.get("state_dict")
    elif kind == "full_resume":
        state_source = "ema" if prefer_ema else "live"
        state = payload.get("ema_state" if prefer_ema else "model_state")
    else:
        raise ValueError(f"unsupported combined checkpoint kind: {kind!r}")
    if not isinstance(state, dict):
        raise ValueError(f"combined checkpoint contains no {state_source} state")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"strict combined load drifted: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return LoadedCombinedWeights(
        path=path,
        checkpoint_kind=kind,
        state_source=state_source,
        tensors=len(state),
    )


class SharedDualPredictor:
    """Preprocess once, extract shared features once, then postprocess by task."""

    def __init__(
        self,
        model: nn.Module,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        amp: bool = True,
    ) -> None:
        if imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32")
        if not 0 <= conf <= 1 or not 0 <= iou <= 1:
            raise ValueError("conf and iou must be in [0,1]")
        if max_det < 1:
            raise ValueError("max_det must be positive")
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.amp = amp
        self.letterbox = LetterBox(
            new_shape=(imgsz, imgsz),
            auto=False,
            stride=32,
        )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as error:
            raise TypeError("prediction model has no parameters") from error

    def _images(
        self,
        source: np.ndarray | Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        values = [source] if isinstance(source, np.ndarray) else list(source)
        if not values:
            raise ValueError("inference source cannot be empty")
        for index, image in enumerate(values):
            if not isinstance(image, np.ndarray):
                raise TypeError(f"source[{index}] is not a numpy image")
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(
                    f"source[{index}] must be uint8 BGR HWC, got {image.shape} {image.dtype}"
                )
        return values

    def _preprocess(self, originals: Sequence[np.ndarray]) -> torch.Tensor:
        resized = [self.letterbox(image=image) for image in originals]
        array = np.stack(resized)
        array = array[..., ::-1].transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )
        return tensor.float() / 255.0

    @staticmethod
    def _inference_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise TypeError(
                "task head inference must return a rank-3 tensor or (tensor, raw)"
            )
        return value

    def _results(
        self,
        raw: Any,
        *,
        task: Task,
        processed: torch.Tensor,
        originals: Sequence[np.ndarray],
        paths: Sequence[str],
    ) -> list[Results]:
        head = getattr(self.model, f"{task.value}_head")
        names = dict(getattr(self.model, f"{task.value}_names"))
        prediction = self._inference_tensor(raw)
        selected = nms.non_max_suppression(
            prediction,
            self.conf,
            self.iou,
            classes=None,
            agnostic=False,
            max_det=self.max_det,
            nc=len(names),
            end2end=bool(head.end2end),
        )
        results: list[Results] = []
        for pred, original, path in zip(selected, originals, paths, strict=True):
            pred = pred.clone()
            pred[:, :4] = ops.scale_boxes(
                processed.shape[2:],
                pred[:, :4],
                original.shape,
            )
            result = Results(
                original,
                path=path,
                names=names,
                boxes=pred[:, :6],
            )
            if task is Task.POSE:
                kpt_shape = tuple(int(value) for value in head.kpt_shape)
                keypoints = pred[:, 6:].view(pred.shape[0], *kpt_shape)
                keypoints = ops.scale_coords(
                    processed.shape[2:],
                    keypoints,
                    original.shape,
                )
                result.update(keypoints=keypoints)
            results.append(result)
        return results

    def predict(
        self,
        source: np.ndarray | Sequence[np.ndarray],
        *,
        task: InferenceTask = "both",
        paths: Sequence[str] | None = None,
    ) -> dict[str, list[Results]]:
        selected = normalize_tasks(task)
        originals = self._images(source)
        resolved_paths = (
            [f"image-{index}" for index in range(len(originals))]
            if paths is None
            else [str(value) for value in paths]
        )
        if len(resolved_paths) != len(originals):
            raise ValueError("paths length must equal source batch length")
        processed = self._preprocess(originals)
        self.model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            enabled=self.amp and self.device.type == "cuda",
        ):
            raw_outputs = self.model(processed, task=task)
        expected = {value.value for value in selected}
        if set(raw_outputs) != expected:
            raise RuntimeError(
                f"shared inference returned {set(raw_outputs)}, expected {expected}"
            )
        return {
            selected_task.value: self._results(
                raw_outputs[selected_task.value],
                task=selected_task,
                processed=processed,
                originals=originals,
                paths=resolved_paths,
            )
            for selected_task in (Task.DETECT, Task.POSE)
            if selected_task in selected
        }

