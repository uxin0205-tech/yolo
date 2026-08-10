"""Static, hardware-friendly Ball-only P2 prediction for SP2."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from ultralytics.nn.modules import Conv, DFL, DWConv, Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


SP2_HIDDEN_CHANNELS = 32
SP2_AUXILIARY_LOSS_WEIGHT = 1.0
BALL_CLASS_ID = 0
BAT_CLASS_ID = 1


def _light_tower(input_channels: int, hidden_channels: int, output_channels: int) -> nn.Sequential:
    """Two fixed-shape depthwise-separable blocks followed by a prediction layer."""
    return nn.Sequential(
        DWConv(input_channels, input_channels, 3),
        Conv(input_channels, hidden_channels, 1),
        DWConv(hidden_channels, hidden_channels, 3),
        Conv(hidden_channels, hidden_channels, 1),
        nn.Conv2d(hidden_channels, output_channels, 1),
    )


def ball_only_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow batch copy whose targets contain class-0 Ball only."""
    classes = batch["cls"].view(-1)
    keep = classes.to(dtype=torch.long) == BALL_CLASS_ID
    filtered = dict(batch)
    for key in ("batch_idx", "cls", "bboxes"):
        filtered[key] = batch[key][keep]
    return filtered


class SelectiveP2Detect(Detect):
    """P2 predicts Ball only; the standard P3-P5 head predicts both classes."""

    def __init__(
        self,
        standard: Detect,
        *,
        p2_channels: int,
        hidden_channels: int = SP2_HIDDEN_CHANNELS,
        auxiliary_loss_weight: float = SP2_AUXILIARY_LOSS_WEIGHT,
    ) -> None:
        nn.Module.__init__(self)
        if standard.nc != 2 or standard.nl != 4:
            raise ValueError("SP2 requires a four-scale two-class Detect source")
        if hidden_channels < 1:
            raise ValueError("SP2 hidden channels must be positive")
        if auxiliary_loss_weight <= 0:
            raise ValueError("SP2 auxiliary loss weight must be positive")
        self.nc = 2
        self.nl = 4
        self.reg_max = standard.reg_max
        self.no = self.nc + 4 * self.reg_max
        self.end2end = False
        self.hidden_channels = hidden_channels
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.ball_class_id = BALL_CLASS_ID
        self.bat_class_id = BAT_CLASS_ID

        # Reuse the already initialized/transferred P3-P5 towers exactly.
        self.main_cv2 = nn.ModuleList(list(standard.cv2)[1:])
        self.main_cv3 = nn.ModuleList(list(standard.cv3)[1:])
        self.ball_cv2 = _light_tower(p2_channels, hidden_channels, 4 * self.reg_max)
        self.ball_cv3 = _light_tower(p2_channels, hidden_channels, 1)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.stride = standard.stride.detach().clone()
        self.inplace = standard.inplace

    @property
    def p2_stride(self) -> Tensor:
        return self.stride[:1]

    @property
    def main_stride(self) -> Tensor:
        return self.stride[1:]

    def _ball_predictions(self, feature: Tensor) -> dict[str, Tensor | list[Tensor]]:
        batch = feature.shape[0]
        return {
            "boxes": self.ball_cv2(feature).view(batch, 4 * self.reg_max, -1),
            "scores": self.ball_cv3(feature).view(batch, 1, -1),
            "feats": [feature],
        }

    def _main_predictions(self, features: list[Tensor]) -> dict[str, Tensor | list[Tensor]]:
        batch = features[0].shape[0]
        boxes = torch.cat(
            [self.main_cv2[index](feature).view(batch, 4 * self.reg_max, -1) for index, feature in enumerate(features)],
            dim=-1,
        )
        scores = torch.cat(
            [self.main_cv3[index](feature).view(batch, self.nc, -1) for index, feature in enumerate(features)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": features}

    def _decode(self, predictions: dict[str, Any], strides: Tensor) -> Tensor:
        anchors, stride_tensor = make_anchors(predictions["feats"], strides, 0.5)
        anchors = anchors.transpose(0, 1)
        stride_tensor = stride_tensor.transpose(0, 1)
        boxes = self.decode_bboxes(
            self.dfl(predictions["boxes"]), anchors.unsqueeze(0)
        ) * stride_tensor
        return boxes

    def forward(self, features: list[Tensor]) -> Any:
        if len(features) != 4:
            raise ValueError(f"SP2 requires P2-P5 features, got {len(features)}")
        ball = self._ball_predictions(features[0])
        main = self._main_predictions(features[1:])
        raw = {"ball": ball, "main": main}
        if self.training:
            return raw

        ball_scores = ball["scores"].sigmoid()
        zero_bat = torch.zeros_like(ball_scores)
        ball_output = torch.cat(
            (self._decode(ball, self.p2_stride), ball_scores, zero_bat), dim=1
        )
        main_output = torch.cat(
            (self._decode(main, self.main_stride), main["scores"].sigmoid()), dim=1
        )
        merged = torch.cat((ball_output, main_output), dim=-1)
        return merged if self.export else (merged, raw)

    def bias_init(self) -> None:
        """Initialize all prediction biases using their actual class count and stride."""
        self.ball_cv2[-1].bias.data.fill_(2.0)
        self.ball_cv3[-1].bias.data.fill_(math.log(5 / (640 / self.p2_stride[0]) ** 2))
        for index, (box_tower, cls_tower) in enumerate(zip(self.main_cv2, self.main_cv3)):
            box_tower[-1].bias.data.fill_(2.0)
            cls_tower[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.main_stride[index]) ** 2
            )


class _LossModelProxy:
    """Minimal model surface required by the pinned v8DetectionLoss."""

    def __init__(self, model: nn.Module, *, nc: int, stride: Tensor, reg_max: int) -> None:
        self._model = model
        self.model = [SimpleNamespace(nc=nc, stride=stride, reg_max=reg_max)]
        self.args = model.args
        self.class_weights = None

    def parameters(self):
        return self._model.parameters()


class SelectiveDetectionLoss:
    """Standard P3-P5 loss plus a class-0-only P2 auxiliary loss."""

    def __init__(self, model: nn.Module) -> None:
        head = model.model[-1]
        if not isinstance(head, SelectiveP2Detect):
            raise TypeError("SelectiveDetectionLoss requires SelectiveP2Detect")
        self.weight = head.auxiliary_loss_weight
        self.main_loss = v8DetectionLoss(
            _LossModelProxy(model, nc=2, stride=head.main_stride, reg_max=head.reg_max)
        )
        self.ball_loss = v8DetectionLoss(
            _LossModelProxy(model, nc=1, stride=head.p2_stride, reg_max=head.reg_max)
        )

    def __call__(self, predictions: Any, batch: dict[str, Tensor]):
        # Ultralytics validation passes eval output as (decoded, raw_predictions),
        # while training passes raw_predictions directly.
        if isinstance(predictions, tuple):
            if len(predictions) != 2 or not isinstance(predictions[1], dict):
                raise ValueError("invalid SP2 validation prediction wrapper")
            predictions = predictions[1]
        if not isinstance(predictions, dict):
            raise TypeError("SP2 loss requires raw prediction dictionaries")
        if set(predictions) != {"ball", "main"}:
            raise ValueError("SP2 predictions must contain ball and main branches")
        main_total, main_components = self.main_loss(predictions["main"], batch)
        p2_batch = ball_only_batch(batch)
        ball_total, ball_components = self.ball_loss(predictions["ball"], p2_batch)
        return (
            main_total + self.weight * ball_total,
            main_components + self.weight * ball_components,
        )
