"""只供訓練使用的 P3 HOG 輔助監督，不改動部署特徵。

輸入影像必須是與主模型相同、完成增強後的 RGB BCHW 浮點張量
（一般為 [0, 1]，不是另行讀取或反增強的影像）。GT 介面沿用
Ultralytics 的 ``bboxes: (N, 4), normalized xywh`` 與
``batch_idx: (N,) 或 (N, 1)``，座標相對於該 BCHW 畫布。

cell (row, col) 覆蓋 [row*s:(row+1)*s, col*s:(col+1)*s] 像素，
中心為 ((col+.5)*s, (row+.5)*s)。只有中心落在任一有效 GT 框內的
cell 接受監督；採左／上含、右／下不含的半開邊界。框可伸出畫布，
但非有限座標會報錯，非正面積框不參與。影像尺寸必須整除 cell_size，
P3 尺寸必須精確相符；本模組不偷偷插值 target 或補邊。

這是 9-bin unsigned HOG 的 per-cell 機率監督，不是 block L2-Hys
HOG descriptor：bin k 的中心為 k*pi/9，以相鄰中心線性分配 magnitude，
最後一個 bin 與第零個 bin 週期相接，再在 cell 內 L1 正規化。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HOGTargets:
    probabilities: Tensor  # B, bins, H/cell_size, W/cell_size；無梯度
    valid_mask: Tensor  # B, H/cell_size, W/cell_size；bool
    energy: Tensor  # cell 內 magnitude 總和；無梯度


@dataclass(frozen=True)
class HOGAuxiliaryOutput:
    loss: Tensor  # per_image_loss 的 batch mean；未乘 μ
    per_image_loss: Tensor  # 各影像有效 cell 平均；無有效 cell 為 0
    loss_sum: Tensor  # per_image_loss.sum()；供 macro 的 batch-sum 正規化
    logits: Tensor
    targets: HOGTargets


def _check_parameters(cell_size: int, bins: int, energy_epsilon: float) -> None:
    if not isinstance(cell_size, int) or isinstance(cell_size, bool) or cell_size < 2:
        raise ValueError("cell_size 必須是至少 2 的整數")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins 必須是至少 2 的整數")
    if not math.isfinite(energy_epsilon) or energy_epsilon < 0:
        raise ValueError("energy_epsilon 必須是有限的非負數")


def _box_cell_mask(
    images: Tensor, bboxes: Tensor, batch_idx: Tensor, cell_size: int
) -> Tensor:
    batch, _, height, width = images.shape
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("bboxes 必須是 (N, 4) 的 normalized xywh")
    if batch_idx.ndim not in (1, 2) or (batch_idx.ndim == 2 and batch_idx.shape[1] != 1):
        raise ValueError("batch_idx 必須是 (N,) 或 (N, 1)")
    if batch_idx.numel() != bboxes.shape[0]:
        raise ValueError("bboxes 與 batch_idx 的 GT 數量不一致")
    boxes = bboxes.detach().to(device=images.device, dtype=torch.float32)
    indices = batch_idx.detach().flatten().to(device=images.device)
    if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(indices).all()):
        raise ValueError("GT 座標與 batch_idx 必須有限")
    if bool(((indices < 0) | (indices >= batch) | (indices != indices.long())).any()):
        raise ValueError("batch_idx 必須是有效的 batch 整數索引")
    indices = indices.long()
    cell_h, cell_w = height // cell_size, width // cell_size
    mask = torch.zeros((batch, cell_h, cell_w), dtype=torch.bool, device=images.device)
    if boxes.shape[0] == 0:
        return mask
    x = (torch.arange(cell_w, device=images.device, dtype=torch.float32) + 0.5) / cell_w
    y = (torch.arange(cell_h, device=images.device, dtype=torch.float32) + 0.5) / cell_h
    left_top = boxes[:, :2] - boxes[:, 2:] * 0.5
    right_bottom = boxes[:, :2] + boxes[:, 2:] * 0.5
    inside = (
        (x[None, None, :] >= left_top[:, 0, None, None])
        & (x[None, None, :] < right_bottom[:, 0, None, None])
        & (y[None, :, None] >= left_top[:, 1, None, None])
        & (y[None, :, None] < right_bottom[:, 1, None, None])
        & (boxes[:, 2, None, None] > 0)
        & (boxes[:, 3, None, None] > 0)
    )
    # 先逐框判斷，再依 batch 合併；不建立 B*N*Hc*Wc 張量。
    counts = torch.zeros((batch, cell_h, cell_w), dtype=torch.int32, device=images.device)
    counts.index_add_(0, indices, inside.to(torch.int32))
    return counts > 0


def _chunk_histograms(images: Tensor, cell_size: int, bins: int) -> Tensor:
    images = images.float()
    gray = images[:, 0] * 0.299 + images[:, 1] * 0.587 + images[:, 2] * 0.114
    # 中央差分；畫布邊界使用單側差分。常數影像不會產生補零邊界假邊緣。
    dx = torch.cat(
        (gray[..., 1:2] - gray[..., :1],
         (gray[..., 2:] - gray[..., :-2]) * 0.5,
         gray[..., -1:] - gray[..., -2:-1]), dim=-1,
    )
    dy = torch.cat(
        (gray[:, 1:2] - gray[:, :1],
         (gray[:, 2:] - gray[:, :-2]) * 0.5,
         gray[:, -1:] - gray[:, -2:-1]), dim=-2,
    )
    magnitude = torch.sqrt(dx.square() + dy.square())
    phase = torch.remainder(torch.atan2(dy, dx), math.pi) * (bins / math.pi)
    lower = torch.floor(phase)
    fraction = phase - lower
    lower = lower.to(torch.long).remainder(bins)
    batch, height, width = gray.shape
    cell_h, cell_w = height // cell_size, width // cell_size

    def cells(value: Tensor) -> Tensor:
        return value.reshape(batch, cell_h, cell_size, cell_w, cell_size).permute(
            0, 1, 3, 2, 4
        ).reshape(batch, cell_h * cell_w, cell_size * cell_size)

    lower_cells = cells(lower)
    histogram = magnitude.new_zeros((batch, cell_h * cell_w, bins))
    histogram.scatter_add_(2, lower_cells, cells(magnitude * (1 - fraction)))
    histogram.scatter_add_(2, (lower_cells + 1).remainder(bins), cells(magnitude * fraction))
    return histogram.reshape(batch, cell_h, cell_w, bins).permute(0, 3, 1, 2)


@torch.no_grad()
def build_hog_targets(
    images: Tensor,
    bboxes: Tensor,
    batch_idx: Tensor,
    *,
    cell_size: int = 8,
    bins: int = 9,
    energy_epsilon: float = 1e-6,
    chunk_size: int = 8,
) -> HOGTargets:
    """以停梯度 FP32 建立 target；chunk_size 只控制 target 暫存峰值。"""
    _check_parameters(cell_size, bins, energy_epsilon)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size 必須是正整數")
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] < 1:
        raise ValueError("images 必須是非空的 RGB BCHW 張量")
    if not images.is_floating_point():
        raise TypeError("images 必須是浮點張量（通常 RGB [0, 1]）")
    height, width = images.shape[-2:]
    if height < cell_size or width < cell_size or height % cell_size or width % cell_size:
        raise ValueError("images 的 H/W 必須是 cell_size 的正整數倍")
    with torch.autocast(device_type=images.device.type, enabled=False):
        mask = _box_cell_mask(images, bboxes, batch_idx, cell_size)
        histograms = torch.cat(
            [_chunk_histograms(chunk, cell_size, bins) for chunk in images.detach().split(chunk_size)],
            dim=0,
        )
        energy = histograms.sum(dim=1)
        # clamp 的 tiny 只避免除以零，不把無能量 cell 誤當成有效 target。
        probabilities = histograms / energy.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(1)
        valid_mask = mask & (energy > energy_epsilon)
    return HOGTargets(probabilities, valid_mask, energy)


class HOGAuxiliary(nn.Module):
    """raw pre-MASF P3 -> 1x1 Conv 的訓練輔助頭；loss 尚未乘 μ。"""

    def __init__(
        self,
        channels: int,
        *,
        cell_size: int = 8,
        bins: int = 9,
        energy_epsilon: float = 1e-6,
        target_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels 必須是正整數")
        _check_parameters(cell_size, bins, energy_epsilon)
        if not isinstance(target_chunk_size, int) or isinstance(target_chunk_size, bool) or target_chunk_size < 1:
            raise ValueError("target_chunk_size 必須是正整數")
        self.projection = nn.Conv2d(channels, bins, kernel_size=1)
        self.cell_size = cell_size
        self.bins = bins
        self.energy_epsilon = energy_epsilon
        self.target_chunk_size = target_chunk_size

    def forward(
        self, raw_p3: Tensor, images: Tensor, bboxes: Tensor, batch_idx: Tensor
    ) -> HOGAuxiliaryOutput:
        if raw_p3.ndim != 4 or raw_p3.shape[1] != self.projection.in_channels:
            raise ValueError("raw_p3 必須是 BCHW，channel 必須符合 channels")
        if images.ndim != 4 or raw_p3.shape[0] != images.shape[0]:
            raise ValueError("raw_p3 與 images 的 batch 必須相同")
        if raw_p3.device != images.device:
            raise ValueError("raw_p3 與 images 必須在相同 device")
        expected = (images.shape[-2] // self.cell_size, images.shape[-1] // self.cell_size)
        if tuple(raw_p3.shape[-2:]) != expected:
            raise ValueError(f"raw_p3 空間尺寸必須是 {expected}，不會自動插值 target")
        targets = build_hog_targets(
            images, bboxes, batch_idx, cell_size=self.cell_size, bins=self.bins,
            energy_epsilon=self.energy_epsilon, chunk_size=self.target_chunk_size,
        )
        logits = self.projection(raw_p3)
        per_cell = -(targets.probabilities * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
        valid = targets.valid_mask.to(per_cell.dtype)
        # 全空時仍保留通往原生 feature／aux head 的零梯度路徑。
        per_image_loss = (per_cell * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
        return HOGAuxiliaryOutput(
            loss=per_image_loss.mean(), per_image_loss=per_image_loss,
            loss_sum=per_image_loss.sum(), logits=logits, targets=targets,
        )
