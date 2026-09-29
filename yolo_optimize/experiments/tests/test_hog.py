"""HOG 的方向、座標與梯度契約；只需 CPU，不啟動模型或 GPU。"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_optimize.hog import HOGAuxiliary, build_hog_targets


def ramp(angle: float, *, height: int = 16, width: int = 16) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    gray = x.float() * math.cos(angle) + y.float() * math.sin(angle)
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    return gray[None, None].repeat(1, 3, 1, 1)


def full_box() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[0.5, 0.5, 1.0, 1.0]]), torch.tensor([0])


class HOGTargetsTests(unittest.TestCase):
    def test_horizontal_gradient_and_reversed_polarity_share_unsigned_bin(self) -> None:
        images = ramp(0)
        boxes, indices = full_box()
        forward = build_hog_targets(images, boxes, indices)
        reverse = build_hog_targets(1 - images, boxes, indices)
        torch.testing.assert_close(forward.probabilities[:, 0], torch.ones(1, 2, 2))
        torch.testing.assert_close(forward.probabilities[:, 1:], torch.zeros(1, 8, 2, 2))
        torch.testing.assert_close(forward.probabilities, reverse.probabilities, atol=2e-6, rtol=0)
        self.assertTrue(bool(forward.valid_mask.all()))

    def test_vertical_gradient_interpolates_two_adjacent_bins(self) -> None:
        target = build_hog_targets(ramp(math.pi / 2), *full_box())
        expected = torch.zeros_like(target.probabilities)
        expected[:, 4:6] = 0.5
        torch.testing.assert_close(target.probabilities, expected, atol=2e-6, rtol=0)

    def test_orientation_near_pi_interpolates_across_periodic_boundary(self) -> None:
        # 170 度是 bin 8（160 度）與週期 bin 0（180 度）的中點。
        target = build_hog_targets(ramp(math.radians(170)), *full_box())
        expected = torch.zeros_like(target.probabilities)
        expected[:, 8] = 0.5
        expected[:, 0] = 0.5
        torch.testing.assert_close(target.probabilities, expected, atol=3e-6, rtol=0)

    def test_mask_uses_augmented_normalized_boxes_and_cell_centers(self) -> None:
        images = ramp(0, height=16, width=24).repeat(2, 1, 1, 1)
        # 第一張的左上 cell、第二張的右下 cell；同一框重複不重複計權。
        boxes = torch.tensor([
            [1 / 6, 1 / 4, 1 / 3, 1 / 2],
            [5 / 6, 3 / 4, 1 / 3, 1 / 2],
            [5 / 6, 3 / 4, 1 / 3, 1 / 2],
            [0.5, 0.5, 0.0, 1.0],
        ])
        target = build_hog_targets(images, boxes, torch.tensor([[0.0], [1.0], [1.0], [0.0]]))
        expected = torch.zeros(2, 2, 3, dtype=torch.bool)
        expected[0, 0, 0] = True
        expected[1, 1, 2] = True
        self.assertTrue(torch.equal(target.valid_mask, expected))

    def test_center_on_right_boundary_is_excluded(self) -> None:
        boxes = torch.tensor([[0.125, 0.5, 0.25, 1.0]])
        target = build_hog_targets(ramp(0), boxes, torch.tensor([0]))
        self.assertFalse(bool(target.valid_mask.any()))

    def test_chunking_does_not_change_targets(self) -> None:
        images = torch.cat([ramp(0), ramp(math.pi / 2), ramp(math.radians(170))])
        boxes = full_box()[0].repeat(3, 1)
        indices = torch.arange(3)
        small = build_hog_targets(images, boxes, indices, chunk_size=1)
        large = build_hog_targets(images, boxes, indices, chunk_size=8)
        torch.testing.assert_close(small.probabilities, large.probabilities)
        torch.testing.assert_close(small.energy, large.energy)
        self.assertTrue(torch.equal(small.valid_mask, large.valid_mask))


class HOGAuxiliaryTests(unittest.TestCase):
    def test_aux_loss_backpropagates_to_features_not_images_or_gt(self) -> None:
        torch.manual_seed(4)
        features = torch.randn(1, 5, 2, 2, requires_grad=True)
        images = ramp(0).requires_grad_()
        boxes, indices = full_box()
        boxes.requires_grad_()
        head = HOGAuxiliary(5)
        output = head(features, images, boxes, indices)
        self.assertFalse(output.targets.probabilities.requires_grad)
        self.assertFalse(output.targets.energy.requires_grad)
        output.loss.backward()
        self.assertIsNone(images.grad)
        self.assertIsNone(boxes.grad)
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertGreater(float(head.projection.weight.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(output.loss)))

    def test_empty_gt_returns_differentiable_zero(self) -> None:
        features = torch.randn(2, 7, 2, 2, requires_grad=True)
        head = HOGAuxiliary(7)
        output = head(features, ramp(0).repeat(2, 1, 1, 1), torch.empty(0, 4), torch.empty(0))
        self.assertEqual(float(output.loss.detach()), 0.0)
        self.assertFalse(bool(output.targets.valid_mask.any()))
        output.loss.backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))
        torch.testing.assert_close(head.projection.weight.grad, torch.zeros_like(head.projection.weight))

    def test_flat_nonzero_image_does_not_create_boundary_edges_or_nan(self) -> None:
        features = torch.randn(1, 3, 2, 2, requires_grad=True)
        head = HOGAuxiliary(3)
        output = head(features, torch.full((1, 3, 16, 16), 0.7), *full_box())
        self.assertEqual(float(output.loss.detach()), 0.0)
        self.assertFalse(bool(output.targets.valid_mask.any()))
        self.assertTrue(bool(torch.isfinite(output.targets.probabilities).all()))
        torch.testing.assert_close(output.targets.energy, torch.zeros(1, 2, 2))
        output.loss.backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))

    def test_loss_is_mean_over_valid_cells_not_full_map(self) -> None:
        head = HOGAuxiliary(3)
        torch.nn.init.zeros_(head.projection.weight)
        torch.nn.init.zeros_(head.projection.bias)
        features = torch.randn(1, 3, 2, 2)
        boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]])
        output = head(features, ramp(0), boxes, torch.tensor([0]))
        self.assertEqual(int(output.targets.valid_mask.sum()), 1)
        self.assertAlmostEqual(float(output.loss.detach()), math.log(9), places=6)

    def test_spatial_mismatch_and_invalid_gt_are_rejected(self) -> None:
        head = HOGAuxiliary(3)
        with self.assertRaisesRegex(ValueError, "空間尺寸"):
            head(torch.randn(1, 3, 3, 2), ramp(0), *full_box())
        with self.assertRaisesRegex(ValueError, "正整數倍"):
            build_hog_targets(ramp(0, height=17), *full_box())
        with self.assertRaisesRegex(ValueError, "整數索引"):
            build_hog_targets(ramp(0), full_box()[0], torch.tensor([0.5]))
        with self.assertRaisesRegex(ValueError, "必須有限"):
            build_hog_targets(ramp(0), torch.tensor([[float("nan"), 0.5, 1.0, 1.0]]), torch.tensor([0]))

    def test_batch_sum_is_invariant_to_microbatch_partition(self) -> None:
        torch.manual_seed(8)
        head = HOGAuxiliary(3)
        images = ramp(0).repeat(4, 1, 1, 1)
        features = torch.randn(4, 3, 2, 2, requires_grad=True)
        boxes = full_box()[0].repeat(3, 1)
        # 第四張沒有 GT，但仍是 logical batch 中的一張負樣本。
        whole = head(features, images, boxes, torch.arange(3))
        self.assertEqual(tuple(whole.per_image_loss.shape), (4,))
        self.assertEqual(float(whole.per_image_loss[-1].detach()), 0.0)
        torch.testing.assert_close(whole.loss * 4, whole.loss_sum)
        pieces = [
            head(features[:2], images[:2], boxes[:2], torch.arange(2)).loss_sum,
            head(features[2:], images[2:], boxes[2:], torch.tensor([0])).loss_sum,
        ]
        partitioned = sum(pieces)
        torch.testing.assert_close(whole.loss_sum, partitioned)
        # 同一 reference batch 的梯度不能因 physical microbatch 改變。
        whole_gradient = torch.autograd.grad(whole.loss_sum / 64, features, retain_graph=True)[0]
        partition_gradient = torch.autograd.grad(partitioned / 64, features)[0]
        torch.testing.assert_close(whole_gradient, partition_gradient)


if __name__ == "__main__":
    unittest.main()
