from torch import nn

from yolo_combine.formal_training import _apply_shared_bn_affine


class TinyBNGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = nn.Module()
        self.graph.shared = nn.BatchNorm2d(2)
        self.graph.detect_head = nn.Module()
        self.graph.detect_head.bn = nn.BatchNorm2d(2)
        self.graph.pose_head = nn.Module()
        self.graph.pose_head.bn = nn.BatchNorm2d(2)


def test_shared_bn_affine_can_be_disabled_without_freezing_head_bn() -> None:
    model = TinyBNGraph()

    _apply_shared_bn_affine(model, trainable=False)

    assert model.graph.shared.weight.requires_grad is False
    assert model.graph.shared.bias.requires_grad is False
    assert model.graph.detect_head.bn.weight.requires_grad is True
    assert model.graph.pose_head.bn.weight.requires_grad is True

