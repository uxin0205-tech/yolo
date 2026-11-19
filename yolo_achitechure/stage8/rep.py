"""沿用原生 RepConv 訓練；先線性折疊，activation 由明確的算子模式決定。"""

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv, RepConv


def metadata(source, target):
    for name in ('i', 'f', 'type', 'np'):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    target.np = sum(p.numel() for p in target.parameters())


def from_conv(original):
    c = original.conv
    if (type(original) is not Conv or c.kernel_size != (3, 3)
            or c.stride != (2, 2) or c.padding != (1, 1)
            or c.dilation != (1, 1) or c.bias is not None):
        raise ValueError('只接受原 stride2 Conv17／20')
    rep = RepConv(c.in_channels, c.out_channels, 3, 2, g=c.groups,
                  act=original.act, bn=False).to(c.weight)
    rep.conv1.conv.load_state_dict(c.state_dict(), strict=True)
    rep.conv1.bn.load_state_dict(original.bn.state_dict(), strict=True)
    for branch in (rep.conv1, rep.conv2):
        branch.bn.eps = original.bn.eps
        branch.bn.momentum = original.bn.momentum
    with torch.no_grad():
        rep.conv2.bn.weight.zero_()
        rep.conv2.bn.bias.zero_()
    metadata(original, rep)
    return rep.eval()


class AsymmetricRepConv(nn.Module):
    """ACB 形式；3x3／1x3／3x1 對齊相加，單次 activation。"""

    def __init__(self, original):
        super().__init__()
        seed_rep = from_conv(original)
        self.conv1 = seed_rep.conv1
        c = original.conv
        self.bn = None
        self.act = original.act
        for name, kernel, padding in (('conv2', (1, 3), (0, 1)), ('conv3', (3, 1), (1, 0))):
            branch = Conv(c.in_channels, c.out_channels, kernel, 2, p=padding, g=c.groups, act=False).to(c.weight)
            branch.bn.eps, branch.bn.momentum = original.bn.eps, original.bn.momentum
            with torch.no_grad():
                branch.bn.weight.zero_()
                branch.bn.bias.zero_()
            setattr(self, name, branch)
        metadata(original, self)
        self.eval()

    def forward(self, inputs):
        return self.act(self.conv1(inputs) + self.conv2(inputs) + self.conv3(inputs))

    def get_equivalent_kernel_bias(self):
        weights, biases = [], []
        for branch in branches(self):
            bn = branch.bn
            scale = bn.weight / (bn.running_var + bn.eps).sqrt()
            kernel = branch.conv.weight * scale[:, None, None, None]
            h, w = kernel.shape[-2:]
            weights.append(torch.nn.functional.pad(kernel, ((3-w)//2, (3-w)//2, (3-h)//2, (3-h)//2)))
            biases.append(bn.bias - bn.running_mean * scale)
        return sum(weights), sum(biases)


def branches(rep):
    return (rep.conv1, rep.conv2, rep.conv3) if type(rep) is AsymmetricRepConv else (rep.conv1, rep.conv2)


class FoldedRepConv(nn.Module):
    """單 Conv3×3+bias → 相同 activation；不把非線性折入 Conv。"""

    def __init__(self, rep):
        super().__init__()
        if type(rep) not in (RepConv, AsymmetricRepConv) or rep.bn is not None:
            raise ValueError('需要原生雙分支 RepConv')
        if any(m.training for m in rep.modules() if isinstance(m, nn.BatchNorm2d)):
            raise ValueError('folding 需要固定 BN 統計')
        c = rep.conv1.conv
        if c.weight.dtype != torch.float32:
            raise ValueError('先在 FP32 folding，再轉 dtype')
        self.conv = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size,
                              c.stride, c.padding, groups=c.groups, bias=True).to(c.weight)
        with torch.no_grad():
            weight, bias = rep.get_equivalent_kernel_bias()
            self.conv.weight.copy_(weight)
            self.conv.bias.copy_(bias)
        self.act = rep.act
        metadata(rep, self)
        self.eval()

    def forward(self, inputs):
        return self.act(self.conv(inputs))
