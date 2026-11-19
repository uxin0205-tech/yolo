"""沿用 optimize 的 full35 與 detach-before-MASF task bridge。"""

import copy
from contextlib import contextmanager
from dataclasses import asdict

import torch
from achitechure_1.masf import P3MASFFull35
from activation_lab.activations import (
    ShiftPiecewiseQuadraticProfile,
    ShiftPiecewiseQuadraticSiLU,
)
from torch import nn
from ultralytics.nn.modules.head import Detect, Pose26


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, beta):
        ctx.beta = beta
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.beta, None


class TaskBridge:
    def forward(self, x):
        if not self.training:
            return super().forward([self.p3_masf(x[0]), x[1], x[2]])
        if not self.end2end or not 0 < self.bridge_beta <= 1:
            raise ValueError('invalid task bridge')
        if any(m.training for m in self.p3_masf.modules() if isinstance(m, nn.BatchNorm2d)):
            raise ValueError('MASF BN running statistics must stay frozen')
        many = self.forward_head([self.p3_masf(x[0]), x[1], x[2]], **self.one2many)
        feature = GradientScale.apply(self.p3_masf(x[0].detach()), self.bridge_beta)
        one = self.forward_head([feature, x[1].detach(), x[2].detach()], **self.one2one)
        return {'one2many': many, 'one2one': one}


class MASFDetect(TaskBridge, Detect):
    pass


class MASFPose(TaskBridge, Pose26):
    pass


def attach(model, task):
    head = model.model[-1]
    expected = Detect if task == 'detect' else Pose26
    if type(head) is not expected or head.f != [16, 19, 22] or len(model.model) != 24:
        raise ValueError('unsupported task head or feature routing')
    if any(isinstance(m, P3MASFFull35) for m in model.modules()):
        raise ValueError('parent already contains MASF')
    channels = head.cv2[0][0].conv.in_channels
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        masf = P3MASFFull35(channels)
        for module in (masf.context.dw3, masf.context.dw5, masf.context.project):
            module.act = ShiftPiecewiseQuadraticSiLU()
        masf.alpha.data.zero_()
    head.__class__ = MASFDetect if task == 'detect' else MASFPose
    head.add_module('p3_masf', masf)
    head.bridge_beta = 1.0
    return model.eval()


@contextmanager
def without_masf(model):
    """只供契約稽核；finally 恢復，不建立永久 hook。"""
    head = model.model[-1]
    block = head.p3_masf
    del head.p3_masf
    try:
        yield
    finally:
        head.add_module('p3_masf', block)


def activation_contract():
    # JSON round-trip fixes tuple/list representation across save/reload.
    import json
    return json.loads(json.dumps(asdict(ShiftPiecewiseQuadraticProfile())))


def validate(model, task):
    head = model.model[-1]
    if type(head) is not (MASFDetect if task == 'detect' else MASFPose) or head.bridge_beta != 1.0:
        raise ValueError('MASF head/bridge contract changed')
    block = head.p3_masf
    if type(block) is not P3MASFFull35 or block.channels != 256 or head.f != [16, 19, 22]:
        raise ValueError('MASF placement/architecture mismatch')
    expected = ShiftPiecewiseQuadraticProfile()
    for module in (block.context.dw3, block.context.dw5, block.context.project):
        if type(module.act) is not ShiftPiecewiseQuadraticSiLU or module.act.profile != expected:
            raise ValueError('qSiLU class or coefficients changed')
    if not torch.isfinite(block.alpha):
        raise ValueError('nonfinite alpha')


def configure(model, scope):
    if scope != 'task-head-and-masf':
        raise ValueError(scope)
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('model.23.') and '.dfl.' not in name)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return [p for p in model.parameters() if p.requires_grad]


def folded_context(block):
    """僅供線性 BN folding 診斷；不折疊 qSiLU。"""
    from ultralytics.utils.torch_utils import fuse_conv_and_bn
    result = copy.deepcopy(block).eval()
    for module in (result.context.dw3, result.context.dw5, result.context.project):
        module.conv = fuse_conv_and_bn(module.conv, module.bn)
        del module.bn
        module.forward = module.forward_fuse
    return result
