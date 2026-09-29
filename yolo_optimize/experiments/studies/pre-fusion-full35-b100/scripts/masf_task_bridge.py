"""僅訓練時讓部署分支監督 MASF；推論仍走原本單次 MASF。"""
import torch
from masf_p3 import P3MASFDetect


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = coefficient
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.coefficient, None


class TaskAlignedMASFDetect(P3MASFDetect):
    def forward(self, x):
        if not self.training:
            return super().forward(x)
        assert self.end2end and 0 < self.bridge_coefficient <= .25
        assert not any(m.training for m in self.p3_masf.modules()
                       if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
        many = self.forward_head([self.p3_masf(x[0]), x[1], x[2]], **self.one2many)
        feature = self.p3_masf(x[0].detach())
        feature = _GradientScale.apply(feature, self.bridge_coefficient)
        one = self.forward_head([feature, x[1].detach(), x[2].detach()], **self.one2one)
        return {'one2many': many, 'one2one': one}
