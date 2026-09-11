"""獨立 challenger：精確二值前向，training-only signed-dot surrogate backward。

不修改來源 BinaryScore／guard；不新增推論 scale、selector 或 FP score。
"""
import torch
from yolo_attention.binary_basis import BinaryScore, fast_hadamard_transform
from yolo_attention.config import BasisKind, ScaleMode
import yolo_attention.binary_basis as binary


class ExactDotSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        ctx.save_for_backward(q, k)
        return binary.xnor_popcount_dot(q, k)

    @staticmethod
    def backward(ctx, gradient):
        q,k=ctx.saved_tensors
        return k @ gradient.transpose(-2,-1), q @ gradient


class TrainableBinaryScore(BinaryScore):
    def forward(self, q, k):
        if not self.training or not torch.is_grad_enabled():
            return super().forward(q,k)
        if self.basis is not BasisKind.HADAMARD or self.scale_mode is not ScaleMode.POWER_OF_TWO:
            raise ValueError('此 challenger 僅稽核 fixed-PoT Hadamard')
        if not self.use_ste or bool(self.calibration_enabled):
            raise ValueError('訓練 challenger 需要 clipped STE 且不可同時重新校準')
        scale=q.shape[-2]**-0.5
        qh=fast_hadamard_transform(q,dim=-2,normalize=True)
        kh=fast_hadamard_transform(k,dim=-2,normalize=True)
        z0=ExactDotSurrogate.apply(self._sign(q),self._sign(k))
        z1=ExactDotSurrogate.apply(self._sign(qh),self._sign(kh))
        return self._coefficient(0,q,k,scale)*z0 + self._coefficient(1,qh,kh,scale)*z1


def install(model):
    count=0
    for module in model.modules():
        if module.__class__.__name__!='HardwareFriendlyAttention':continue
        score=module.score
        if type(score) is not BinaryScore:
            raise TypeError('只接受未修改 B100 score')
        if score.basis is not BasisKind.HADAMARD or score.scale_mode is not ScaleMode.POWER_OF_TWO:
            raise ValueError('來源 score 契約不符')
        score.__class__=TrainableBinaryScore
        count+=1
    if count!=2:raise ValueError('必須明確安裝兩個 sites')
    return count


def remove(model):
    for module in model.modules():
        if isinstance(module,TrainableBinaryScore):module.__class__=BinaryScore
