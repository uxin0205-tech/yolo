"""原 AffineBranchScore 類別逐字抽取；不匯入 checkpoint／訓練流程。"""
import torch
from yolo_attention.config import FixedScaleFormat, ScoreAffineConfig
from yolo_attention.score_affine import QuantizedScoreAffine
from .basis import BASES, branch_dots
CANDIDATES = {"S1": FixedScaleFormat.Q10, "S2": FixedScaleFormat.POWER_OF_TWO}

class AffineBranchScore(torch.nn.Module):
    def __init__(self, candidate, coefficients):
        super().__init__()
        self.candidate, self.variant, self.bases = candidate, "B2", BASES["B2"]
        self.affine = QuantizedScoreAffine(ScoreAffineConfig(format=CANDIDATES[candidate]), coefficients)

    def forward(self, q, k, *, height=None, width=None):
        if q.shape != k.shape or q.ndim != 4 or q.shape[1:3] != (4, 32):
            raise ValueError("Stage 2 requires matching B2 Q/K shapes")
        if height is None or width is None or height * width != q.shape[-1]:
            raise ValueError("explicit H/W required; never infer square tokens")
        dots = list(branch_dots(q, k, self.bases, training=self.training))
        return self.affine(dots, height=height, width=width)
