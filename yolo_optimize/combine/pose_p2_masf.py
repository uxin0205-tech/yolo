"""P2 共享特徵 MASF；Pose 監督適應，完整評估 COCO 副作用。"""
import copy
from dataclasses import replace
from types import MappingProxyType
from local_source import LocalSource
from j0_runtime import install as install_j0, PoseOnlyEMA, impl
import torch
from achitechure_1.masf import P3MASFFull35
from masf_p2 import P2MASFC3k2
from ultralytics.nn.modules.block import C3k2
import yolo_combine.stage_policy as policy
from yolo_combine.metrics import AccuracyGate


class P2Source(LocalSource):
    def provenance(self, kind='float'):
        return {**super().provenance(kind), 'masf_location': 'layer2 P2 shared, no additional head',
                'masf_seed': 20261002, 'supervision': 'Pose only, COCO full validation',
                'bn_policy': 'shared BN statistics and affine frozen; Pose head BN train'}

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        assert type(pair.detect.model[2]) is C3k2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261002)
            masf = P3MASFFull35(256)
        with torch.no_grad():
            masf.alpha.zero_()
        for model in (pair.detect, pair.pose):
            model.model[2].__class__ = P2MASFC3k2
            model.model[2].add_module('p2_masf', copy.deepcopy(masf))
        return pair


class P2EMA(PoseOnlyEMA):
    @torch.no_grad()
    def update(self, model):
        active = {n for n, p in model.named_parameters() if p.requires_grad}
        assert all('.pose_head.' in n or '.p2_masf.' in n for n in active)
        self.updates += 1
        decay = self.decay(self.updates)
        live = model.state_dict()
        for name, value in self.ema.state_dict().items():
            if ('.pose_head.' in name or name in active) and value.dtype.is_floating_point:
                value.mul_(decay).add_(live[name].detach(), alpha=1 - decay)


def install():
    config = install_j0()
    original = policy._classify
    def classify(name, parameter):
        result = original(name, parameter)
        return replace(result, role='masf') if '.p2_masf.' in name else result
    policy._classify = classify
    stage = replace(policy.JOINT_STAGES['j0'], learning_rates=MappingProxyType({
        **dict(policy.JOINT_STAGES['j0'].learning_rates), 'masf': 2e-4}))
    stages = MappingProxyType({**dict(policy.JOINT_STAGES), 'j0': stage})
    impl.JOINT_STAGES = stages
    impl.SourceBundle = P2Source
    impl.ModelEMA = P2EMA
    # P2 會改共享特徵，不能套用 P3-only 的 COCO 完全相同斷言。
    # 依使用者指示沿用 combine 八項 gate；最終仍逐項呈現 COCO 差值。
    impl.AccuracyGate = AccuracyGate
    return config, stages
