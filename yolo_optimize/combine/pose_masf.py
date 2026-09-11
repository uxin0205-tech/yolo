"""Pose P3 專用 MASF：不改共享節點、Detect 或原生 one2one detach。"""
from local_source import LocalSource
import torch
from ultralytics.nn.modules.head import Pose26
from achitechure_1.masf import P3MASFFull35


class MASFHead(Pose26):
    def forward(self, x):
        features = list(x)
        features[0] = self.p3_masf(features[0])
        return super().forward(features)


class PoseMASFSource(LocalSource):
    def provenance(self, kind='float'):
        return {**super().provenance(kind), 'pose_masf': 'P3 pose-only; fresh context; alpha=0; native detach',
                'masf_seed': 20261002, 'comparison': 'j0-no-masf-v1; same initial trunk/head and J0 configuration'}

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        head = pair.pose.model[23]
        assert type(head) is Pose26
        channels = head.cv2[0][0].conv.in_channels
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261002)
            masf = P3MASFFull35(channels)
        with torch.no_grad():
            masf.alpha.zero_()
        head.__class__ = MASFHead
        head.add_module('p3_masf', masf)
        # 原已訓練 head 的 transfer 數目不含新建 MASF，來源報告明確區分。
        return pair
