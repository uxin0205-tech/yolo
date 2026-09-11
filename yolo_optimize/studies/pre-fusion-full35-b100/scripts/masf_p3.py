"""融合前 MASF 位置實驗；不靠永久 hook 或跨 forward 暫存特徵。"""
import copy
from common import SOURCE, registry, sha256
import torch
from ultralytics.nn.modules.head import Detect
from achitechure_1.model import graft_p3_masf, inspect_yolo26_graph


class P3MASFDetect(Detect):
    def forward(self, x):
        # 不修改原始 list，保留原生 one-to-one detach 與 loss／推論邏輯。
        return super().forward([self.p3_masf(x[0]), x[1], x[2]])


def get_masf(model):
    return getattr(model.model[16], 'p3_masf', None) or getattr(model.model[23], 'p3_masf', None)


def attach(model, variant):
    graph = inspect_yolo26_graph(model)
    assert graph.detect_inputs == (16, 19, 22) and len(model.model) == 24
    if get_masf(model) is not None:
        raise ValueError('parent 不應已有 MASF；禁止直接搬動已使用的 shared 模組')
    if variant == 'control':
        return model
    if variant not in ('shared', 'fork'):
        raise ValueError(variant)
    record = registry()
    source = SOURCE / record['bittrue']
    if sha256(source) != record['bittrue_sha256']:
        raise ValueError('MASF context 初始化來源不符')
    source_model = torch.load(source, map_location='cpu', weights_only=False)['model'].float()
    masf = copy.deepcopy(source_model.model[16].p3_masf)
    del source_model
    with torch.no_grad():
        masf.alpha.zero_()
    # 只重用 B100 context 初始化；零 gate 確保 late E8 原預測完全保留。
    if variant == 'shared':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260922)
            graft_p3_masf(model, 'full35')
        model.model[16].p3_masf = masf
    else:
        head = model.model[23]
        if type(head) is not Detect:
            raise TypeError('只接受已核對的原生 Detect')
        head.__class__ = P3MASFDetect
        head.add_module('p3_masf', masf)
    assert inspect_yolo26_graph(model).detect_inputs == (16, 19, 22)
    return model


def parameter_role(name):
    if '.p3_masf.' in name:
        return 'alpha' if name.endswith('.alpha') else 'context'
    if any(name.startswith(f'model.23.{part}.0.')
           for part in ('cv2', 'cv3', 'one2one_cv2', 'one2one_cv3')):
        return 'p3_head'
    return 'fixed'
