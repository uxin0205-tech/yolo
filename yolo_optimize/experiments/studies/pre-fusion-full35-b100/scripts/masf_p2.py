"""最後一個 MASF 候選：P2 backbone 特徵增強，不新增 Detect 尺度。"""
import copy
from common import ROOT, SOURCE, sha256
import torch
from ultralytics.nn.modules.block import C3k2
from achitechure_1.masf import P3MASFFull35
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from qk_challenger import install, remove
from pwl_contract import verify_pwl

PARENT = ROOT / 'artifacts/masf-head-control-v1/epoch-08-resume.pt'
PARENT_SHA = '3aeeaec2b379d465ceb5aee6051e5c11772e856497676ec9c87f2cb511a0c06c'
EMA_AGE = 14800
RATES = {'head': 1e-5, 'context': 1e-5, 'alpha': 1e-4}


class P2MASFC3k2(C3k2):
    def forward(self, x):
        return self.p2_masf(super().forward(x))


def get_masf(model):
    return getattr(model.model[2], 'p2_masf', None)


def role(name):
    if '.p2_masf.' in name:
        return 'alpha' if name.endswith('.alpha') else 'context'
    return 'head' if name.startswith('model.23.') else 'fixed'


def prepare(variant):
    assert variant in ('control', 'p2') and sha256(PARENT) == PARENT_SHA
    payload = torch.load(PARENT, map_location='cpu', weights_only=False)
    assert payload['epoch'] == 8 and payload['ema_updates'] == EMA_AGE
    model = copy.deepcopy(payload['ema']).float()
    remove(model)
    if hasattr(model, 'criterion'):
        del model.criterion
    convert_yolo26_model(model, VariantConfig.from_yaml(SOURCE / 'configs/attention/float-pwl-final.yaml'))
    install(model)
    verify_pwl(model)
    assert len(model.model) == 24 and model.model[23].f == [16, 19, 22]
    assert model.model[23].stride.tolist() == [8., 16., 32.]
    assert type(model.model[2]) is C3k2
    assert not any('masf' in name for name, _ in model.named_modules())
    if variant == 'p2':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260930)
            # 模組名稱沿用來源實作，通道數相符，但不移植 P3 已訓練的 context。
            masf = P3MASFFull35(256)
        with torch.no_grad():
            masf.alpha.zero_()
        model.model[2].__class__ = P2MASFC3k2
        model.model[2].add_module('p2_masf', masf)
    return model
