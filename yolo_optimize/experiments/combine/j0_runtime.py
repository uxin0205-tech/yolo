"""只適用 J0 的本地適配；原融合專案保持唯讀。"""
from dataclasses import replace
import json
from local_source import HERE, LocalSource, initialize
import torch
from ultralytics.utils.torch_utils import ModelEMA
from yolo_combine.joint_config import JointExperimentConfig
from yolo_combine.metrics import AccuracyGate
import yolo_combine.formal_training as formal
import yolo_combine._formal_training_impl as impl
import yolo_combine.validation as validation
from validate import InternalValidator


def require_training_enabled():
    plan = json.loads((HERE / 'plan.json').read_text())
    if not plan.get('training_enabled') or plan.get('further_work_requires_user_masf_decision'):
        raise RuntimeError('目前只交付現有權重驗證；須等使用者決定 MASF，禁止啟動新訓練。')


class J0Config(JointExperimentConfig):
    def _validate(self):
        # 原公開設定限制完整 J0/J1/J2；保留其全部驗證，只縮小執行範圍。
        JointExperimentConfig._validate(replace(self, stages=('j0', 'j1', 'j2')))
        assert self.stages == ('j0',) and not self.enable_j3
        assert not self.shared_bn_affine_trainable


class PoseOnlyEMA(ModelEMA):
    """固定共享 trunk／Detect 的 EMA，避免對未訓練權重重做浮點加權。"""
    @torch.no_grad()
    def update(self, model):
        assert all('.pose_head.' in n for n, p in model.named_parameters() if p.requires_grad)
        self.updates += 1
        decay = self.decay(self.updates)
        live = model.state_dict()
        for name, value in self.ema.state_dict().items():
            if '.pose_head.' in name and value.dtype.is_floating_point:
                value.mul_(decay).add_(live[name].detach(), alpha=1 - decay)


class J0Gate(AccuracyGate):
    def evaluate(self, candidate):
        result = super().evaluate(candidate)
        # J0 所有 Detect state 都必須固定；不是允許 COCO 退步 0.08。
        changed = {k: d for k, d in result.deltas.items() if k.startswith('coco/') and abs(d) > 1e-8}
        if changed:
            raise AssertionError(f'J0 固定 Detect 的完整 AP 發生變動：{changed}')
        return result


def install():
    initialize()
    torch.set_num_threads(4)
    for name in ('no-masf-assembly-v1.json', 'baseline-v1/summary.json'):
        assert json.loads((HERE / 'artifacts' / name).read_text())['status'] == 'passed'
    impl.SourceBundle = LocalSource
    impl.ModelEMA = PoseOnlyEMA
    impl.AccuracyGate = J0Gate
    validation.DetectionValidator = InternalValidator
    config = J0Config.load(HERE / 'full35/configs/j0.yaml')
    report = config.preflight()
    assert report.ready, report
    return config
