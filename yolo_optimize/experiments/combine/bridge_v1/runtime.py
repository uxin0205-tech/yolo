"""新 bridge J1 的局部政策；原 combine 程式保持唯讀。"""
from dataclasses import replace
import json
from types import MappingProxyType
from safe_source import HERE, initialize
from adapted_source import AdaptedBridgeSource
import torch
from ultralytics.utils.torch_utils import ModelEMA
import yolo_combine.formal_training as formal
import yolo_combine._formal_training_impl as impl
import yolo_combine.stage_policy as policy
import yolo_combine.validation as validation
from yolo_combine.joint_config import JointExperimentConfig
from yolo_combine.metrics import AccuracyGate
from validate import InternalValidator


class JointConfig(JointExperimentConfig):
    def _validate(self):
        JointExperimentConfig._validate(replace(self, stages=('j0', 'j1', 'j2')))
        assert self.stages == ('j1',) and not self.enable_j3


class ActiveEMA(ModelEMA):
    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        active = {n for n, p in model.named_parameters() if p.requires_grad}
        live = model.state_dict()
        for name, value in self.ema.state_dict().items():
            # 只更新有訓練的參數與原 head BN 統計；固定 MASF/shared BN 不做浮點混合。
            head_buffer = ('.detect_head.' in name or '.pose_head.' in name) and '.p3_masf.' not in name
            if value.dtype.is_floating_point and (name in active or head_buffer):
                value.mul_(d).add_(live[name].detach(), alpha=1-d)


class CocoProtectedGate(AccuracyGate):
    def evaluate(self, candidate):
        report = super().evaluate(candidate)
        extra = tuple(k for k, v in report.deltas.items() if k.startswith('coco/') and v < -.005 - 1e-12)
        failed = tuple(sorted(set(report.failed_metrics + extra)))
        return replace(report, passed=not failed, failed_metrics=failed)


def install():
    initialize()
    torch.set_num_threads(4)
    plan = json.loads((HERE / 'plan.json').read_text())
    assert plan['training_enabled_after_preflight'] and plan['user_selected'] == 'P3 bridge MASF E8'
    for name in ('preflight-v1.json', 'initial-v1/summary.json'):
        assert json.loads((HERE / 'artifacts' / name).read_text())['status'] == 'passed'
    original_classify = policy._classify
    def classify(name, parameter):
        result = original_classify(name, parameter)
        return replace(result, role='masf') if '.detect_head.p3_masf.' in name else result
    policy._classify = classify
    original_bn = policy._apply_bn_modes
    def bn_modes(model, stage):
        shared, heads = original_bn(model, stage)
        for m in model.detect_head.p3_masf.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                was_training = m.training
                m.eval()
                if was_training:
                    shared += 1
                    heads -= 1
        return shared, heads
    policy._apply_bn_modes = bn_modes
    # head／Neck 沿用 combine J1；MASF 沿用原研究 1e-5，alpha 同組可訓練。
    stage = replace(policy.JOINT_STAGES['j1'], learning_rates=MappingProxyType({
        **dict(policy.JOINT_STAGES['j1'].learning_rates), 'masf': 1e-5}))
    stages = MappingProxyType({**dict(policy.JOINT_STAGES), 'j1': stage})
    impl.JOINT_STAGES = stages
    impl.SourceBundle = AdaptedBridgeSource
    impl.ModelEMA = ActiveEMA
    impl.AccuracyGate = CocoProtectedGate
    validation.DetectionValidator = InternalValidator
    config = JointConfig.load(HERE / 'full35/configs/j1.yaml')
    assert config.preflight().ready, config.preflight()
    return config, stage


class SafetyStop(RuntimeError):
    pass


class Session(formal.FormalJointTrainingSession):
    def _resolved_config(self):
        result = super()._resolved_config()
        result['stage_policies']['j1']['learning_rates'] = dict(impl.JOINT_STAGES['j1'].learning_rates)
        result['masf_bn_statistics_frozen'] = True
        result['coco_max_drop'] = .005
        result['fresh_joint_optimizer_and_loss_horizon'] = 20
        return result

    def _save_selected(self, labels, **kwargs):
        outputs = super()._save_selected(labels, **kwargs)
        # 原 last 邊界先保存，再安全停止；不丟失已完成 epoch。
        baseline = self.config.load_baseline()
        failures = {k: kwargs['metrics'][k] - baseline[k] for k in baseline
                    if k.startswith('coco/') and kwargs['metrics'][k] < baseline[k] - .005}
        if failures:
            with (self.run_dir / 'safety-stop.json').open('x') as handle:
                json.dump({'status': 'SAFETY_STOP', 'delta': failures,
                    'saved': {k: str(v) for k, v in outputs.items()}}, handle, indent=2)
            raise SafetyStop(str(failures))
        return outputs
