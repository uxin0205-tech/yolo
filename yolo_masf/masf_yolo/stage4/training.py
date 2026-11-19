"""兩個任務共用既有完整 epoch、AMP retry、凍結 EMA／resume 引擎。"""

import math

import torch
from binaryqk.data import dataset
from binaryqk.evaluate import evaluate
from binaryqk.protocol import COMMON
from binaryqk.train import Stage1Policy
from yolo_attention.stage3_kd_training import metric_vector as detect_metrics

from . import checkpoint
from .common import REVIEW, read, sha256, verify_reference
from .data import pose_dataset
from .evaluate import evaluate_pose
from .models import configure


def vector(task, report):
    return detect_metrics(report) if task == 'detect' else report['metrics']


def score(task, report):
    if not report['formal']:
        return 0.
    return vector(task, report)['AP50_95' if task == 'detect' else 'pose_AP50_95']


class MASFStudy(Stage1Policy):
    codec = checkpoint
    verify = staticmethod(verify_reference)
    configure = staticmethod(configure)

    def __init__(self, task, baseline_path):
        self.task = task
        self.baseline = read(baseline_path)
        self.dataset = dataset if task == 'detect' else pose_dataset
        self.evaluate = evaluate if task == 'detect' else evaluate_pose
        self.expected_train_images = 118287 if task == 'detect' else 5964

    def recipe(self, phase):
        if phase != 'masf':
            raise ValueError(phase)
        logical = 256 if self.task == 'detect' else 128
        return {**COMMON, 'version': 5, 'phase': phase, 'task': self.task,
                'epochs': 5, 'patience': 0, 'seed': 1, 'scope': 'task-head-and-masf',
                'batch': 16, 'logical_batch': logical, 'reference_batch': 64,
                'steps_per_epoch': math.ceil(self.expected_train_images / logical),
                'lr0': 1e-5, 'group_lrs': {'head': 5e-6, 'context': 1e-5, 'alpha': 1e-4},
                'betas': [.948, .999], 'weight_decay': .00027, 'warmup_epochs': 1,
                'warmup_start_factor': .1, 'linear_final_fraction': .5, 'scheduler': 'warmup-cosine',
                'smoke_steps': 2, 'bn': 'all running statistics frozen',
                'ema': 'trainable parameters only; all frozen parameters and buffers bitwise fixed',
                'normalization': 'inherited B2/S1/integer PWL/D1 fixed' if self.task == 'detect' else 'inherited native Pose attention',
                'loss': 'native E2E; reference64/actual macro; beta1 before one2one head',
                'selection': 'best improved primary AP; no monitored AP drop >.001; else parent',
                'reference_review_sha256': sha256(REVIEW)}

    def build(self, variant, phase, parent, calibration):
        return checkpoint.build(self.task)

    @staticmethod
    def mutable(model):
        return set()

    @staticmethod
    def gradient_names(model):
        # alpha=0 時 context 第一個 step 的梯度必為零；head 與 alpha 應立即有梯度。
        return ['model.23.p3_masf.alpha', 'model.23.one2one_cv2.0.0.conv.weight']

    @staticmethod
    def update_names(model):
        return MASFStudy.gradient_names(model) + ['model.23.p3_masf.context.project.conv.weight']

    @staticmethod
    def optimizer(model, recipe):
        modules, groups = dict(model.named_modules()), {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            role = 'alpha' if name.endswith('.p3_masf.alpha') else 'context' if '.p3_masf.' in name else 'head'
            nodecay = role == 'alpha' or name.endswith('.bias') or isinstance(
                modules[name.rsplit('.', 1)[0]], torch.nn.modules.batchnorm._BatchNorm)
            key = (role, nodecay)
            group = groups.setdefault(key, {'params': [], 'role': role, 'lr': recipe['group_lrs'][role],
                                            'group_name': role + ('/no_decay' if nodecay else '/decay'),
                                            'weight_decay': 0. if nodecay else recipe['weight_decay']})
            group['params'].append(parameter)
        return torch.optim.AdamW(list(groups.values()), betas=tuple(recipe['betas']))

    @staticmethod
    def set_lr(optimizer, recipe, epoch, position, step, smoke):
        from binaryqk.scale_study import ScaleStudy
        return ScaleStudy.set_lr(optimizer, recipe, epoch, position, step, smoke)

    def score(self, report):
        return score(self.task, report)

    def accept_candidate(self, report, history):
        current, base = vector(self.task, report), vector(self.task, self.baseline)
        return all(current[k] >= base[k] - .001 for k in base)

    def guard_metrics(self, report, history, smoke):
        if smoke:
            return
        current, base = vector(self.task, report), vector(self.task, self.baseline)
        if any(not math.isfinite(v) for v in current.values()) or self.score(report) < self.score(self.baseline) - .02:
            raise ValueError('MASF nonfinite metrics or catastrophic AP drop')
        if not history and any(abs(current[k] - base[k]) > 1e-8 for k in base):
            raise ValueError('alpha-zero parent AP equivalence failed')
