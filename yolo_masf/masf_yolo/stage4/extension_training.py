"""同一 E5 resume 的兩個配對續訓方法；保留所有訓練狀態。"""

import copy
import math
from pathlib import Path

import torch
from binaryqk.train import _same

from . import checkpoint
from .common import ARTIFACTS, ROOT, read, sha256
from .training import MASFStudy, vector

REVIEW = ARTIFACTS / 'extension-reference.json'


def review():
    data = read(REVIEW)
    if not data['reviewed_before_implementation']:
        raise ValueError('prior optimize review required')
    for name, digest in data['reference_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'extension reference changed: {name}')
    for parent in data['parents'].values():
        for name in ('resume', 'inference'):
            if sha256(parent[name]) != parent[name + '_sha256']:
                raise ValueError(f'extension parent {name} changed')
    return data


class ContinuationStudy(MASFStudy):
    verify = staticmethod(review)

    def __init__(self, task, arm):
        if arm not in ('L1', 'L2'):
            raise ValueError(arm)
        self.reviewed = review()
        self.arm = arm
        original = Path(self.reviewed['original_run'])
        super().__init__(task, original / f'{task}-parent/metrics.json')
        self.initial_expected = read(original / f'{task}-training/metrics-05.json')
        self.source = self.reviewed['parents'][task]

    def payload(self):
        value = torch.load(self.source['resume'], map_location='cpu', weights_only=True)
        if value['schema'] != 'binaryqk-resume-v2' or value['next_epoch'] != 5:
            raise ValueError('extension requires completed E5 full resume')
        return value

    def recipe(self, phase):
        base = super().recipe('masf')
        if phase != 'masf-extension':
            raise ValueError(phase)
        return {**base, 'version': 6, 'phase': phase, 'arm': self.arm,
                'epochs': 15, 'seed': 6, 'warmup_epochs': 0, 'linear_final_fraction': .2,
                'group_lrs': copy.deepcopy(self.reviewed['protocol']['starting_lr'][self.arm]),
                'minimum_epochs': 5, 'observed_patience': 5, 'criterion_initial_age': 5,
                'criterion_terminal_weights': [.1, .9], 'reference_review_sha256': sha256(REVIEW),
                'continuation_resume': self.source['resume'], 'continuation_sha256': self.source['resume_sha256'],
                'loss': 'native E2E terminal .1/.9; original macro normalization, beta1; no clock reset',
                'selection': 'save best primary AP; report class safety and trained alpha-on/off separately',
                'start_state': 'exact live+EMA+AdamW moments+scaler+EMA age from original task E5'}

    def build(self, variant, phase, parent, calibration):
        payload = self.payload()
        model, manifest = checkpoint.load(self.source['inference'])
        if manifest != payload['manifest']:
            raise ValueError('E5 exported manifest differs from resume')
        model.load_state_dict(payload['live_state'], strict=True)
        model.stage4_manifest = {**manifest, 'parent': self.source['resume'],
                                 'parent_sha256': self.source['resume_sha256'],
                                 'continuation': {'arm': self.arm, 'source_epoch': 5,
                                    'original_task_parent': manifest['parent'],
                                    'original_task_parent_sha256': manifest['parent_sha256'],
                                    'live_and_ema_restored': True, 'reference_review_sha256': sha256(REVIEW),
                                    'maximum_total_epochs': 20, 'terminal_e2e_weights': [.1, .9]}}
        checkpoint.manifest_for(model)
        return model

    def initialize_state(self, model, ema, optimizer, scaler, recipe):
        payload = self.payload()
        current = optimizer.state_dict()['param_groups']
        saved = payload['optimizer']['param_groups']
        if len(current) != len(saved) or any(
                a['group_name'] != b['group_name'] or a['role'] != b['role'] or len(a['params']) != len(b['params'])
                for a, b in zip(current, saved)):
            raise ValueError('optimizer parameter grouping changed')
        model.load_state_dict(payload['live_state'], strict=True)
        ema.ema.load_state_dict(payload['ema_state'], strict=True)
        ema.updates = payload['ema_updates']
        optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        if not _same(model.state_dict(), {n: v.to(next(model.parameters()).device)
                                         for n, v in payload['live_state'].items()}):
            raise ValueError('live continuation changed')
        # The LR changes only after exact restoration; moments and step counts stay intact.
        for group in optimizer.param_groups:
            group['lr'] = recipe['group_lrs'][group['role']]

    @staticmethod
    def initialize_criterion(model, start_epoch):
        criterion = model.init_criterion()
        criterion.o2m = criterion.o2m_copy = criterion.final_o2m = .1
        criterion.o2o = .9
        criterion.updates = 5 + start_epoch
        model.criterion = criterion

    @staticmethod
    def set_lr(optimizer, recipe, epoch, position, step, smoke):
        current = epoch * recipe['steps_per_epoch'] + step
        total = recipe['epochs'] * recipe['steps_per_epoch']
        factor = 1. if smoke else .2 + .8 * .5 * (1 + math.cos(math.pi * current / max(total - 1, 1)))
        values = {}
        for group in optimizer.param_groups:
            group['lr'] = recipe['group_lrs'][group['role']] * factor
            values[group['group_name']] = group['lr']
        return values

    @staticmethod
    def should_stop(recipe, completed, best_epoch, history):
        return completed >= recipe['minimum_epochs'] and completed - best_epoch >= recipe['observed_patience']

    @staticmethod
    def accept_candidate(report, history):
        return True

    def guard_metrics(self, report, history, smoke):
        if smoke:
            return
        values, expected = vector(self.task, report), vector(self.task, self.initial_expected)
        if not history and any(abs(values[k] - expected[k]) > 1e-8 for k in expected):
            raise ValueError('continuation initial EMA no longer matches E5')
        if self.score(report) < self.score(self.baseline) - .02:
            raise ValueError('continuation catastrophic AP drop')
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError('nonfinite extension metric')
