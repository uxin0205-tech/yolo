"""先核對 optimize 的 Stage 2 訓練政策；共用既有訓練 engine。"""

import math
from pathlib import Path

import torch

from . import checkpoint, scale_checkpoint
from .model import ROOT, SITES, sha256
from .protocol import COMMON
from .train import Stage1Policy

REVIEW = ROOT / 'yolo_binaryqk/artifacts/stage2/reference-review.json'


def verify_reference():
    import json
    review = json.loads(REVIEW.read_text())
    if review['schema'] != 'binaryqk-stage2-reference-review-v1' or not review['reviewed_before_implementation']:
        raise ValueError('missing Stage 2 reference review')
    for path, digest in review['sources_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'reference changed: {path}')
    if sha256(review['parent']) != review['parent_sha256']:
        raise ValueError('Stage 2 parent SHA changed')
    return review


def recipe(phase='scale_bias'):
    if phase != 'scale_bias':
        raise ValueError('Stage 2 phase must be scale_bias')
    return {**COMMON, 'version': 3, 'phase': phase, 'epochs': 5, 'patience': 0,
            'scope': 'attention-and-detect-head', 'seed': 1, 'logical_batch': 256, 'reference_batch': 64,
            'lr0': 1e-5, 'betas': [.948, .999], 'weight_decay': .00027,
            'warmup_epochs': 1, 'warmup_start_factor': .1, 'linear_final_fraction': .5,
            'scheduler': 'warmup-cosine', 'steps_per_epoch': 463,
            'group_lrs': {'attention': 1e-5, 'head': 5e-6, 'scale': 2e-4, 'bias': 1e-5},
            'bn': 'all running statistics frozen; selected affine trainable',
            'loss': 'native E2E batch-sum * 64/actual macro images; 16x16; retain final partial macro',
            'ema': 'native .9999 tau2000; trainable parameters only; all buffers fixed',
            'normalization': 'inherited native softmax; integer Q10 score affine forward',
            'reference_review_sha256': sha256(REVIEW), 'smoke_steps': 2}


def configure(model, scope):
    if scope != 'attention-and-detect-head':
        raise ValueError('unsupported Stage 2 scope')
    model.train()
    head = f'model.{len(model.model) - 1}.'
    prefixes = (*[f'{site}.' for site in SITES], head)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes) and '.dfl.' not in name)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    return [p for p in model.parameters() if p.requires_grad]


def optimizer(model, cfg):
    modules = dict(model.named_modules())
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = 'head'
        if any(name.startswith(site + '.') for site in SITES):
            role = 'scale' if name.endswith('.affine.coefficients') else 'bias' if '.affine.table_' in name else 'attention'
        no_decay = (role in {'scale', 'bias'} or name.endswith('.bias')
                    or isinstance(modules[name.rsplit('.', 1)[0]], torch.nn.modules.batchnorm._BatchNorm))
        key = role + ('/no_decay' if no_decay else '/decay')
        group = groups.setdefault(key, {'params': [], 'param_names': [], 'group_name': key, 'role': role,
                                       'lr': cfg['group_lrs'][role], 'weight_decay': 0. if no_decay else cfg['weight_decay']})
        group['params'].append(parameter)
        group['param_names'].append(name)
    return torch.optim.AdamW(list(groups.values()), betas=tuple(cfg['betas']))


class ScaleStudy(Stage1Policy):
    codec = scale_checkpoint
    verify = staticmethod(verify_reference)
    recipe = staticmethod(recipe)
    configure = staticmethod(configure)
    optimizer = staticmethod(optimizer)
    source_paths = (ROOT / 'yolo_attention/src/yolo_attention/config.py',
                    ROOT / 'yolo_attention/src/yolo_attention/score_affine.py')

    @staticmethod
    def mutable(model):
        return set()

    @staticmethod
    def build(candidate, phase, parent, calibration):
        review = verify_reference()
        source = Path(parent or review['parent']).resolve()
        if phase != 'scale_bias' or str(source) != review['parent'] or sha256(source) != review['parent_sha256']:
            raise ValueError('both candidates require the same reviewed B2 parent')
        model, manifest = checkpoint.load(source)
        if manifest['variant'] != 'B2' or manifest['coefficients'] != calibration['coefficients']['B2']:
            raise ValueError('Stage 2 parent does not match selected B2')
        scale_checkpoint.install(model, candidate)
        model.research_lineage = {'parent': str(source), 'parent_sha256': review['parent_sha256'], 'phase': phase}
        return model

    @staticmethod
    def gradient_names(model):
        return [f'{s}.{suffix}' for s in SITES for suffix in (
            'qkv.q.conv.weight', 'qkv.k.conv.weight', 'score.affine.coefficients',
            'score.affine.table_x', 'score.affine.table_y')]

    @staticmethod
    def set_lr(optimizer, cfg, epoch, position, step, smoke):
        current = epoch * cfg['steps_per_epoch'] + step
        warmup = cfg['warmup_epochs'] * cfg['steps_per_epoch']
        total = cfg['epochs'] * cfg['steps_per_epoch']
        if smoke:
            factor = 1.
        elif current < warmup:
            factor = cfg['warmup_start_factor'] + (1 - cfg['warmup_start_factor']) * (current + 1) / warmup
        else:
            progress = (current - warmup) / (total - warmup - 1)
            factor = cfg['linear_final_fraction'] + (1 - cfg['linear_final_fraction']) * .5 * (1 + math.cos(math.pi * progress))
        values = {}
        for group in optimizer.param_groups:
            group['lr'] = cfg['group_lrs'][group['role']] * factor
            values[group['group_name']] = group['lr']
        return values

    @staticmethod
    def guard_metrics(report, history, smoke):
        if smoke:
            return
        value = report['official']['AP50_95']
        if not history:
            floor = verify_reference()['parent_official_AP50_95'] * .9
            if value < floor:
                raise ValueError('Stage 2 initial AP below parent 90%; diagnose before training')
        elif value < history[0]['evaluation']['official']['AP50_95'] - .05:
            raise ValueError('Stage 2 AP fell over .05 from own epoch0; diagnose before continuing')
