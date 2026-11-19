"""D1 的單一 KD 恢復政策；重用 binaryqk 完整 epoch／EMA／resume 引擎。"""

import copy
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from . import stage3_checkpoint as checkpoint
from .config import Stage3KDConfig
from .stage3_checkpoint import ARTIFACTS, ROOT, SITES, sha256
from .stage3_diagnostics import write_json
from .stage3_kd_loss import parts, protected_mask

REVIEW = ARTIFACTS/'kd-reference-review.json'
PARENT_SHA = 'a691a600715b58bf7d1438e379022443fde784c95ae75bf13336da28cc728a11'
TEACHER_SHA = '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'


def verify_reference():
    data = json.loads(REVIEW.read_text())
    if data['schema'] != 'attention-stage3-kd-review-v1' or not data['reviewed_before_implementation']:
        raise ValueError('KD requires prior optimize reference review')
    for path, digest in data['sources_sha256'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError(f'KD reference changed: {path}')
    for key, digest in (('parent', PARENT_SHA), ('teacher', TEACHER_SHA)):
        if data[key+'_sha256'] != digest or sha256(data[key]) != digest:
            raise ValueError(f'KD {key} SHA mismatch')
    return data


def metric_vector(report):
    official = report['official']
    result = {key: official[key] for key in ('AP50', 'AP50_95')}
    for name in ('person', 'sports_ball', 'baseball_bat'):
        result.update({name+'_'+k: official['classes'][name][k] for k in ('AP50', 'AP50_95')})
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result.values()):
        raise ValueError('all eight official AP metrics must be finite')
    return result


def eligible(report, baseline):
    values, base = metric_vector(report), metric_vector(baseline)
    config = Stage3KDConfig()
    return (values['AP50_95'] >= base['AP50_95']+config.min_gain
            and all(values[k] >= base[k]-config.max_drop for k in base))


def validate_adaptation(data, lineage, candidate):
    if candidate != 'D1' or lineage['parent_sha256'] != PARENT_SHA:
        raise ValueError('KD must inherit reviewed D1')
    if data.get('method') != 'spatial-attention-transfer-bat-exclusion-v1':
        raise ValueError('unknown KD method')
    expected = json.loads(json.dumps(asdict(Stage3KDConfig())))
    if data.get('config') != expected or data.get('teacher_sha256') != TEACHER_SHA:
        raise ValueError('KD teacher/config mismatch')
    for key in ('calibration_sha256', 'recipe_sha256', 'reference_review_sha256'):
        value = data.get(key, '')
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('KD provenance digest missing')
    if not math.isfinite(data.get('mu', float('nan'))) or not 0 < data['mu'] < 1e6:
        raise ValueError('invalid calibrated KD multiplier')
    if data.get('source_lineage', {}).get('phase') != 'normalization':
        raise ValueError('missing D1 original lineage')


def attach_provenance(model, calibration, calibration_sha, recipe_sha):
    data = json.loads(REVIEW.read_text())
    model.stage3_adaptation = {
        'method': 'spatial-attention-transfer-bat-exclusion-v1',
        'config': json.loads(json.dumps(asdict(Stage3KDConfig()))),
        'mu': calibration['mu'], 'teacher': data['teacher'], 'teacher_sha256': TEACHER_SHA,
        'calibration_sha256': calibration_sha, 'recipe_sha256': recipe_sha,
        'reference_review_sha256': sha256(REVIEW),
        'source_lineage': copy.deepcopy(model.research_lineage),
        'teacher_exported': False, 'affine_and_normalization_constants': 'frozen',
    }
    model.research_lineage = {'parent': data['parent'], 'parent_sha256': PARENT_SHA, 'phase': 'kd_recovery'}
    checkpoint.manifest_for(model)


def configure(model, scope):
    from binaryqk.scale_study import configure as configure_parent
    if scope != 'attention-head-fixed-affine':
        raise ValueError('KD scope must freeze affine')
    configure_parent(model, 'attention-and-detect-head')
    for name, parameter in model.named_parameters():
        if '.score.affine.' in name:
            parameter.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


def teacher_for(device):
    from binaryqk.model import load_parent
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        model = load_parent().to(device).eval()
    model.requires_grad_(False)
    if model.names[34] != 'baseball bat' or model.names[32] != 'sports ball':
        raise ValueError('KD COCO class mapping changed')
    return model


class Codec:
    load = staticmethod(checkpoint.load)

    @staticmethod
    def manifest_for(model, calibration_sha):
        return checkpoint.manifest_for(model)

    @staticmethod
    def save(path, model, manifest):
        if checkpoint.manifest_for(model) != manifest:
            raise ValueError('KD hardware/provenance contract drift')
        return checkpoint.save(path, model)


class KDStudy:
    codec = Codec
    verify = staticmethod(verify_reference)
    configure = staticmethod(configure)

    def __init__(self, calibration_path):
        self.calibration_path = Path(calibration_path)
        self.calibration = json.loads(self.calibration_path.read_text())
        if self.calibration.get('status') != 'passed' or self.calibration['student_sha256'] != PARENT_SHA:
            raise ValueError('train-only KD calibration required')
        self.teacher = None
        self.teacher_hash = None
        self.baseline = json.loads(Path(verify_reference()['parent_metrics']).read_text())

    def recipe(self, phase):
        from binaryqk.scale_study import recipe
        if phase != 'kd_recovery':
            raise ValueError('KD phase mismatch')
        return {**recipe(), 'version': 4, 'phase': phase, 'scope': 'attention-head-fixed-affine',
                'reference_review_sha256': sha256(REVIEW),
                'group_lrs': {'attention': 1e-5, 'head': 5e-6},
                'normalization': 'B2/S1 fixed affine + integer PWL + D1 reciprocal; Float-PWL backward proxy',
                'loss': 'native E2E sum + mu*masked spatial KD sum; macro normalization 64/actual images',
                'selection': 'overall gain >= .001 and no eight AP drops > .001 vs D1; else D1',
                'kd': {**asdict(Stage3KDConfig()), 'mu': self.calibration['mu'],
                       'calibration_sha256': sha256(self.calibration_path)}}

    def build(self, variant, phase, parent, calibration):
        import hashlib
        data = self.verify()
        if variant != 'D1-KD' or phase != 'kd_recovery' or str(Path(parent).resolve()) != data['parent']:
            raise ValueError('KD must start from selected D1')
        model, _ = checkpoint.load(parent)
        digest = hashlib.sha256(json.dumps(self.recipe(phase), sort_keys=True).encode()).hexdigest()
        attach_provenance(model, self.calibration, sha256(self.calibration_path), digest)
        return model

    @staticmethod
    def mutable(model):
        return set()

    @staticmethod
    def optimizer(model, recipe):
        from binaryqk.scale_study import optimizer
        return optimizer(model, recipe)

    @staticmethod
    def set_lr(*args):
        from binaryqk.scale_study import ScaleStudy
        return ScaleStudy.set_lr(*args)

    @staticmethod
    def gradient_names(model):
        return [f'{s}.qkv.{p}.conv.weight' for s in SITES for p in ('q', 'k')]

    def loss(self, model, batch):
        from binaryqk.model import frozen_hashes
        if self.teacher is None:
            self.teacher = teacher_for(batch['img'].device)
            self.teacher_hash = frozen_hashes(self.teacher)
        native, kd, _, items = parts(model, self.teacher, batch)
        return native+self.calibration['mu']*kd, items

    def accept_candidate(self, report, history):
        return eligible(report, self.baseline)

    def guard_metrics(self, report, history, smoke):
        from binaryqk.model import frozen_hashes
        if self.teacher is not None and frozen_hashes(self.teacher) != self.teacher_hash:
            raise ValueError('KD teacher weights or BN changed')
        if smoke:
            return
        values, base = metric_vector(report), metric_vector(self.baseline)
        if report.get('images') != 5000 or not report.get('formal'):
            raise ValueError('KD requires full val5000')
        if not history and any(abs(values[k]-base[k]) > 1e-8 for k in base):
            raise ValueError('KD epoch0 differs from frozen D1; stop before training')
        if any(values[k] < base[k]-.05 for k in base):
            raise ValueError('KD AP catastrophic regression; stop and retain D1')


def calibrate(output, callback):
    from binaryqk.data import dataset
    from binaryqk.model import frozen_hashes
    from binaryqk.run import loader, seed, transfer
    config = Stage3KDConfig()
    data = verify_reference()
    seed(1)
    ds, _, args = dataset('train', 16, epochs=5)
    model, _ = checkpoint.load(data['parent'])
    model.to('cuda:0')
    model.args = args
    configure(model, 'attention-head-fixed-affine')
    model.criterion = model.init_criterion()
    teacher = teacher_for(torch.device('cuda:0'))
    original_teacher = frozen_hashes(teacher)
    original_student = {n: checkpoint.buffer_digest(v) for n, v in model.state_dict().items()}
    records = []
    torch.cuda.reset_peak_memory_stats()
    batches = iter(loader(ds, 16, 1))
    for i in range(config.calibration_batches):
        batch = transfer(next(batches), 'cuda:0')
        # FP32 calibration avoids half-precision autograd.grad underflow; smoke checks actual AMP training.
        native, kd, features, _ = parts(model, teacher, batch)
        gn = torch.autograd.grad(native, features, retain_graph=True)
        gk = torch.autograd.grad(kd, features, retain_graph=True)
        n = float(sum(x.float().square().sum() for x in gn).sqrt())
        k = float(sum(x.float().square().sum() for x in gk).sqrt())
        if not all(math.isfinite(x) and x > 0 for x in (n, k)):
            raise ValueError('KD/native feature gradient missing')
        qk_names = KDStudy.gradient_names(model)
        qk = torch.autograd.grad(kd, [model.get_parameter(name) for name in qk_names])
        qk_norms = {name: float(g.float().norm()) for name, g in zip(qk_names, qk)}
        if any(not math.isfinite(v) or v <= 0 for v in qk_norms.values()):
            raise ValueError('KD does not reach both Q/K sites')
        row = {'batch': i+1, 'images': len(batch['img']), 'native_feature_gradient': n,
               'kd_feature_gradient': k, 'ratio': n/k, 'kd_per_image': float(kd.detach())/len(batch['img']),
               'kd_only_qk_gradients': qk_norms,
               'ball_instances': int((batch['cls'] == 32).sum()), 'bat_instances': int((batch['cls'] == 34).sum()),
               'excluded_feature_cells': [int((~protected_mask(batch, f.shape, config)).sum()) for f in features]}
        records.append(row)
        callback({'kind': 'kd-calibration', **row})
        del native, kd, features, gn, gk, qk
    mu = config.feature_gradient_target*statistics.median(r['ratio'] for r in records)
    ratios = [mu/r['ratio'] for r in records]
    if not 0 < mu < 1e6 or any(not .025 <= r <= .4 for r in ratios):
        raise ValueError(f'KD calibration gradient ratio outside gate: {ratios}')
    if frozen_hashes(teacher) != original_teacher:
        raise ValueError('KD teacher changed in calibration')
    if {n: checkpoint.buffer_digest(v) for n, v in model.state_dict().items()} != original_student:
        raise ValueError('KD calibration changed student')
    checkpoint.manifest_for(model)
    result = {'status': 'passed', 'mu': mu, 'records': records, 'weighted_feature_ratios': ratios,
              'student_sha256': PARENT_SHA, 'teacher_sha256': TEACHER_SHA,
              'teacher_state_unchanged': True, 'student_state_unchanged': True,
              'validation_used': False, 'dtype': 'FP32 gradients; AMP verified separately by smoke',
              'target': '.1 of native feature gradient; not whole-parameter gradient norm',
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(Path(output)/'calibration.json', result)
    return result
