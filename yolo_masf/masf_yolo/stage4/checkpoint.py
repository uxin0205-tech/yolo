"""自包含、FP32、manifest 強制驗證的第四階段 checkpoint。"""

import copy
import json
from pathlib import Path

import torch
from binaryqk.checkpoint import activations
from binaryqk.train import atomic_torch, state_cpu
from ultralytics.nn.tasks import DetectionModel, PoseModel

from yolo_attention import stage3_checkpoint as stage3

from .common import REVIEW, parent, read, sha256, write_json
from .models import activation_contract, attach, validate, without_masf


def build(task):
    path, digest = parent(task)
    if task == 'detect':
        model, upstream = stage3.load(path)
    else:
        payload = torch.load(path, map_location='cpu', weights_only=False)  # SHA-pinned trusted source only.
        model = (payload.get('ema') or payload['model']).float().eval()
        upstream = {'architecture': copy.deepcopy(model.yaml), 'names': copy.deepcopy(model.names),
                    'activation_calls': activations(model), 'attention': 'inherited native float'}
    if hasattr(model, 'criterion'):
        del model.criterion
    attach(model, task)
    model.stage4_manifest = {
        'schema': 'masf-stage4-v1', 'task': task, 'parent': str(path), 'parent_sha256': digest,
        'upstream': upstream, 'reference_review_sha256': sha256(REVIEW),
        'masf': {'placement': 'model.23.p3_masf', 'raw_p3': 'model.16', 'head_inputs': [16, 19, 22],
                 'channels': 256, 'context': 'DW3+DW5 -> pointwise', 'alpha_initial': 0.0,
                 'bridge_beta': 1.0, 'activation_class': 'ShiftPiecewiseQuadraticSiLU',
                 'activation_profile': activation_contract(), 'activation_calls': [
                     f'model.23.p3_masf.context.{p}.act' for p in ('dw3', 'dw5', 'project')],
                 'context_seed': 1, 'bn': 'unfolded, running stats frozen', 'arithmetic': 'FP32 reference'},
        'repconv': None, 'data': 'COCO2017 full' if task == 'detect' else 'bbat5-v1 immutable',
        'evaluator': 'COCO official val5000 FP32 E2E' if task == 'detect' else 'native Pose val683 FP32 E2E',
    }
    manifest_for(model)
    return model


def manifest_for(model, calibration_sha=None):
    data = model.stage4_manifest
    if data.get('schema') != 'masf-stage4-v1' or data['masf']['activation_profile'] != activation_contract():
        raise ValueError('missing or mismatched Stage4 manifest')
    validate(model, data['task'])
    expected = {'placement': 'model.23.p3_masf', 'raw_p3': 'model.16', 'head_inputs': [16, 19, 22],
                'channels': 256, 'context': 'DW3+DW5 -> pointwise', 'alpha_initial': 0.0,
                'bridge_beta': 1.0, 'activation_class': 'ShiftPiecewiseQuadraticSiLU',
                'activation_profile': activation_contract(), 'activation_calls': [
                    f'model.23.p3_masf.context.{p}.act' for p in ('dw3', 'dw5', 'project')],
                'context_seed': 1, 'bn': 'unfolded, running stats frozen', 'arithmetic': 'FP32 reference'}
    if data['masf'] != expected:
        raise ValueError('MASF numeric/placement manifest mismatch')
    with without_masf(model):
        if data['task'] == 'detect':
            if stage3.manifest_for(model) != data['upstream']:
                raise ValueError('inherited binary/scale/PWL/reciprocal/activation contract changed')
        elif activations(model) != data['upstream']['activation_calls'] or model.names != data['upstream']['names']:
            raise ValueError('inherited Pose class or activation changed')
    return copy.deepcopy(data)


def save(path, model, manifest):
    path = Path(path)
    if path.exists() or manifest_for(model) != manifest:
        raise ValueError('checkpoint exists or manifest changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    state = state_cpu(model)
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('checkpoint must be finite FP32')
    atomic_torch(path, {'manifest': manifest, 'state_dict': state})
    write_json(path.with_suffix('.json'), {'sha256': sha256(path), 'manifest': manifest,
               'deployment_alpha': float(model.model[-1].p3_masf.alpha.detach().cpu())})


def load(path):
    path = Path(path)
    metadata = read(path.with_suffix('.json'))
    if sha256(path) != metadata['sha256']:
        raise ValueError('Stage4 checkpoint SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    data = payload['manifest']
    if data.get('schema') != 'masf-stage4-v1' or json.loads(json.dumps(data)) != metadata['manifest']:
        raise ValueError('missing or mismatched Stage4 manifest')
    upstream = data['upstream']
    if data['task'] == 'detect':
        from binaryqk.model import convert
        from binaryqk.scale_checkpoint import install as install_affine
        base = upstream['parent_manifest']
        model = DetectionModel(copy.deepcopy(base['architecture']), ch=3, nc=80, verbose=False)
        model.names = copy.deepcopy(base['names'])
        convert(model, 'B2', base['initial_coefficients'])
        install_affine(model, 'S1')
        stage3.install(model, upstream['candidate'], base, upstream['lineage'])
        if 'adaptation' in upstream:
            model.stage3_adaptation = copy.deepcopy(upstream['adaptation'])
    elif data['task'] == 'pose':
        model = PoseModel(copy.deepcopy(upstream['architecture']), ch=3, nc=2, data_kpt_shape=(2, 3), verbose=False)
        model.names = copy.deepcopy(upstream['names'])
    else:
        raise ValueError('unknown task')
    attach(model, data['task'])
    model.stage4_manifest = copy.deepcopy(data)
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in payload['state_dict'].values()):
        raise ValueError('invalid Stage4 tensor')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    if manifest_for(model) != data:
        raise ValueError('Stage4 reloaded contract mismatch')
    return model, data
