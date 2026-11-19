"""整合主線自包含 checkpoint；必須保存 S1 與 normalization／reciprocal 契約。"""

import copy
import hashlib
import json
from pathlib import Path

import torch
from ultralytics.nn.tasks import DetectionModel

from .attention import HardwareFriendlyAttention
from .config import Stage3NormalizationConfig
from .normalization import CompletePWLSoftmax, build_stage3_normalizer

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT/'yolo_attention'
ARTIFACTS = PROJECT/'artifacts/stage3'
REVIEW = ARTIFACTS/'reference-review.json'
SITES = ('model.10.m.0.attn', 'model.22.m.0.1.attn')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def review():
    data = json.loads(REVIEW.read_text())
    if data.get('reviewed_before_implementation') is not True:
        raise ValueError('missing Stage 3 reference review')
    for path, digest in data['immutable_reference_sha256'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError(f'optimize reference changed: {path}')
    if sha256(data['parent']) != data['parent_sha256']:
        raise ValueError('Stage 3 parent SHA changed')
    return data


def install(model, candidate, parent_manifest, lineage):
    spec = Stage3NormalizationConfig(candidate)
    for site in SITES:
        source = model.get_submodule(site)
        model.set_submodule(site, HardwareFriendlyAttention.from_stage2(source, build_stage3_normalizer(spec)))
    model.stage3_candidate = candidate
    model.stage3_parent_manifest = copy.deepcopy(parent_manifest)
    model.research_lineage = copy.deepcopy(lineage)
    return model.eval()


def build(candidate):
    from binaryqk.scale_checkpoint import load as load_parent
    data = review()
    model, parent_manifest = load_parent(data['parent'])
    if parent_manifest['scale_candidate'] != 'S1':
        raise ValueError('Stage 3 requires the selected S1 parent')
    return install(model, candidate, parent_manifest,
                   {'parent': data['parent'], 'parent_sha256': data['parent_sha256'], 'phase': 'normalization'})


def buffer_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def normalizer_settings(module):
    from dataclasses import asdict, is_dataclass
    fields = ('score_floor', 'segments', 'score_step', 'score_min', 'top_k',
              'score_fraction_bits', 'endpoint_fraction_bits', 'fraction_bits', 'endpoint_bits')
    return {name: {'type': type(child).__name__,
                   **{key: getattr(child, key) for key in fields if hasattr(child, key)},
                   **({'config': asdict(child.config)} if is_dataclass(getattr(child, 'config', None)) else {})}
            for name, child in module.named_modules()}


def manifest_for(model):
    from binaryqk.checkpoint import activations
    from binaryqk.scale_checkpoint import validate_manifest as validate_parent
    parent = model.stage3_parent_manifest
    validate_parent(parent)
    spec = Stage3NormalizationConfig(model.stage3_candidate)
    expected = build_stage3_normalizer(spec).eval()
    expected_buffers = {name: buffer_digest(v) for name, v in expected.state_dict().items()}
    actual_sites = {name for name, m in model.named_modules() if isinstance(m, HardwareFriendlyAttention)}
    if actual_sites != set(SITES):
        raise ValueError('Stage 3 requires both Attention sites')
    for site in SITES:
        module = model.get_submodule(site)
        if type(module.normalizer) is not type(expected) or not getattr(module, 'integrated_stage3', False):
            raise ValueError('Stage 3 normalization fallback or graph mismatch')
        if {n: buffer_digest(v) for n, v in module.normalizer.state_dict().items()} != expected_buffers:
            raise ValueError('normalization deployment constants changed')
        if normalizer_settings(module.normalizer) != normalizer_settings(expected):
            raise ValueError('normalization numeric configuration changed')
        if module.score.affine.export_constants() != parent['constants'][site]:
            raise ValueError('Stage 2 fixed scale/bias contract changed')
    if activations(model) != parent['activation_calls']:
        raise ValueError('inherited activation changed')
    if model.names != parent['names']:
        raise ValueError('inherited class names changed')
    lineage = model.research_lineage
    digest = lineage.get('parent_sha256', '')
    if (lineage.get('phase') not in {'normalization', 'kd_recovery'} or not lineage.get('parent') or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('missing Stage 3 lineage')
    adaptation = None
    if lineage['phase'] == 'kd_recovery':
        adaptation = getattr(model, 'stage3_adaptation', None)
        if not isinstance(adaptation, dict):
            raise ValueError('missing KD adaptation contract')
        from .stage3_kd_training import validate_adaptation
        validate_adaptation(adaptation, lineage, spec.candidate)
    fixed = isinstance(expected, CompletePWLSoftmax)
    numerator = {'N1': 'float-pwl-8-16', 'N2': 'float-pwl-10-20', 'N3': 'integer-pwl-10-20',
                 'N4': 'score-indexed-lut', 'N5': 'power-of-two', 'N6': 'normalized-hard-sigmoid',
                 'N7': 'multimax-top1', 'N8': 'multimax-top3', 'N9': 'multimax-top5'}
    normalization = {'exp_backend': 'integer-pwl-10-20' if fixed else numerator[spec.candidate],
                     'reciprocal_backend': expected.reciprocal.config.backend.value if fixed else
                         ('none-stick-breaking' if spec.candidate in {'N7', 'N8', 'N9'} else 'native-float-division'),
                     'pwl_range': [-10, 0] if fixed or spec.candidate in {'N2', 'N3'} else
                         ([-8, 0] if spec.candidate == 'N1' else None),
                     'pwl_segments': 20 if fixed or spec.candidate in {'N2', 'N3'} else
                         (16 if spec.candidate == 'N1' else None),
                     'buffers_sha256': expected_buffers, 'max_tokens': 1024,
                     'probability_format': 'UQ1.24' if fixed else 'FP32',
                     'remaining_float': ['QKV', 'PV', 'projection', 'Conv', 'SiLU'],
                     'complete_fixed_normalization': fixed,
                     'backward_proxy': 'Float-PWL[-10,0]/20 with exact denominator' if fixed else 'not-trained'}
    if fixed:
        normalization['reciprocal'] = expected.reciprocal.contract()
        normalization['exp'] = {'input': 'S1 Q10 -> row-max -> Q8.8 (round ties positive infinity)',
                                'endpoints': expected.exp.endpoint_table.tolist(), 'format': 'UQ1.15',
                                'interpolation': 'full integer product then floor >>7',
                                'denominator': 'sum UQ1.15, uint26', 'tail': 'clamp to -10',
                                'exp_zero_code': 32768}
    result = {'schema': 'attention-stage3-v1', 'candidate': spec.candidate,
            'parent_manifest': copy.deepcopy(parent), 'normalization': normalization,
            'lineage': copy.deepcopy(model.research_lineage), 'activation_calls': activations(model),
            'next_stage_started': False}
    if adaptation is not None:
        result['adaptation'] = copy.deepcopy(adaptation)
    return result


def save(path, model):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    manifest = manifest_for(model)
    state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in state.values()):
        raise ValueError('checkpoint requires finite FP32 parameters')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'manifest': manifest, 'state_dict': state}, path)
    path.with_suffix('.json').write_text(json.dumps({'sha256': sha256(path), 'manifest': manifest},
                                                   ensure_ascii=False, indent=2)+'\n')
    return manifest


def load(path):
    from binaryqk.model import convert
    from binaryqk.scale_checkpoint import install as install_parent
    from binaryqk.scale_checkpoint import validate_manifest as validate_parent
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != metadata['sha256']:
        raise ValueError('Stage 3 checkpoint SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    manifest = payload['manifest']
    if (manifest.get('schema') != 'attention-stage3-v1'
            or json.loads(json.dumps(manifest)) != metadata['manifest']):
        raise ValueError('missing or mismatched Stage 3 manifest')
    parent = manifest['parent_manifest']
    validate_parent(parent)
    model = DetectionModel(copy.deepcopy(parent['architecture']), ch=3, nc=80, verbose=False)
    model.names = copy.deepcopy(parent['names'])
    convert(model, 'B2', parent['initial_coefficients'])
    install_parent(model, 'S1')
    install(model, manifest['candidate'], parent, manifest['lineage'])
    if 'adaptation' in manifest:
        model.stage3_adaptation = copy.deepcopy(manifest['adaptation'])
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in payload['state_dict'].values()):
        raise ValueError('nonfinite or non-FP32 Stage 3 checkpoint')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    if manifest_for(model) != manifest:
        raise ValueError('reloaded Stage 3 normalization/reciprocal contract mismatch')
    return model, manifest
