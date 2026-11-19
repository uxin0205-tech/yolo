"""qSiLU／Rep-only SiLU 的訓練政策與完整 D1 部署契約。"""

from collections import defaultdict
from torch import nn
from activation_lab.activations import build_activation, get_profile
from binaryqk.basis import BASES
from binaryqk.model import ProbabilityValue
from binaryqk.scale_checkpoint import AffineBranchScore, CANDIDATES, validate_manifest
from stage7 import model as previous
from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.config import ScoreAffineConfig, Stage3NormalizationConfig
from yolo_attention.normalization import build_stage3_normalizer, PiecewiseLinearSoftmax
from yolo_attention.stage3_checkpoint import buffer_digest, normalizer_settings

SITES = ('graph.model.10.m.0.attn', 'graph.model.22.m.0.1.attn')
MODES = ('float_pwl', 'rep_silu_pwl', 'hardware')


def activation_policy(model, mode, layers):
    if mode not in MODES:
        raise ValueError('缺少明確的訓練／部署算子模式')
    policy = model.stage7_manifest['activation']
    selected = {f'graph.model.{i}.act' for i in layers} if mode == 'rep_silu_pwl' else set()
    groups, assignments = [], {}
    for group in policy['alias_groups']:
        pieces = defaultdict(list)
        for path in group:
            target = 'silu' if path in selected else 'qsilu_pq'
            assignments[path] = target
            pieces[target].append(path)
        groups.extend(pieces.values())
    return {'sites': assignments, 'alias_groups': sorted(sorted(g) for g in groups),
            'call_sites': policy['call_sites'], 'silu_scope': sorted(selected)}


def set_operators(model, mode, layers=None):
    layers = layers if layers is not None else model.stage8_manifest['layers']
    actual = activation_policy(model, mode, layers)
    for group in actual['alias_groups']:
        target = actual['sites'][group[0]]
        activation = nn.SiLU(inplace=False) if target == 'silu' else build_activation(target)
        for path in group:
            model.set_submodule(path, activation)
    for path in SITES:
        attention = model.get_submodule(path)
        normalizer = (build_stage3_normalizer(Stage3NormalizationConfig('D1')) if mode == 'hardware'
                      else PiecewiseLinearSoftmax(score_floor=-10, segments=20))
        attention.normalizer = normalizer.to(next(attention.parameters()).device)
    return actual


def validate_operators(model, mode):
    data = model.stage7_manifest
    base = data['parent_manifest']
    if (data['schema'] != 'activation-stage7-v1' or model.stage6_manifest != base
            or model.contract() != base['graph'] or base['repconv'] is not None):
        raise ValueError('上游共享圖／manifest 改變')
    if any('masf' in n.lower() or 'hog_aux' in n for n, _ in model.named_modules()):
        raise ValueError('不可加入未選 MASF／HOG')
    parent = base['upstream']['parent_manifest']
    validate_manifest(parent)
    sites = {n for n, m in model.named_modules() if isinstance(m, HardwareFriendlyAttention)}
    if sites != set(SITES):
        raise ValueError('Attention sites 改變')
    for path in SITES:
        attn = model.get_submodule(path)
        score = attn.score
        if (type(attn) is not HardwareFriendlyAttention or not attn.integrated_stage3
                or (attn.num_heads, attn.key_dim, attn.head_dim) != (4, 32, 64)
                or type(score) is not AffineBranchScore or score.variant != 'B2'
                or score.bases != BASES['B2'] or score.candidate != 'S1'
                or score.affine.config != ScoreAffineConfig(format=CANDIDATES['S1'])
                or type(attn.aggregate) is not ProbabilityValue
                or score.affine.export_constants() != parent['constants'][path.removeprefix('graph.')]):
            raise ValueError('B2／S1／Attention 契約改變')
    expected = activation_policy(model, mode, model.stage8_manifest['layers'])
    if model.stage8_manifest['activation_policy'] != expected:
        raise ValueError('activation manifest 不符')
    actual = {n for n, m in model.named_modules(remove_duplicate=False)
              if isinstance(m, (nn.SiLU, nn.Hardswish)) or type(m).__module__ == 'activation_lab.activations'}
    if actual != set(expected['sites']) or previous.aliases(model, actual) != expected['alias_groups']:
        raise ValueError('activation sites／alias 不符')
    for path, name in expected['sites'].items():
        module = model.get_submodule(path)
        if name == 'silu':
            if type(module) is not nn.SiLU or module.inplace:
                raise ValueError('Rep SiLU 必須明確定義，不能擴散到其他位置')
        elif type(module) is not type(build_activation(name)) or module.profile != get_profile(name):
            raise ValueError('qSiLU 類別／固定係數改變')
    if mode == 'hardware':
        previous.manifest_for(model)
    else:
        expected_norm = PiecewiseLinearSoftmax(score_floor=-10, segments=20)
        for path in SITES:
            norm = model.get_submodule(path).normalizer
            if (type(norm) is not PiecewiseLinearSoftmax
                    or normalizer_settings(norm) != normalizer_settings(expected_norm)
                    or {n: buffer_digest(v) for n,v in norm.state_dict().items()}
                    != {n: buffer_digest(v) for n,v in expected_norm.state_dict().items()}):
                raise ValueError('訓練 Float PWL 常數／配置改變')
