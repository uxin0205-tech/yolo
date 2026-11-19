"""第 3 階段定點 reference 與兩處真實 score/P/PV 診斷；不修改資料或 parent。"""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ReciprocalBackend, ReciprocalConfig, Stage3NormalizationConfig
from .normalization import CompletePWLSoftmax, build_stage3_normalizer
from .reciprocal import IntegerReciprocal
from .stage3_checkpoint import ARTIFACTS, SITES, review

CANDIDATES = tuple([f'N{i}' for i in range(1, 10)] + ['D1', 'D2', 'D3'])


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def numeric_reference(output=None):
    torch.set_num_threads(2)
    policy = review()
    denominators = [torch.arange(1, 65537)]
    # 每個可表示 Q1.16 mantissa，搭配大分母的量化邊界與前後鄰點。
    mantissa = torch.arange(65536, 131072)
    for exponent in range(16, 26):
        center = mantissa << (exponent-16)
        denominators.extend([center-1, center, center+1])
    values = torch.unique(torch.cat(denominators).clamp(1, 1 << 25))
    reciprocal = {}
    for name, backend in zip(('D1', 'D2', 'D3'), ReciprocalBackend):
        unit = IntegerReciprocal(ReciprocalConfig(backend=backend))
        code, exponent = unit(values)
        relative = (values.double()*code.double()*torch.pow(2., -16-exponent.double())-1).abs()
        maximum = relative.max().item()
        assert maximum < policy['gates']['reciprocal_relative_error_max'][name]
        reciprocal[name] = {'backend': backend.value, 'denominators_tested': values.numel(),
                            'relative_error_max': maximum, 'relative_error_mean': relative.mean().item(),
                            'worst_denominator': values[relative.argmax()].item(), 'passed': True,
                            'constants': unit.contract()}
    exp = CompletePWLSoftmax(ReciprocalConfig()).eval()
    scores = torch.arange(-128*1024, 1).reshape(1, -1).float()/1024
    # 分段純 exp 檢查與 normalization token 上限分開，避免將大 grid 當成一個合法 row。
    q8 = (torch.arange(-128*1024, 1)+2) >> 2
    weights = exp.exp.integer_weights(q8)
    approximation = weights.double()/32768
    truth = scores.flatten().double().exp()
    assert weights[-1] == 32768 and weights.min() >= 0 and weights.max() <= 32768
    result = {'schema': 'stage3-cpu-reference-v1', 'passed': True,
              'denominator_test_strategy': 'dense 1..65536, all Q1.16 mantissas and neighbor codes at exponents16..25',
              'reciprocal': reciprocal,
              'exp': {'range': [-10, 0], 'segments': 20, 'points_tested': q8.numel(),
                      'absolute_error_max': (approximation-truth).abs().max().item(),
                      'exp_zero': 1, 'endpoint_table': exp.exp.endpoint_table.tolist(),
                      'tail_saturation_code': weights[0].item(), 'endpoint_storage_bits': 336},
              'probability_rounding_bound': {'fraction_bits': 24, 'per_element': 2.**-25,
                                              'row_1024_worst_sum': 1024*2.**-25},
              'denominator_bits': {'N400': (400*32768).bit_length(), 'N1024': (1024*32768).bit_length()},
              'integer_runtime_tests': 'unit tests prohibit div/log/exp/frexp/ldexp in reciprocal path',
              'scope': 'CPU numerical reference; no GPU, task AP, HLS or board measurement'}
    write_json(output or ARTIFACTS/'cpu-reference.json', result)
    return result


class Accumulator:
    def __init__(self):
        self.data = {}
        self.score_ranges = {}
        self.exp_data = {}

    def exp_error(self, site, candidate, weights, exact, float_pwl):
        error = (weights-exact).double().abs()
        quantization = (weights-float_pwl).double().abs()
        rows = torch.stack([error.sum((0, 2, 3)), error.amax((0, 2, 3)),
                            quantization.sum((0, 2, 3)), quantization.amax((0, 2, 3))], dim=1).cpu().tolist()
        for head, row in enumerate(rows):
            item = self.exp_data.setdefault((site, candidate, head), {'count': 0, 'abs_sum': 0.,
                'max_error': 0., 'vs_float_sum': 0., 'vs_float_max': 0.})
            item['count'] += weights.shape[0]*weights.shape[2]*weights.shape[3]
            item['abs_sum'] += row[0]; item['max_error'] = max(item['max_error'], row[1])
            item['vs_float_sum'] += row[2]; item['vs_float_max'] = max(item['vs_float_max'], row[3])

    def update(self, site, candidate, p, reference, values, weights=None, reciprocal=None):
        # 所有 metrics 在 parent 的同一組 scores/V 上成對比較，不重新跑 parent AP。
        errors = (p-reference).double()
        pv = values.float() @ p.float().transpose(-2, -1)
        pv_ref = values.float() @ reference.float().transpose(-2, -1)
        pv_error = (pv-pv_ref).double()
        row_error = (p.double().sum(-1)-1).abs()
        axes = (0, 2, 3)
        columns = [errors.abs().sum(axes), errors.abs().amax(axes),
                   row_error.sum((0, 2)), row_error.amax((0, 2)),
                   pv_error.square().sum(axes), pv_ref.double().square().sum(axes),
                   pv_error.abs().sum(axes), pv_error.abs().amax(axes)]
        if weights is not None and reciprocal is not None:
            denominator = weights.sum(-1, keepdim=True)
            codes, exponent = reciprocal(denominator)
            r = codes.double()*torch.pow(2., -16-exponent.double())
            row_before_round = denominator.double()*r
            unrounded = weights.double()*r
            same_exp_exact = weights.double()/denominator.double()
            columns += [(row_before_round-1).abs().amax(axes),
                        (p.double()-unrounded).abs().amax(axes),
                        (p.double()-same_exp_exact).abs().amax(axes)]
        else:
            columns += [torch.zeros_like(columns[0]) for _ in range(3)]
        rows = torch.stack(columns, dim=1).cpu().tolist()
        for head, row in enumerate(rows):
            key = (site, candidate, head)
            entry = self.data.setdefault(key, {'sum': [0.]*len(row), 'p_count': 0, 'row_count': 0, 'pv_count': 0})
            for index, value in enumerate(row):
                if index in (1, 3, 7, 8, 9, 10):
                    entry['sum'][index] = max(entry['sum'][index], value)
                else:
                    entry['sum'][index] += value
            entry['p_count'] += p.shape[0]*p.shape[2]*p.shape[3]
            entry['row_count'] += p.shape[0]*p.shape[2]
            entry['pv_count'] += pv.shape[0]*pv.shape[2]*pv.shape[3]

    def scores(self, site, scores):
        centered = scores-scores.amax(-1, keepdim=True)
        exact = centered.softmax(-1)
        for head in range(scores.shape[1]):
            x = centered[:, head]
            p = exact[:, head]
            metrics = torch.stack([x.min(), x.max(), (x < -8).float().sum(), (x < -10).float().sum(),
                                   (p*(x < -8)).sum(), (p*(x < -10)).sum()]).cpu().tolist()
            key = (site, head)
            data = self.score_ranges.setdefault(key, {'min': 0., 'max': 0., 'count': 0, 'rows': 0,
                                                      'below8': 0., 'below10': 0., 'mass8': 0., 'mass10': 0.})
            data['min'] = min(data['min'], metrics[0]); data['max'] = max(data['max'], metrics[1])
            data['count'] += x.numel(); data['rows'] += x.shape[0]*x.shape[1]
            for name, value in zip(('below8', 'below10', 'mass8', 'mass10'), metrics[2:]):
                data[name] += value

    def summary(self):
        rows = []
        for (site, candidate, head), entry in self.data.items():
            a = entry['sum']
            rows.append({'site': site, 'candidate': candidate, 'head': head,
                         'probability_mae': a[0]/entry['p_count'], 'probability_max_error': a[1],
                         'row_sum_error_mean': a[2]/entry['row_count'], 'row_sum_error_max': a[3],
                         'pv_relative_l2': math.sqrt(a[4]/max(a[5], 1e-30)),
                         'pv_mae': a[6]/entry['pv_count'], 'pv_max_error': a[7],
                         'reciprocal_row_error_max': a[8], 'probability_quantization_error_max': a[9],
                         'probability_error_vs_same_exp_max': a[10]})
        return rows


@torch.inference_mode()
def real_data_diagnostics(output, emit, device='cuda:0'):
    from binaryqk.data import dataset
    from binaryqk.run import transfer
    from binaryqk.scale_checkpoint import load as load_parent
    policy = review()
    model, _ = load_parent(policy['parent'])
    model.to(device).eval()
    dataset_val, _, _ = dataset('val', batch=2)
    loader = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=2,
                        collate_fn=dataset_val.collate_fn, pin_memory=device.startswith('cuda'))
    normalizers = {name: build_stage3_normalizer(Stage3NormalizationConfig(name)).to(device).eval()
                   for name in CANDIDATES}
    collected, scores, handles, image_names = Accumulator(), {}, [], []

    def capture(site):
        def hook(_module, _args, output):
            scores[site] = output.detach()
        return hook

    def compare(site):
        def hook(_module, args):
            reference, values = args
            s = scores.pop(site)
            collected.scores(site, s)
            centered = s-s.amax(-1, keepdim=True)
            exact_exp = centered.exp()
            float_pwl = normalizers['N2'].approximate_weights(centered)
            for name, normalizer in normalizers.items():
                probability = normalizer(s)
                if not torch.isfinite(probability).all():
                    raise FloatingPointError(f'nonfinite {name} at {site}')
                complete = isinstance(normalizer, CompletePWLSoftmax)
                integer_weights = normalizer.exp.last_weights_int if complete else None
                collected.update(site, name, probability, reference, values,
                                 integer_weights,
                                 normalizer.reciprocal if complete else None)
                if name in {'N1', 'N2'}:
                    exp_weights = normalizer.approximate_weights(centered)
                elif name == 'N3' or complete:
                    codes = integer_weights if complete else normalizer.last_weights_int
                    exp_weights = codes.float()*(2.**-15)
                else:
                    continue
                collected.exp_error(site, name, exp_weights, exact_exp, float_pwl)
        return hook

    try:
        for site in SITES:
            module = model.get_submodule(site)
            handles += [module.score.register_forward_hook(capture(site)),
                        module.aggregate.register_forward_pre_hook(compare(site))]
        for raw in loader:
            batch = transfer(raw, device)
            model(batch['img'])
            image_names.extend(Path(path).name for path in raw['im_file'])
            emit({'kind': 'real-data-diagnostics', 'images': len(image_names), 'total': 16})
            if len(image_names) >= policy['gates']['diagnostic_images']:
                break
    finally:
        for handle in handles:
            handle.remove()
    rows = collected.summary()
    if len(rows) != len(SITES)*4*len(CANDIDATES) or len(image_names) != policy['gates']['diagnostic_images']:
        raise ValueError('incomplete two-site/head/image diagnostics')
    gates = {}
    for name in CANDIDATES:
        maximum = max(row['pv_relative_l2'] for row in rows if row['candidate'] == name)
        gates[name] = {'evaluate_full': maximum <= policy['gates']['real_data_pv_relative_l2_max_for_full_eval'],
                       'worst_site_head_pv_relative_l2': maximum,
                       'rule': 'worst-head PV relative L2 <=0.5; no AP inference from diagnostics'}
    result = {'images': image_names, 'images_count': len(image_names), 'sites': list(SITES),
              'rows': rows, 'gates': gates, 'parent_sha256': policy['parent_sha256'],
              'exp_components': [{'site': site, 'candidate': candidate, 'head': head,
                                   'exp_mae': data['abs_sum']/data['count'], 'exp_max_error': data['max_error'],
                                   'vs_float_pwl_mae': data['vs_float_sum']/data['count'],
                                   'vs_float_pwl_max_error': data['vs_float_max']}
                                  for (site, candidate, head), data in collected.exp_data.items()],
              'score_statistics': [{'site': site, 'head': head, **data,
                                    'mean_mass_below8': data['mass8']/data['rows'],
                                    'mean_mass_below10': data['mass10']/data['rows']}
                                   for (site, head), data in collected.score_ranges.items()],
              'formal_AP': False, 'exact_softmax_role': 'numerical reference only', 'next_stage_started': False}
    write_json(Path(output)/'diagnostics.json', result)
    return result
