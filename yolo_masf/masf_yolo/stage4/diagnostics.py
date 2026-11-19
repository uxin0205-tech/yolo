"""CPU 等價、資料流、部署分支梯度、qSiLU 與嚴格重載稽核。"""

import copy
import tempfile
from pathlib import Path

import torch
from binaryqk.model import frozen_hashes

from yolo_attention import stage3_checkpoint

from . import checkpoint
from .common import ARTIFACTS, parent, write_json
from .models import configure, folded_context


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in tensors(v)]
    if isinstance(value, (tuple, list)):
        return [x for v in value for x in tensors(v)]
    return []


def capture(model, image):
    features = {}
    handles = [model.model[i].register_forward_hook(
        lambda module, inputs, output, i=i: features.__setitem__(i, output.detach().clone())) for i in (16, 19, 22)]
    try:
        with torch.no_grad():
            output = tensors(model(image))
        return output, features
    finally:
        for handle in handles:
            handle.remove()


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    results = {}
    for task in ('detect', 'pose'):
        path, _ = parent(task)
        if task == 'detect':
            base, _ = stage3_checkpoint.load(path)
        else:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            base = (payload.get('ema') or payload['model']).float().eval()
        model = checkpoint.build(task)
        image = torch.randn(1, 3, 64, 96)
        before, raw = capture(base, image)
        after, same_raw = capture(model, image)
        assert len(before) == len(after) and all(torch.equal(a, b) for a, b in zip(before, after))
        assert all(torch.equal(raw[k], same_raw[k]) for k in raw)
        head, block = model.model[-1], model.model[-1].p3_masf
        block.alpha.data.fill_(.2)
        changed, unchanged_raw = capture(model, image)
        assert all(torch.equal(raw[k], unchanged_raw[k]) for k in raw)
        assert any(not torch.equal(a, b) for a, b in zip(before, changed))
        configure(model, 'task-head-and-masf')
        grads = []
        for beta in (1., .25):
            model.zero_grad(set_to_none=True)
            head.bridge_beta = beta
            features = [raw[i].detach().clone().requires_grad_() for i in (16, 19, 22)]
            out = head(features)['one2one']
            loss = out['boxes'].square().mean() + out['scores'].square().mean()
            loss.backward()
            assert all(x.grad is None for x in features)
            grads.append({n: p.grad.detach().clone() for n, p in head.named_parameters() if p.grad is not None})
        for name in grads[0]:
            factor = .25 if name.startswith('p3_masf.') else 1.
            assert torch.allclose(grads[1][name], grads[0][name] * factor, atol=1e-7, rtol=1e-5), name
        assert grads[0]['p3_masf.context.project.conv.weight'].norm() > 0
        head.bridge_beta = 1.
        block.alpha.data.zero_()
        model.zero_grad(set_to_none=True)
        pred = head([raw[i].detach() for i in (16, 19, 22)])['one2one']
        (pred['boxes'].square().mean() + pred['scores'].square().mean()).backward()
        assert block.alpha.grad.abs() > 0 and block.context.project.conv.weight.grad.abs().max() == 0
        block.alpha.data.fill_(.2)
        model.eval()
        with tempfile.TemporaryDirectory(prefix='masf-stage4-') as temp:
            p = Path(temp) / 'model.pt'
            checkpoint.save(p, model, checkpoint.manifest_for(model))
            restored, _ = checkpoint.load(p)
            assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
            reload_output, _ = capture(restored, image)
            assert all(torch.equal(a, b) for a, b in zip(changed, reload_output))
            restored.model[-1].p3_masf.context.dw3.act = torch.nn.SiLU()
            try:
                checkpoint.manifest_for(restored)
                raise AssertionError('qSiLU fallback accepted')
            except ValueError:
                pass
        fold_test = copy.deepcopy(block).eval()
        for m in fold_test.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-.2, .2)
                m.running_var.uniform_(.5, 1.5)
        folded = folded_context(fold_test)
        with torch.no_grad():
            error = float((fold_test(raw[16]) - folded(raw[16])).abs().max())
        assert error < 1e-4
        parameters = sum(p.numel() for p in block.parameters())
        frozen = frozen_hashes(model, set())
        results[task] = {'alpha_zero_exact': True, 'raw_p3_p4_p5_exact_with_alpha_nonzero': True,
                         'bridge_context_gradient_ratio': .25, 'bridge_head_gradient_ratio': 1.,
                         'one2one_no_trunk_gradient': True, 'zero_alpha_context_gradient_zero': True,
                         'strict_reload_exact': True, 'qsilu_fallback_rejected': True,
                         'nontrivial_bn_fold_max_abs_fp32': error, 'added_parameters': parameters,
                         'inference_masf_conv_MACs_640': (256 * 34 + 256 * 256) * 80 * 80,
                         'alpha_multiplications_640': 256 * 80 * 80, 'hardware_measured': False,
                         'frozen_tensor_count': len(frozen)}
        del model, base, restored
    write_json(ARTIFACTS / 'cpu-diagnostics.json', {'passed': True, 'tasks': results})
    return results
