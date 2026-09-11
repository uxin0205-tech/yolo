"""P2 零殘差、梯度、原 head 與完整初始 AP 的必要驗證。"""
from common import ROOT, setup, prepare_coco, write_json
import torch
from ultralytics.nn.modules.head import Detect
from masf_p2 import prepare, get_masf
from train_rep17 import training_mode
from verify_qk_challenger import tensors
from continue_a0 import validate
from pwl_contract import verify_pwl
import json


def main():
    output = ROOT / 'artifacts/masf-p2-preflight-v1'
    output.mkdir(parents=True, exist_ok=False)
    setup()
    torch.set_num_threads(4)
    control, candidate = prepare('control').eval(), prepare('p2').eval()
    verify_pwl(control)
    sites = verify_pwl(candidate)
    assert all(torch.equal(v, candidate.state_dict()[n]) for n, v in control.state_dict().items())
    image = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(20260930))
    with torch.inference_mode():
        expected, actual = tensors(control(image)), tensors(candidate(image))
    assert len(expected) == len(actual) and all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert sum(isinstance(m, Detect) for m in candidate.modules()) == 1
    assert candidate.model[23].f == [16, 19, 22] and candidate.model[23].stride.tolist() == [8., 16., 32.]
    masf = get_masf(candidate)
    for p in candidate.parameters():
        p.requires_grad_(False)
    for p in masf.parameters():
        p.requires_grad_(True)
    training_mode(candidate)
    with torch.no_grad():
        masf.alpha.fill_(.01)
    prediction = candidate(image)['one2many']
    objective = prediction['boxes'].square().mean() + prediction['scores'].square().mean()
    grads = torch.autograd.grad(objective, tuple(masf.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert float(grads[0].abs().sum()) > 0
    assert sum(float(g.square().sum()) for g in grads[1:]) > 0
    with torch.no_grad():
        masf.alpha.zero_()
    metrics = validate(candidate.eval(), 'bittrue', prepare_coco(), output / 'initial-p2')
    verified = json.loads((ROOT / 'artifacts/direction1-candidate-verification-v1/summary.json').read_text())
    reference = next(c['metrics'] for c in verified['candidates'] if c['name'] == 'control-e8')
    delta = {k: metrics[k] - reference[k] for k in reference}
    assert all(abs(x) <= 1e-8 for x in delta.values()), delta
    write_json(output / 'summary.json', {'status': 'passed', 'p2_layer': 2,
        'p2_channels': 256, 'p2_stride': 4, 'zero_alpha_forward_exact': True,
        'base_state_exact': True, 'nonzero_alpha_context_gradient_verified': True,
        'detect_module_count': 1, 'detect_inputs': [16, 19, 22], 'detect_strides': [8, 16, 32],
        'pwl': sites, 'initial_full_coco_images': 5000, 'initial_ap_delta': delta,
        'added_parameters': sum(p.numel() for p in masf.parameters()),
        'estimated_conv_macs_per_image640': 160 * 160 * (256 * 34 + 256 * 256),
        'mac_estimate_excludes': 'BN/activation/addition/memory traffic; not hardware latency'})
    print('JOB_DONE: P2 零殘差、梯度、原 Detect 及完整 AP 通過', flush=True)


if __name__ == '__main__':
    main()
