"""實際整圖驗證零殘差等價與 fork 的 P4／P5 隔離，再核對完整 COCO。"""
import json
from common import ROOT, setup, prepare_coco, write_json
import torch
from train_masf_p3 import prepare_model, PARENT_SHA
from masf_p3 import get_masf
from probe_b100_masf_off import capture
from verify_qk_challenger import tensors
from continue_a0 import validate


def main():
    output = ROOT / 'artifacts/masf-p3-preflight-v1'
    output.mkdir(parents=True, exist_ok=False)
    setup()
    torch.set_num_threads(4)
    data = prepare_coco()
    parent = prepare_model('control').eval()
    image = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(20260922))
    with torch.inference_mode():
        expected = tensors(parent(image))
    original = parent.state_dict()
    raw = capture(parent, image)
    reference = json.loads((ROOT / 'artifacts/a0-scope-late-v1/summary.json').read_text())['epochs'][-1]['ema']
    report = {'status': 'running', 'parent_sha256': PARENT_SHA, 'arms': {},
              'full_coco_images': 5000, 'initial_ap_tolerance': 1e-8, 'training_performed': False}
    contexts = {}
    for variant in ('shared', 'fork'):
        model = prepare_model(variant).eval()
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
        with torch.inference_mode():
            actual = tensors(model(image))
        assert len(actual) == len(expected)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), variant
        masf = get_masf(model)
        contexts[variant] = {name: value.clone() for name, value in masf.context.state_dict().items()}
        with torch.no_grad():
            masf.alpha.fill_(.1)
        changed = capture(model, image)
        if variant == 'fork':
            # 包含 layer16 raw P3；新 MASF 只在 Detect 入口執行。
            assert all(torch.equal(raw[index], changed[index]) for index in (16, 19, 22))
            with torch.inference_mode():
                raw_preds = parent(image)[1]
                fork_preds = model(image)[1]
            p3_cells = raw[16].shape[-2] * raw[16].shape[-1]
            for branch in ('one2many', 'one2one'):
                for kind in ('boxes', 'scores'):
                    assert torch.equal(raw_preds[branch][kind][..., p3_cells:],
                                       fork_preds[branch][kind][..., p3_cells:])
            assert any(not torch.equal(a, b) for a, b in zip(tensors(raw_preds), tensors(fork_preds)))
        else:
            assert all(not torch.equal(raw[index], changed[index]) for index in (16, 19, 22))
        with torch.no_grad():
            masf.alpha.zero_()
        metrics = validate(model, 'bittrue', data, output / variant)
        delta = {key: metrics[key] - reference[key] for key in reference}
        assert all(abs(value) <= 1e-8 for value in delta.values()), delta
        report['arms'][variant] = {'initial_cpu160_exact': True, 'base_state_exact': True,
                                   'p4_p5_raw_and_logits_isolated': variant == 'fork',
                                   'nonzero_alpha_probe': .1, 'initial_metrics': metrics,
                                   'initial_delta': delta}
        write_json(output / 'summary.json', report)
        del model
    assert contexts['shared'].keys() == contexts['fork'].keys()
    assert all(torch.equal(value, contexts['fork'][name]) for name, value in contexts['shared'].items())
    report.update(status='passed', shared_fork_context_initialization_exact=True)
    write_json(output / 'summary.json', report)
    print('JOB_DONE MASF_P3_PREFLIGHT', flush=True)


if __name__ == '__main__':
    main()
