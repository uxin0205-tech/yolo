"""融合前 B100 的 shared MASF 零殘差診斷；不訓練、不搬移已學模組。"""
import json
from common import ROOT, SOURCE, registry, setup, prepare_coco, sha256, write_json
import torch
from ultralytics import YOLO
from achitechure_1.model import inspect_yolo26_graph
from continue_a0 import validate


def capture(model, sample):
    features = {}
    handles = []
    try:
        for index in (16, 19, 22):
            def save_feature(module, inputs, output, key=index):
                features[key] = output.detach().clone()
            handles.append(model.model[index].register_forward_hook(save_feature))
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    assert set(features) == {16, 19, 22}
    return features


def main():
    output = ROOT / 'artifacts/b100-masf-off-probe-v1'
    output.mkdir(parents=True, exist_ok=False)
    setup()
    torch.set_num_threads(8)
    record = registry()
    source = SOURCE / record['bittrue']
    if sha256(source) != record['bittrue_sha256']:
        raise ValueError('B100 來源雜湊不符')
    baseline = json.loads((ROOT / 'artifacts/baseline-bittrue-v2/summary.json').read_text())
    assert baseline['sha256'] == record['bittrue_sha256'] and baseline['images'] == 5000
    model = YOLO(str(source)).model.float().eval()
    graph = inspect_yolo26_graph(model)
    assert graph.p3_index == 16 and graph.detect_inputs == (16, 19, 22)
    masf = model.model[16].p3_masf
    assert type(masf).__name__ == 'P3MASFFull35' and masf.alpha.numel() == 1
    original = {name: value.clone() for name, value in model.state_dict().items()}
    original_alpha = float(masf.alpha.detach())
    assert original_alpha != 0 and torch.isfinite(masf.alpha).all()
    generator = torch.Generator().manual_seed(20260921)
    sample = torch.rand(1, 3, 160, 160, generator=generator)
    before = capture(model, sample)
    with torch.no_grad():
        masf.alpha.zero_()
    after = capture(model, sample)
    changed = [name for name, value in original.items()
               if not torch.equal(value, model.state_dict()[name])]
    assert changed == ['model.16.p3_masf.alpha'], changed
    feature = torch.randn(1, masf.channels, 20, 20, generator=generator)
    with torch.inference_mode():
        assert torch.equal(masf(feature), feature)
    propagation = {str(index): {
        'changed': not torch.equal(before[index], after[index]),
        'relative_l2': float((after[index] - before[index]).norm()
                             / before[index].norm().clamp_min(1e-12)),
    } for index in before}
    # 診斷用 hook 已移除，正式 AP 只改 alpha，不修改輸出或資料。
    assert not any(module._forward_hooks for module in model.modules())
    proof = {'status': 'cpu_passed', 'source': str(source), 'sha256': record['bittrue_sha256'],
             'original_alpha': original_alpha, 'effective_alpha': 0.0,
             'changed_state_keys': changed, 'zero_residual_identity_exact': True,
             'cpu160_propagation': propagation, 'diagnostic_only': True,
             'optimizer_steps': 0, 'accuracy_winner': False,
             'relocation_performed': False, 'baseline': baseline['metrics']}
    write_json(output / 'cpu-proof.json', proof)
    metrics = validate(model, 'bittrue', prepare_coco(), output / 'off-validation')
    assert sha256(source) == record['bittrue_sha256']
    write_json(output / 'summary.json', {
        **proof, 'status': 'passed', 'images': 5000, 'off': metrics,
        'delta_off_minus_original': {key: metrics[key] - baseline['metrics'][key] for key in metrics},
        'evaluator': 'Ultralytics internal; imgsz640; halfFalse; batch32',
        'interpretation': '瞬間關閉的反事實診斷；不等同重新訓練的無 MASF 或 Detect-only 模型',
    })
    print('JOB_DONE B100_MASF_OFF', metrics, flush=True)


if __name__ == '__main__':
    main()
