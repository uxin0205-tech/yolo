"""CPU 梯度連通診斷；不更新權重，不將合成梯度當成 AP 證據。"""
import copy
import json
from common import ROOT, setup, sha256, write_json
import torch
import masf_p3


def main():
    output = ROOT / 'artifacts/masf-task-gradient-probe-v1'
    output.mkdir(parents=True, exist_ok=False)
    setup()
    torch.set_num_threads(4)
    source = ROOT / 'artifacts/masf-head-fork-v1/epoch-10-resume.pt'
    source_sha = sha256(source)
    snap = torch.load(source, map_location='cpu', weights_only=False)
    head = copy.deepcopy(snap['ema'].float().model[23])
    del snap
    head.train()
    for module in head.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    before = {n: x.clone() for n, x in head.state_dict().items()}
    generator = torch.Generator().manual_seed(20260928)
    channels = [head.cv2[i][0].conv.in_channels for i in range(3)]
    x = [torch.randn(1, c, s, s, generator=generator).requires_grad_()
         for c, s in zip(channels, (20, 10, 5))]
    native = head(x)
    # 新路徑只在 CPU 診斷中建立：先切斷 raw P3，再執行同一個 MASF。
    # BN 統計固定，第二次 context 前向不修改狀態。
    forked = head.p3_masf(x[0].detach())
    proposed = head.forward_head([forked, x[1].detach(), x[2].detach()], **head.one2one)
    assert all(torch.equal(native['one2one'][k], proposed[k]) for k in ('boxes', 'scores'))
    parameters = tuple(head.p3_masf.parameters())

    def gradient(pred):
        objective = pred['boxes'].square().mean() + pred['scores'].square().mean()
        return torch.autograd.grad(objective, parameters + tuple(x), allow_unused=True, retain_graph=True)

    old_one = gradient(native['one2one'])
    old_many = gradient(native['one2many'])
    new_one = gradient(proposed)
    count = len(parameters)
    norm = lambda grads: sum(float(g.square().sum()) for g in grads if g is not None) ** .5
    assert all(g is None for g in old_one)
    assert norm(old_many[:count]) > 0 and norm(new_one[:count]) > 0
    assert all(g is None for g in new_one[count:])
    assert all(torch.equal(v, head.state_dict()[n]) for n, v in before.items())
    assert sha256(source) == source_sha
    pairs = {}
    for variant in ('control', 'fork'):
        report = json.loads((ROOT / f'artifacts/masf-head-{variant}-v1/summary.json').read_text())
        assert report['status'] == 'complete'
        assert [e['epoch'] for e in report['epochs']] == list(range(6, 11))
        assert all(e['images'] == 118287 and e['macros'] == 925 for e in report['epochs'])
        pairs[variant] = report
    assert pairs['control']['first_macro_trace_sha256'] == pairs['fork']['first_macro_trace_sha256']
    comparisons = [{'epoch': a['epoch'], 'control': a['ema'], 'fork': b['ema'],
                    'fork_minus_control': {k: b['ema'][k] - a['ema'][k] for k in a['ema']}}
                   for a, b in zip(pairs['control']['epochs'], pairs['fork']['epochs'])]
    write_json(output / 'summary.json', {
        'status': 'passed', 'source': str(source), 'source_sha256': source_sha,
        'device': 'cpu', 'training_updates': 0, 'synthetic_features_only': True,
        'native_one2one_masf_gradient_absent': True,
        'native_one2many_masf_gradient_norm': norm(old_many[:count]),
        'proposed_one2one_masf_gradient_norm': norm(new_one[:count]),
        'proposed_one2one_raw_backbone_gradient_absent': True,
        'one2one_forward_exact': True, 'all_head_state_unchanged': True,
        'source_file_unchanged': True, 'paired_epochs': comparisons,
        'accuracy_cause_proven': False, 'new_training_recipe_validated': False,
        'interpretation': '原生 detach 是既有設計；僅證明可保留前向並讓 one2one 監督 MASF，不證明 AP 會提高。',
    })
    print('JOB_DONE: E10 配對驗收與 CPU 梯度路徑診斷通過；未更新權重', flush=True)


if __name__ == '__main__':
    main()
