"""最後 MASF P2 配對 queue：不新增 head，正常只作最多600秒等待。"""
import json
import sys
from common import ROOT, setup, write_json
import torch
from run_scope_pair import wait_job
from export_model import export
from pwl_contract import verify_pwl


def main():
    setup()
    torch.set_num_threads(4)
    path = ROOT / 'artifacts/masf-p2-queue-v1-state.json'
    if path.exists():
        raise FileExistsError(path)
    state = {'status': 'preflight', 'jobs': [], 'monitor_wait_seconds': 600,
             'last_masf_direction': True, 'failure_next_phase': 'combine_without_masf',
             'acceptance': 'same-epoch overall/person nonnegative; at least one +0.001; no parent regression',
             'new_detect_head': False, 'pwl_range': [-10, 0]}
    write_json(path, state)
    jobs = [('masf-p2-preflight-v1', 'probe_masf_p2.py', [])]
    for smoke in (True, False):
        for variant in ('control', 'p2'):
            name = f'masf-p2-{variant}-' + ('smoke-v1' if smoke else 'v1')
            arguments = ['--variant', variant, '--name', name] + (['--smoke'] if smoke else [])
            jobs.append((name, 'train_masf_p2.py', arguments))
    proof = {}
    for name, script, arguments in jobs:
        job = {'name': name, 'status': 'running'}
        state['jobs'].append(job)
        write_json(path, state)
        command = [sys.executable, str(ROOT / 'scripts/monitor.py'), '--name', name, '--',
                   sys.executable, '-u', str(ROOT / 'scripts' / script), *arguments]
        code = wait_job(command)
        if code:
            job['status'] = state['status'] = 'error'
            write_json(path, state)
            return code
        output = ROOT / 'artifacts' / name
        report = json.loads((output / 'summary.json').read_text())
        job['status'] = report['status']
        if 'smoke' in name:
            assert report['status'] == 'passed' and report['smoke_images'] == 128
            snapshot = output / 'epoch-01-resume.pt'
            destination = export(snapshot, output / 'smoke-inference.pt')
            verify_pwl(torch.load(destination, map_location='cpu', weights_only=False)['model'])
            proof[report['variant']] = {'trace': report['first_macro_trace_sha256'],
                                       'loss': report['first_macro_loss'], 'norms': report['first_gradient_norms']}
            if len(proof) == 2:
                assert proof['control']['trace'] == proof['p2']['trace']
                assert abs(proof['control']['loss'] - proof['p2']['loss']) < 1e-5
                assert abs(proof['control']['norms']['head'] - proof['p2']['norms']['head']) < 1e-5
                state.update(status='training', smoke_proof=proof)
                print('P2_PASSED: 同資料、首 loss、head 梯度、匯出與 [-10,0] PWL 通過', flush=True)
        if report['status'] == 'paused_for_analysis':
            state['status'] = 'awaiting_safety_review'
            write_json(path, state)
            print('STAGE_REVIEW: P2 配對工作觸發精度安全線', flush=True)
            return 0
        write_json(path, state)
    control = json.loads((ROOT / 'artifacts/masf-p2-control-v1/summary.json').read_text())
    candidate = json.loads((ROOT / 'artifacts/masf-p2-p2-v1/summary.json').read_text())
    assert control['first_macro_trace_sha256'] == candidate['first_macro_trace_sha256']
    keys = ('coco/box/map50_95', 'coco/person/box/map50_95')
    comparisons = []
    for a, b in zip(control['epochs'], candidate['epochs']):
        assert a['epoch'] == b['epoch'] and a['images'] == b['images'] == 118287
        delta = {k: b['ema'][k] - a['ema'][k] for k in a['ema']}
        eligible = all(delta[k] >= 0 for k in keys) and max(delta[k] for k in keys) >= .001
        eligible = eligible and all(b['ema'][k] >= candidate['reference'][k] for k in keys)
        comparisons.append({'epoch': b['epoch'], 'delta': delta, 'eligible': eligible})
    assert len(comparisons) == 5
    state.update(status='awaiting_analysis', comparisons=comparisons,
                 threshold_passed=any(c['eligible'] for c in comparisons),
                 next_phase_if_failed='combine_without_masf', automatic_training_started=False)
    write_json(path, state)
    print('ALL_DONE: 最後 P2 MASF 配對完成；依门檻分析，失敗則放棄 MASF 接融合', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
