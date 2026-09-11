"""MASF 三組最小位置實驗；完整前置與 smoke 通過才正式訓練。"""
import json
import sys
from common import ROOT, write_json
from run_scope_pair import wait_job
from export_model import export
import torch


def main():
    state_path = ROOT / 'artifacts/masf-p3-queue-v1-state.json'
    if state_path.exists():
        raise FileExistsError('禁止覆寫既有 queue')
    state = {'status': 'preflight', 'jobs': [], 'monitor_wait_seconds': 600,
             'epochs_per_arm': 5, 'variants': ['control', 'shared', 'fork'],
             'accuracy_winner': None, 'goal': '驗證 MASF 能否增準；不是刪除 MASF'}
    write_json(state_path, state)
    def job(name, script, extra=()):
        entry = {'name': name, 'status': 'running'}
        state['jobs'].append(entry)
        write_json(state_path, state)
        command = [sys.executable, str(ROOT / 'scripts/monitor.py'), '--name', name, '--',
                   sys.executable, '-u', str(ROOT / 'scripts' / script), *extra]
        code = wait_job(command)
        if code:
            entry['status'] = state['status'] = 'error'
            write_json(state_path, state)
            return code, None
        summary = json.loads((ROOT / 'artifacts' / name / 'summary.json').read_text())
        entry['status'] = summary['status']
        write_json(state_path, state)
        return 0, summary
    code, summary = job('masf-p3-preflight-v1', 'probe_masf_p3.py')
    if code:
        return code
    assert summary['status'] == 'passed'
    proof = {}
    torch.set_num_threads(4)
    for variant in state['variants']:
        name = f'masf-p3-{variant}-smoke-v1'
        code, summary = job(name, 'train_masf_p3.py', ('--name', name, '--variant', variant, '--smoke'))
        if code:
            return code
        assert summary['status'] == 'passed' and summary['smoke_images'] == 128
        assert summary['optimizer_steps'] == 1 and summary['ema_updates'] == 7401
        exported = export(ROOT / 'artifacts' / name / 'epoch-01-resume.pt',
                          ROOT / 'artifacts' / name / 'smoke-inference.pt')
        proof[variant] = {'trace': summary['first_macro_trace_sha256'],
                          'loss_sum': summary['first_macro_native_loss_sum'],
                          'gradient_norms': summary['first_gradient_norm_by_role'],
                          'first_update_ratio': summary['first_update_ratio'],
                          'alpha_after_update': summary['alpha_after_update'],
                          'export_cpu160_reload_exact': True, 'exported': str(exported)}
    assert len({entry['trace'] for entry in proof.values()}) == 1
    assert len({entry['loss_sum'] for entry in proof.values()}) == 1
    for variant in ('shared', 'fork'):
        assert proof[variant]['gradient_norms']['alpha'] > 0
        assert proof[variant]['alpha_after_update'] != 0
    write_json(ROOT / 'artifacts/masf-p3-smoke-proof-v1.json', {'status': 'passed', 'arms': proof})
    state['status'] = 'training'
    write_json(state_path, state)
    print('MASF_P3_PASSED: 初始等價、P4/P5隔離、梯度與序列化皆通過', flush=True)
    for variant in state['variants']:
        name = f'masf-p3-{variant}-v1'
        code, summary = job(name, 'train_masf_p3.py', ('--name', name, '--variant', variant))
        if code:
            return code
        if summary['status'] == 'paused_for_analysis':
            state['status'] = 'awaiting_analysis'
            write_json(state_path, state)
            print('STAGE_REVIEW: MASF 主要 AP 安全門檻觸發', flush=True)
            return 0
        assert summary['status'] == 'complete' and len(summary['epochs']) == 5
    state['status'] = 'awaiting_analysis'
    write_json(state_path, state)
    print('ALL_DONE: MASF 三組各5回合完成，接續增準與位置比較', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
