"""兩個 E5 續訓 smoke 通過後，自動接續共同 E6–E10。"""
import json
import sys
from common import ROOT, write_json
from run_scope_pair import wait_job
from export_model import export
import torch


def main():
    state_path = ROOT / 'artifacts/masf-head-queue-v1-state.json'
    if state_path.exists():
        raise FileExistsError('禁止覆寫 queue')
    state = {'status': 'smoke', 'jobs': [], 'monitor_wait_seconds': 600,
             'start_epoch': 5, 'target_epoch': 10, 'head_lr_multiplier': 5, 'accuracy_winner': None}
    write_json(state_path, state)
    proof = {}
    torch.set_num_threads(4)
    for smoke in (True, False):
        for variant in ('control', 'fork'):
            name = f'masf-head-{variant}-' + ('smoke-v1' if smoke else 'v1')
            entry = {'name': name, 'status': 'running'}
            state['jobs'].append(entry)
            write_json(state_path, state)
            command = [sys.executable, str(ROOT / 'scripts/monitor.py'), '--name', name, '--',
                       sys.executable, '-u', str(ROOT / 'scripts/continue_masf_head.py'),
                       '--variant', variant, '--name', name]
            if smoke:
                command.append('--smoke')
            code = wait_job(command)
            if code:
                entry['status'] = state['status'] = 'error'
                write_json(state_path, state)
                return code
            output = ROOT / 'artifacts' / name
            summary = json.loads((output / 'summary.json').read_text())
            entry['status'] = summary['status']
            write_json(state_path, state)
            if smoke:
                assert summary['status'] == 'passed' and summary['smoke_images'] == 128
                assert summary['optimizer_steps'] == 4626 and summary['ema_updates'] == 12026
                assert summary['criterion_updates'] == 5 and not summary['warmup_restarted']
                export(output / 'epoch-06-resume.pt', output / 'smoke-inference.pt')
                proof[variant] = {'trace': summary['first_macro_trace_sha256'],
                                  'optimizer_steps': 4626, 'ema_updates': 12026, 'criterion_updates': 5,
                                  'update_ratios': summary['first_update_ratio_by_role'],
                                  'gradient_norms': summary['first_gradient_norm_by_role'],
                                  'export_cpu160_reload_exact': True}
            elif summary['status'] == 'paused_for_analysis':
                state['status'] = 'awaiting_analysis'
                write_json(state_path, state)
                print('STAGE_REVIEW: MASF head 主要 AP 安全門檻觸發', flush=True)
                return 0
            else:
                assert summary['status'] == 'complete' and summary['epochs'][-1]['epoch'] == 10
        if smoke:
            assert proof['control']['trace'] == proof['fork']['trace']
            previous = json.loads((ROOT / 'artifacts/masf-p3-smoke-proof-v1.json').read_text())
            assert proof['control']['trace'] != previous['arms']['control']['trace']
            write_json(ROOT / 'artifacts/masf-head-smoke-proof-v1.json', {'status': 'passed', 'arms': proof})
            state['status'] = 'training'
            write_json(state_path, state)
            print('MASF_HEAD_PASSED: 年齡、恢復狀態、更新幅度與配對資料通過', flush=True)
    state['status'] = 'awaiting_analysis'
    write_json(state_path, state)
    print('ALL_DONE: MASF control/fork E10 完成，接續 AP 與推論分析', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
