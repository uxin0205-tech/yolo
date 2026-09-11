"""校準通過→單次真實更新→匯出→正式 E6–E10；正常最多等待 600 秒。"""
import copy
import json
import sys
from common import ROOT, write_json, setup
import torch
from masf_p3 import P3MASFDetect
from masf_task_bridge import TaskAlignedMASFDetect
from run_scope_pair import wait_job
from export_model import export
from verify_qk_challenger import tensors


def main():
    setup()
    torch.set_num_threads(4)
    state_path = ROOT / 'artifacts/masf-task-queue-v1-state.json'
    if state_path.exists():
        raise FileExistsError(state_path)
    state = {'status': 'smoke', 'monitor_wait_seconds': 600, 'jobs': [], 'accuracy_winner': False}
    write_json(state_path, state)
    control = json.loads((ROOT / 'artifacts/masf-head-fork-v1/summary.json').read_text())
    for smoke in (True, False):
        name = 'masf-task-bridge-smoke-v1' if smoke else 'masf-task-bridge-v1'
        job = {'name': name, 'status': 'running'}
        state['jobs'].append(job)
        write_json(state_path, state)
        command = [sys.executable, str(ROOT / 'scripts/monitor.py'), '--name', name, '--',
                   sys.executable, '-u', str(ROOT / 'scripts/train_masf_task_bridge.py'),
                   '--variant', 'fork', '--name', name]
        if smoke:
            command.append('--smoke')
        code = wait_job(command)
        if code:
            job['status'] = state['status'] = 'error'
            write_json(state_path, state)
            return code
        output = ROOT / 'artifacts' / name
        report = json.loads((output / 'summary.json').read_text())
        assert report['first_macro_trace_sha256'] == control['first_macro_trace_sha256']
        assert abs(report['first_macro_native_loss_sum'] - control['first_macro_native_loss_sum']) < 1e-5
        job['status'] = report['status']
        if smoke:
            assert report['status'] == 'passed' and report['smoke_images'] == 128
            snapshot = output / 'epoch-06-resume.pt'
            saved = torch.load(snapshot, map_location='cpu', weights_only=False)
            head = saved['model'].model[23]
            assert isinstance(head, TaskAlignedMASFDetect)
            model = saved['ema'].float().eval()
            native = copy.deepcopy(model)
            native.model[23].__class__ = P3MASFDetect
            image = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(20260928))
            with torch.inference_mode():
                a, b = tensors(model(image)), tensors(native(image))
            assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
            destination = export(snapshot, output / 'smoke-inference.pt')
            del saved, model, native
            state.update(status='training', smoke_forward_loss_exact=True,
                         smoke_native_inference_exact=True, smoke_export=str(destination))
            print('MASF_TASK_PASSED: 真實更新、配對資料、原生 loss、推論與匯出通過', flush=True)
        else:
            state['status'] = 'awaiting_analysis'
            if report['status'] == 'complete':
                assert [r['epoch'] for r in report['epochs']] == list(range(6, 11))
                comparisons = []
                for a, b in zip(control['epochs'], report['epochs']):
                    assert a['epoch'] == b['epoch'] and b['images'] == 118287 and b['macros'] == 925
                    comparisons.append({'epoch': b['epoch'], 'native': a['ema'], 'bridge': b['ema'],
                                        'delta': {k: b['ema'][k] - a['ema'][k] for k in a['ema']}})
                state['comparisons'] = comparisons
            print('ALL_DONE: MASF 梯度橋接階段結束，等待完整 AP 分析；未自動升格', flush=True)
        write_json(state_path, state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
