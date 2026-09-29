#!/usr/bin/env python3
"""真實資料各跑一個 native/HOG macro，檢查 retry/EMA；不產生可部署權重。

不是 AP 實驗、不是 batch 測速。HOG μ=0.001 只用來覆蓋輔助反向路徑。
"""

import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from yolo_optimize import runtime
from yolo_optimize.training import _build, _raw_macros, _reseed
import torch


def verify_case(config, data, variant):
    state = _build(config,
        runtime.FINAL_ROOT / 'weights/combined/inference/best_joint.pt',
        runtime.FINAL_ROOT / 'weights/combined/full-resume/best_joint.pt',
        data, 32, variant, 'cuda:0')
    state.training_mode()
    head_before = {}
    if variant == 'heads':
        for name, parameter in state.model.named_parameters():
            is_head = '.detect_head.' in name or '.pose_head.' in name
            if parameter.requires_grad and not is_head:
                raise AssertionError(f'heads scope 越界解凍：{name}')
            if is_head and parameter.requires_grad:
                head_before[name] = parameter.detach().clone()
    _reseed(state, 0)
    fixed_before = {name: state.model.state_dict()[name].clone()
                    for name in state.ema.fixed_state_names}
    counters = {name: int(module.num_batches_tracked)
        for name, module in state.model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.num_batches_tracked is not None}
    aux_before = {name: parameter.detach().clone()
        for name, parameter in state.model.aux.named_parameters()} if state.model.aux is not None else {}
    rep_before = (state.model.base.graph.model[17].conv2.bn.weight.detach().clone()
                  if variant == 'rep17' else None)
    iterator = iter(_raw_macros(state, 32))
    macro = next(iterator)
    state.scheduler.prepare_step()
    state.router.mu = 0.001 if variant == 'hog' else 0.0
    report = state.engine.run(detect_batches=macro.detect_batches,
        pose_batches=macro.pose_batches, record_gradient_statistics=False)
    state.scheduler.advance()
    torch.cuda.synchronize()
    state.guard.assert_unchanged(state.model.base)
    live, averaged = state.model.state_dict(), state.ema.ema.state_dict()
    for name, original in fixed_before.items():
        if not torch.equal(live[name], original) or not torch.equal(averaged[name], original):
            raise AssertionError(f'固定 state 出現漂移：{name}')
    deltas = {}
    for name, module in state.model.named_modules():
        if name not in counters:
            continue
        delta = int(module.num_batches_tracked) - counters[name]
        expected = (len(macro.detect_batches) if 'detect_head.' in name else
                    len(macro.pose_batches) if 'pose_head.' in name else 0)
        if delta != expected:
            raise AssertionError(f'BN 重試累計不正確：{name} delta={delta} expected={expected}')
        deltas[name] = delta
    steps = {int(item['step']) for item in state.optimizer.state.values() if 'step' in item}
    if variant == 'heads':
        for role in ('detect_head', 'pose_head'):
            if not any(not torch.equal(original, dict(state.model.named_parameters())[name])
                       for name, original in head_before.items() if f'.{role}.' in name):
                raise AssertionError(f'{role} 沒有實際更新')
    if steps != {1} or state.ema.updates != 1:
        raise AssertionError(f'一個 macro 只能更新一次：optimizer={steps}, EMA={state.ema.updates}')
    aux_update = None
    if state.model.aux is not None:
        # Native engine 在成功 step 後 zero_grad；用實際更新與 AdamW 梯度動量確認反向。
        aux_update = sum(float((parameter.detach() - aux_before[name]).square().sum())
            for name, parameter in state.model.aux.named_parameters())
        moments = [state.optimizer.state[p].get('exp_avg') for p in state.model.aux.parameters()]
        if (not torch.isfinite(torch.tensor(aux_update)) or aux_update <= 0
                or any(m is None or not bool(torch.isfinite(m).all()) for m in moments)
                or not any(bool(m.ne(0).any()) for m in moments)):
            raise AssertionError('HOG 輔助參數未通過有限非零更新與梯度動量檢查')
    rep_updated = None
    if rep_before is not None:
        rep_updated = not torch.equal(rep_before, state.model.base.graph.model[17].conv2.bn.weight)
        if not rep_updated:
            raise AssertionError('RepConv 新分支沒有實際參數更新')
    return {'variant': variant, 'status': 'passed', 'report': asdict(report),
        'repconv_branch_updated': rep_updated,
        'optimizer_steps': sorted(steps), 'ema_updates': state.ema.updates,
        'fixed_state_count': len(fixed_before), 'fixed_state_bit_exact': True,
        'bn_counter_deltas': deltas, 'auxiliary_parameter_update_squared': aux_update,
        'mu_for_smoke_only': state.router.mu, 'physical_detect_batch': 32,
        'training_safety': {'amp': state.metadata['amp'], 'ema': state.metadata['ema']},
        'weights_saved': False, 'updates_discarded': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', nargs='+', choices=('native', 'hog', 'rep17', 'masf_off', 'heads'), default=['native', 'hog'])
    args = parser.parse_args()
    root = runtime.output_path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cases = []
    try:
        config = runtime.load_config(root.parent)
        data = runtime.prepare_data(config, root.parent / 'datasets')
        for variant in args.variant:
            print(f'[安全短驗證] 開始 {variant}，只跑 1 個 macro。', flush=True)
            result = verify_case(config, data, variant)
            cases.append(result)
            runtime.write_json(root / f'{variant}.json', result)
            print(f'[安全短驗證] {variant} 通過；AMP retries={result["report"]["amp_overflow_retries"]}。', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        runtime.write_json(root / 'summary.json', {'status': 'passed', 'cases': cases,
            'elapsed_seconds': time.monotonic() - started, 'accuracy_validated': False,
            'formal_training_started': False, 'updates_discarded': True})
    except BaseException as error:
        runtime.write_json(root / 'summary.json', {'status': 'failed', 'cases': cases,
            'error': repr(error), 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
