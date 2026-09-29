"""MASF control／fork E5 配對續訓至 E10，只提高 P3 head 學習率。"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
from common import ROOT, SOURCE, setup, prepare_coco, sha256, write_json
import torch
from ultralytics.utils.torch_utils import ModelEMA
from train_masf_p3 import Harness, LEARNING_RATES, PARENT_SHA, training_mode, update_ema
from masf_p3 import get_masf, parameter_role
from continue_a0 import check_fixed, validate

SOURCE_SHA = {
    'control': '432832f3c4d410e97ebd96f3d12a9fdc6a7b421b6a0747ae1f70065462fc2b67',
    'fork': '132c74576946cdaa4ef606dfddaa2b56fddf219a0987b9ddfe12ca051d40c33e',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['control', 'fork'], required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum():
        raise ValueError('name 格式不符')
    source = ROOT / f'artifacts/masf-p3-{args.variant}-v1/epoch-05-resume.pt'
    if sha256(source) != SOURCE_SHA[args.variant]:
        raise ValueError('續訓 E5 來源雜湊不符')
    output = ROOT / 'artifacts' / args.name
    output.mkdir(parents=True, exist_ok=False)
    setup()
    data = prepare_coco()
    torch.set_num_threads(8)
    snap = torch.load(source, map_location='cpu', weights_only=False)
    assert snap['epoch'] == 5 and snap['optimizer_steps'] == 4625 and snap['ema_updates'] == 12025
    assert snap['variant'] == args.variant and snap['parent_sha256'] == PARENT_SHA
    assert snap['model'].criterion.updates == 4
    Harness.variant = args.variant
    trainer = Harness(overrides={
        'model': str(SOURCE / 'weights/bittrue/a0.pt'), 'data': str(data), 'epochs': 10,
        'batch': 32, 'nbs': 128, 'imgsz': 640, 'device': '0', 'workers': 4, 'amp': False,
        'optimizer': 'AdamW', 'project': str(output), 'name': 'setup', 'exist_ok': False,
        'seed': 20260927, 'deterministic': True, 'mosaic': 0.0, 'mixup': 0.0,
        'cutmix': 0.0, 'copy_paste': 0.0, 'fliplr': .5, 'cache': False,
        'fraction': 1.0, 'plots': False, 'save_json': False, 'warmup_epochs': 1.0,
        'save': False, 'patience': 4, 'cos_lr': True, 'close_mosaic': 0,
    })
    trainer._setup_train()
    model = trainer.model
    model.load_state_dict(snap['model'].state_dict(), strict=True)
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert set(active) == {name for name, p in snap['model'].named_parameters() if p.requires_grad}
    optimizer = trainer.optimizer
    for current, saved in zip(optimizer.param_groups, snap['optimizer']['param_groups']):
        assert current['role'] == saved['role'] and len(current['params']) == len(saved['params'])
    optimizer.load_state_dict(snap['optimizer'])
    assert all(int(state['step']) == 4625 for state in optimizer.state.values())
    for group in optimizer.param_groups:
        assert group['initial_lr'] == LEARNING_RATES[group['role']]
        if group['role'] == 'p3_head':
            group['initial_lr'] = 1e-5
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    scaler.load_state_dict(snap['scaler'])
    ema = ModelEMA(model)
    ema.ema.load_state_dict(snap['ema'].state_dict(), strict=True)
    ema.updates = snap['ema_updates']
    # 舊快照在 epoch 結束的 criterion.update() 前保存；E6 正確 age 應是 5。
    model.criterion = model.init_criterion()
    for _ in range(5):
        model.criterion.update()
    expected_criterion = snap['model'].criterion
    expected_criterion.update()
    assert model.criterion.updates == expected_criterion.updates == 5
    assert model.criterion.o2m == expected_criterion.o2m and model.criterion.o2o == expected_criterion.o2o
    for label, actual in (('model', model), ('ema', ema.ema)):
        assert all(torch.equal(v.cpu(), snap[label].state_dict()[n]) for n, v in actual.state_dict().items())
    fixed = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name not in active}
    check_fixed(ema.ema, fixed)
    initial = {name: p.detach().clone() for name, p in active.items()}
    del snap, expected_criterion
    reference = json.loads((ROOT / 'artifacts/a0-scope-late-v1/summary.json').read_text())['epochs'][-1]['ema']
    tracked = ('coco/box/map50_95', 'coco/person/box/map50_95')
    report = {
        'status': 'running', 'variant': args.variant, 'continued_from': str(source),
        'continued_from_sha256': SOURCE_SHA[args.variant], 'parent_sha256': PARENT_SHA,
        'start_epoch': 5, 'target_epoch': 10, 'optimizer_initial_steps': 4625, 'ema_initial_updates': 12025,
        'criterion_initial_updates': 5, 'initial_live_and_ema_state_exact': True,
        'optimizer_and_scaler_restored': True, 'warmup_restarted': False,
        'learning_rates': {**LEARNING_RATES, 'p3_head': 1e-5},
        'change': 'P3 head base LR 2e-6 -> 1e-5; scope, context/alpha LR and topology unchanged',
        'physical_batch': 32, 'logical_batch': 128, 'paired_new_stream_seed': 20260927,
        'exact_uninterrupted_dataloader_replay': False,
        'decision_metrics': list(tracked), 'epochs': [], 'script_sha256': sha256(Path(__file__)),
    }
    total = 4625
    trace = hashlib.sha256()
    parameters = list(active.values())
    with (output / 'progress.jsonl').open('x') as log:
        for epoch in range(5, 6 if args.smoke else 10):
            random.seed(epoch)
            np.random.seed(epoch)
            torch.manual_seed(epoch)
            torch.cuda.manual_seed_all(epoch)
            training_mode(model)
            assert model.criterion.updates == epoch
            pending = []
            seen = macro = 0
            started = time.time()
            for index, batch in enumerate(trainer.train_loader):
                if total == 4625:
                    trace.update(json.dumps(batch['im_file']).encode())
                    for key in ('img', 'cls', 'bboxes', 'batch_idx'):
                        trace.update(batch[key].contiguous().numpy().tobytes())
                pending.append(trainer.preprocess_batch(batch))
                if len(pending) < 4 and index + 1 < len(trainer.train_loader):
                    continue
                steps = math.ceil(len(trainer.train_loader) / 4)
                cosine = .5 + .5 * (1 + math.cos(math.pi * (epoch + macro / steps) / 10)) / 2
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * cosine
                for attempt in range(16):
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for item in pending:
                        with torch.autocast('cuda', dtype=torch.float16):
                            loss, _ = model(item)
                            loss = loss.sum()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('loss 非有限')
                        scaler.scale(loss).backward()
                        losses.append(float(loss.detach()))
                    scaler.unscale_(optimizer)
                    if all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                        break
                    scaler.update(new_scale=scaler.get_scale() / 2)
                else:
                    raise FloatingPointError('AMP／gradient 未恢復')
                masf = get_masf(model)
                if total == 4625:
                    report['first_macro_trace_sha256'] = trace.hexdigest()
                    report['first_macro_native_loss_sum'] = sum(losses)
                    report['first_gradient_norm_by_role'] = {
                        role: sum(float(p.grad.float().square().sum()) for n, p in active.items()
                                  if parameter_role(n) == role) ** .5 for role in LEARNING_RATES}
                    assert report['first_gradient_norm_by_role']['p3_head'] > 0
                    if masf is not None:
                        assert report['first_gradient_norm_by_role']['context'] > 0
                torch.nn.utils.clip_grad_norm_(parameters, 10)
                scaler.step(optimizer)
                scaler.update()
                if masf is not None:
                    with torch.no_grad():
                        masf.alpha.clamp_(-.25, .25)
                update_ema(ema, model, active)
                if total == 4625:
                    report['first_update_ratio_by_role'] = {}
                    for role in LEARNING_RATES:
                        names = [n for n in active if parameter_role(n) == role]
                        if names:
                            delta = sum(float((active[n].detach() - initial[n]).float().square().sum()) for n in names) ** .5
                            norm = sum(float(initial[n].float().square().sum()) for n in names) ** .5
                            report['first_update_ratio_by_role'][role] = delta / max(norm, 1e-12)
                    assert 0 < report['first_update_ratio_by_role']['p3_head'] < .001
                    if masf is not None:
                        assert 0 < report['first_update_ratio_by_role']['context'] < .001
                total += 1
                macro += 1
                seen += sum(item['img'].shape[0] for item in pending)
                pending = []
                log.write(json.dumps({'epoch': epoch + 1, 'macro': macro, 'loss_sum': sum(losses),
                                      'amp_retries': attempt, 'optimizer_steps': total,
                                      'o2m_weight': model.criterion.o2m, 'o2o_weight': model.criterion.o2o}) + '\n')
                log.flush()
                if args.smoke:
                    break
            check_fixed(model, fixed)
            check_fixed(ema.ema, fixed)
            assert all(int(state['step']) == total for state in optimizer.state.values())
            assert ema.updates == 7400 + total
            payload = {'model': copy.deepcopy(model).cpu(), 'ema': copy.deepcopy(ema.ema).cpu(),
                       'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                       'ema_updates': ema.updates, 'epoch': epoch + 1, 'optimizer_steps': total,
                       'variant': args.variant, 'parent_sha256': PARENT_SHA,
                       'continued_from_sha256': SOURCE_SHA[args.variant],
                       'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                       'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
            torch.save(payload, output / f'epoch-{epoch + 1:02d}-resume.pt')
            del payload
            if args.smoke:
                assert total == 4626 and ema.updates == 12026 and model.criterion.updates == 5
                report.update(status='passed', smoke_images=seen, optimizer_steps=total, ema_updates=ema.updates,
                              criterion_updates=model.criterion.updates)
                write_json(output / 'summary.json', report)
                print('JOB_DONE MASF_HEAD_SMOKE', args.variant, flush=True)
                return
            assert seen == 118287 and macro == 925
            metrics = validate(ema.ema, 'bittrue', data, output / f'epoch-{epoch + 1:02d}-ema')
            live = validate(model, 'bittrue', data, output / f'epoch-{epoch + 1:02d}-live')
            report['epochs'].append({'epoch': epoch + 1, 'images': seen, 'macros': macro,
                                     'ema': metrics, 'live': live, 'elapsed_seconds': time.time() - started,
                                     'alpha_live': float(masf.alpha.detach()) if masf is not None else None,
                                     'alpha_ema': float(get_masf(ema.ema).alpha.detach()) if masf is not None else None,
                                     'delta_parent': {key: metrics[key] - reference[key] for key in reference}})
            if any(metrics[key] - reference[key] < -.005 for key in tracked):
                report['status'] = 'paused_for_analysis'
                break
            write_json(output / 'summary.json', report)
            model.criterion.update()
        if report['status'] == 'running':
            report['status'] = 'complete'
        write_json(output / 'summary.json', report)
    print('JOB_DONE MASF_HEAD', report['status'], flush=True)


if __name__ == '__main__':
    main()
