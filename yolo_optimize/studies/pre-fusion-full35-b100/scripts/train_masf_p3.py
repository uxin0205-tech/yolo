"""同一 late E8 起點：原生 P3 head／shared MASF／Detect-only MASF。"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
from common import ROOT, SOURCE, setup, prepare_coco, write_json, sha256, registry
import torch
from ultralytics.utils.torch_utils import ModelEMA
from train_rep17 import prepare_model as prepare_parent, PARENT, PARENT_SHA, training_mode, update_ema
from continue_a0 import ContinuationHarness, check_fixed, validate
from masf_p3 import attach, get_masf, parameter_role

LEARNING_RATES = {'p3_head': 2e-6, 'context': 1e-5, 'alpha': 1e-4}


def prepare_model(variant):
    return attach(prepare_parent('control'), variant)


class Harness(ContinuationHarness):
    variant = 'control'

    def get_model(self, cfg=None, weights=None, verbose=True):
        return prepare_model(self.variant)

    def build_optimizer(self, model, *args, **kwargs):
        groups = {}
        for name, parameter in model.named_parameters():
            role = parameter_role(name)
            parameter.requires_grad_(role != 'fixed')
            if role != 'fixed':
                decay = parameter.ndim >= 2 and not name.endswith('.bias')
                groups.setdefault((role, decay), []).append(parameter)
        return torch.optim.AdamW([
            {'params': parameters, 'lr': LEARNING_RATES[role],
             'initial_lr': LEARNING_RATES[role], 'role': role,
             'weight_decay': .00027 if decay else 0.0}
            for (role, decay), parameters in groups.items()
        ], betas=(.948, .999), eps=1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--variant', choices=['control', 'shared', 'fork'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum():
        raise ValueError('name 格式不符')
    output = ROOT / 'artifacts' / args.name
    output.mkdir(parents=True, exist_ok=False)
    setup()
    data = prepare_coco()
    torch.set_num_threads(8)
    Harness.variant = args.variant
    trainer = Harness(overrides={
        'model': str(SOURCE / 'weights/bittrue/a0.pt'), 'data': str(data), 'epochs': 10,
        'batch': 32, 'nbs': 128, 'imgsz': 640, 'device': '0', 'workers': 4, 'amp': False,
        'optimizer': 'AdamW', 'project': str(output), 'name': 'setup', 'exist_ok': False,
        'seed': 20260922, 'deterministic': True, 'mosaic': 0.0, 'mixup': 0.0,
        'cutmix': 0.0, 'copy_paste': 0.0, 'fliplr': .5, 'cache': False,
        'fraction': 1.0, 'plots': False, 'save_json': False, 'warmup_epochs': 1.0,
        'save': False, 'patience': 4, 'cos_lr': True, 'close_mosaic': 0,
    })
    trainer._setup_train()
    model = trainer.model
    model.criterion = model.init_criterion()
    training_mode(model)
    assert len(trainer.train_loader.dataset) == 118287
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    fixed = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name not in active}
    initial = {name: p.detach().clone() for name, p in active.items()}
    optimizer = trainer.optimizer
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    ema = ModelEMA(model)
    ema.updates = 7400
    reference = json.loads((ROOT / 'artifacts/a0-scope-late-v1/summary.json').read_text())['epochs'][-1]['ema']
    tracked = ('coco/box/map50_95', 'coco/person/box/map50_95')
    report = {
        'status': 'running', 'variant': args.variant, 'parent': str(PARENT), 'parent_sha256': PARENT_SHA,
        'parent_state': 'late E8 EMA exploratory parent; not promoted winner',
        'masf_context_initialization': 'B100 trained context only; alpha reset to zero' if args.variant != 'control' else None,
        'masf_source_sha256': registry()['bittrue_sha256'] if args.variant != 'control' else None,
        'scope': 'P3 head branches only, plus MASF when present; all other parameters and all BN statistics fixed',
        'learning_rates': LEARNING_RATES, 'optimizer': 'AdamW', 'fresh_optimizer': True,
        'warmup_epochs': 1, 'criterion_horizon': 10, 'epochs_budget': 5,
        'physical_batch': 32, 'logical_batch': 128, 'ema_initial_updates': 7400,
        'ema_policy': 'native decay on active parameters only; fixed state exact',
        'alpha_initial': 0.0, 'alpha_training_bound': [-.25, .25],
        'native_one2one_detach_preserved': True,
        'trainable_parameters_by_role': {role: sum(p.numel() for n, p in active.items() if parameter_role(n) == role)
                                         for role in LEARNING_RATES},
        'decision_metrics': list(tracked), 'epochs': [], 'script_sha256': sha256(Path(__file__)),
        'topology_sha256': sha256(Path(__file__).with_name('masf_p3.py')),
    }
    trace = hashlib.sha256()
    total = 0
    parameters = list(active.values())
    with (output / 'progress.jsonl').open('x') as log:
        for epoch in range(1 if args.smoke else 5):
            random.seed(epoch)
            np.random.seed(epoch)
            torch.manual_seed(epoch)
            torch.cuda.manual_seed_all(epoch)
            training_mode(model)
            pending = []
            seen = macro = 0
            started = time.time()
            for index, batch in enumerate(trainer.train_loader):
                if total == 0:
                    trace.update(json.dumps(batch['im_file']).encode())
                    for key in ('img', 'cls', 'bboxes', 'batch_idx'):
                        trace.update(batch[key].contiguous().numpy().tobytes())
                pending.append(trainer.preprocess_batch(batch))
                if len(pending) < 4 and index + 1 < len(trainer.train_loader):
                    continue
                steps = math.ceil(len(trainer.train_loader) / 4)
                cosine = .5 + .5 * (1 + math.cos(math.pi * (epoch + macro / steps) / 10)) / 2
                warm = min(1.0, .1 + .9 * (total + 1) / steps)
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * cosine * warm
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
                if total == 0:
                    report['first_macro_trace_sha256'] = trace.hexdigest()
                    report['first_macro_native_loss_sum'] = sum(losses)
                    report['first_gradient_norm_by_role'] = {
                        role: sum(float(p.grad.float().square().sum()) for n, p in active.items()
                                  if parameter_role(n) == role) ** .5 for role in LEARNING_RATES}
                    if masf is not None:
                        assert report['first_gradient_norm_by_role']['alpha'] > 0
                        assert report['first_gradient_norm_by_role']['context'] == 0
                torch.nn.utils.clip_grad_norm_(parameters, 10)
                scaler.step(optimizer)
                scaler.update()
                if masf is not None:
                    with torch.no_grad():
                        masf.alpha.clamp_(-.25, .25)
                update_ema(ema, model, active)
                if total == 0:
                    delta = sum(float((p.detach() - initial[n]).float().square().sum()) for n, p in active.items()) ** .5
                    norm = sum(float(p.float().square().sum()) for p in initial.values()) ** .5
                    report['first_update_ratio'] = delta / max(norm, 1e-12)
                    assert 0 < report['first_update_ratio'] < .001
                total += 1
                macro += 1
                seen += sum(item['img'].shape[0] for item in pending)
                pending = []
                log.write(json.dumps({'epoch': epoch + 1, 'macro': macro, 'loss_sum': sum(losses),
                                      'amp_retries': attempt, 'optimizer_steps': total}) + '\n')
                log.flush()
                if args.smoke:
                    break
            check_fixed(model, fixed)
            check_fixed(ema.ema, fixed)
            payload = {'model': copy.deepcopy(model).cpu(), 'ema': copy.deepcopy(ema.ema).cpu(),
                       'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                       'ema_updates': ema.updates, 'epoch': epoch + 1, 'optimizer_steps': total,
                       'variant': args.variant, 'parent_sha256': PARENT_SHA,
                       'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                       'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
            torch.save(payload, output / f'epoch-{epoch + 1:02d}-resume.pt')
            del payload
            if args.smoke:
                report.update(status='passed', smoke_images=seen, optimizer_steps=total, ema_updates=ema.updates,
                              alpha_after_update=float(masf.alpha.detach()) if masf is not None else None)
                write_json(output / 'summary.json', report)
                print('JOB_DONE MASF_SMOKE', args.variant, flush=True)
                return
            if seen != 118287:
                raise ValueError('完整訓練數量不符')
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
            if hasattr(model.criterion, 'update'):
                model.criterion.update()
        if report['status'] == 'running':
            report['status'] = 'complete'
        write_json(output / 'summary.json', report)
    print('JOB_DONE MASF_P3', report['status'], flush=True)


if __name__ == '__main__':
    main()
