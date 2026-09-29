"""同一已驗證 no-MASF E8 起點，完整 COCO 的 control/P2 配對五回合。"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
from common import ROOT, SOURCE, setup, prepare_coco, write_json, sha256
import torch
from ultralytics.utils.torch_utils import ModelEMA
from continue_a0 import ContinuationHarness, check_fixed, validate
from train_rep17 import training_mode, update_ema
from masf_p2 import prepare, get_masf, role, RATES, PARENT, PARENT_SHA, EMA_AGE


class Harness(ContinuationHarness):
    variant = 'control'

    def get_model(self, cfg=None, weights=None, verbose=True):
        return prepare(self.variant)

    def build_optimizer(self, model, *args, **kwargs):
        groups = {}
        for name, p in model.named_parameters():
            category = role(name)
            p.requires_grad_(category != 'fixed')
            if category != 'fixed':
                groups.setdefault((category, p.ndim >= 2 and not name.endswith('.bias')), []).append(p)
        return torch.optim.AdamW([
            {'params': params, 'lr': RATES[category], 'initial_lr': RATES[category],
             'role': category, 'weight_decay': .00027 if decay else 0.}
            for (category, decay), params in groups.items()
        ], betas=(.948, .999), eps=1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['control', 'p2'], required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert args.name.replace('-', '').isalnum()
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
        'seed': 20260930, 'deterministic': True, 'mosaic': 0., 'mixup': 0.,
        'cutmix': 0., 'copy_paste': 0., 'fliplr': .5, 'cache': False, 'fraction': 1.,
        'plots': False, 'save_json': False, 'warmup_epochs': 1., 'save': False,
        'patience': 4, 'cos_lr': True, 'close_mosaic': 0,
    })
    trainer._setup_train()
    model = trainer.model
    model.criterion = model.init_criterion()
    training_mode(model)
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fixed = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n not in active}
    initial = {n: p.detach().clone() for n, p in active.items()}
    parameters = list(active.values())
    optimizer = trainer.optimizer
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    ema = ModelEMA(model)
    ema.updates = EMA_AGE
    verification = json.loads((ROOT / 'artifacts/direction1-candidate-verification-v1/summary.json').read_text())
    reference = next(c['metrics'] for c in verification['candidates'] if c['name'] == 'control-e8')
    tracked = ('coco/box/map50_95', 'coco/person/box/map50_95')
    report = {'status': 'running', 'variant': args.variant, 'parent': str(PARENT),
              'parent_sha256': PARENT_SHA, 'parent_state': 'verified no-MASF control E8 EMA',
              'scope': 'all original Detect heads plus P2 MASF; other parameters/all BN statistics fixed',
              'head_inputs': [16, 19, 22], 'head_strides': [8, 16, 32], 'new_detect_head': False,
              'learning_rates': RATES, 'optimizer': 'AdamW', 'fresh_optimizer': True,
              'warmup_epochs': 1, 'criterion_horizon': 10, 'epochs_budget': 5,
              'physical_batch': 32, 'logical_batch': 128, 'ema_initial_updates': EMA_AGE,
              'context_initialization': 'fresh seed20260930; no P3 context transfer',
              'alpha_initial': 0., 'alpha_bound': [-.25, .25], 'epochs': [],
              'decision_metrics': list(tracked), 'reference': reference,
              'script_sha256': sha256(Path(__file__)), 'module_sha256': sha256(Path(__file__).with_name('masf_p2.py'))}
    total = 0
    trace = hashlib.sha256()
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
                cosine = .5 + .25 * (1 + math.cos(math.pi * (epoch + macro / steps) / 10))
                warm = min(1., .1 + .9 * (total + 1) / steps)
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * cosine * warm
                for attempt in range(16):
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for item in pending:
                        with torch.autocast('cuda', dtype=torch.float16):
                            loss = model(item)[0].sum()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('loss 非有限')
                        scaler.scale(loss).backward()
                        losses.append(float(loss.detach()))
                    scaler.unscale_(optimizer)
                    if all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                        break
                    scaler.update(new_scale=scaler.get_scale() / 2)
                else:
                    raise FloatingPointError('AMP gradient 未恢復')
                masf = get_masf(model)
                if total == 0:
                    norms = {category: sum(float(p.grad.float().square().sum()) for n, p in active.items()
                                           if role(n) == category) ** .5 for category in RATES}
                    report.update(first_macro_trace_sha256=trace.hexdigest(), first_macro_loss=sum(losses),
                                  first_gradient_norms=norms)
                    assert norms['head'] > 0
                    if masf is not None:
                        assert norms['alpha'] > 0 and norms['context'] == 0
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
            torch.save({'model': copy.deepcopy(model).cpu(), 'ema': copy.deepcopy(ema.ema).cpu(),
                        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                        'epoch': epoch + 1, 'optimizer_steps': total, 'ema_updates': ema.updates,
                        'variant': args.variant, 'parent_sha256': PARENT_SHA,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()},
                       output / f'epoch-{epoch + 1:02d}-resume.pt')
            if args.smoke:
                assert seen == 128 and total == 1 and ema.updates == EMA_AGE + 1
                report.update(status='passed', smoke_images=seen, optimizer_steps=total, ema_updates=ema.updates)
                write_json(output / 'summary.json', report)
                return
            assert seen == 118287 and macro == 925
            metrics = validate(ema.ema, 'bittrue', data, output / f'epoch-{epoch + 1:02d}-ema')
            live = validate(model, 'bittrue', data, output / f'epoch-{epoch + 1:02d}-live')
            report['epochs'].append({'epoch': epoch + 1, 'images': seen, 'macros': macro,
                'ema': metrics, 'live': live, 'elapsed_seconds': time.time() - started,
                'alpha_ema': float(get_masf(ema.ema).alpha) if masf is not None else None})
            if any(metrics[k] - reference[k] < -.005 for k in tracked):
                report['status'] = 'paused_for_analysis'
                break
            write_json(output / 'summary.json', report)
            model.criterion.update()
    if report['status'] == 'running':
        report['status'] = 'complete'
    write_json(output / 'summary.json', report)


if __name__ == '__main__':
    main()
