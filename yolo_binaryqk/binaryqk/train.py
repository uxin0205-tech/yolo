"""Optimize 上游初篩→恢復；精確二值前向、AMP retry、完整 epoch 續訓。"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from . import checkpoint
from .basis import BASES
from .calibration import read_calibration
from .data import dataset
from .evaluate import evaluate
from .model import (
    PARENT,
    PARENT_SHA,
    ROOT,
    SITES,
    bn_buffers,
    convert,
    frozen_hashes,
    load_parent,
    sha256,
    train_scope,
)
from .protocol import learning_rate, recipe_for, should_stop, verify_reference_review
from .run import GPUInUseError, loader, require_idle_gpu, seed, transfer, write_json


class FrozenEMA:
    """平均可訓練權重及 BN statistics；固定係數和凍結權重逐位不變。"""

    def __init__(self, model, mutable_buffers=None):
        self.ema = copy.deepcopy(model).eval()
        mutable = bn_buffers(model) if mutable_buffers is None else set(mutable_buffers)
        self.names = {name for name, p in model.named_parameters() if p.requires_grad} | mutable
        self.updates = 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = .9999 * (1 - math.exp(-self.updates / 2000))
        source = model.state_dict()
        for name, value in self.ema.state_dict().items():
            if name in self.names:
                if value.is_floating_point():
                    value.mul_(decay).add_(source[name], alpha=1 - decay)
                else:
                    value.copy_(source[name])


def optimizer_for(model, recipe):
    groups = {True: [], False: []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[parameter.ndim >= 2 and not name.endswith('.bias')].append(parameter)
    return torch.optim.AdamW([
        {'params': parameters, 'weight_decay': recipe['weight_decay'] if decay else 0.}
        for decay, parameters in groups.items() if parameters
    ], lr=recipe['lr0'], betas=tuple(recipe['betas']))


def atomic_torch(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_resume(path, model, ema, optimizer, manifest, next_epoch, best, history, *, recipe, scaler):
    payload = {'schema': 'binaryqk-resume-v2', 'recipe': recipe, 'manifest': manifest,
               'live_state': state_cpu(model), 'ema_state': state_cpu(ema.ema),
               'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
               'ema_updates': ema.updates, 'next_epoch': next_epoch, 'best': best, 'history': history}
    atomic_torch(path, payload)
    write_json(path.with_suffix('.json'), {'sha256': sha256(path), 'next_epoch': next_epoch})


def restore_resume(path, model, ema, optimizer, manifest, *, recipe, scaler):
    if sha256(path) != json.loads(path.with_suffix('.json').read_text())['sha256']:
        raise ValueError('resume checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != 'binaryqk-resume-v2' or payload['recipe'] != recipe or payload['manifest'] != manifest:
        raise ValueError('resume recipe/manifest mismatch')
    model.load_state_dict(payload['live_state'], strict=True)
    ema.ema.load_state_dict(payload['ema_state'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scaler.load_state_dict(payload['scaler'])
    ema.updates = payload['ema_updates']
    return payload['next_epoch'], payload['best'], payload['history']


def save_best(output, model, manifest, epoch, codec=checkpoint):
    path = output / f'epoch-{epoch:02d}.pt'
    codec.save(path, model, manifest)
    restored, _ = codec.load(path)
    if any(not torch.equal(v.cpu(), restored.state_dict()[name]) for name, v in model.state_dict().items()):
        raise AssertionError('best checkpoint not identical after reload')
    return path.name


def macro_step(model, pending, optimizer, scaler, parameters, recipe, gradient_names=None, loss_fn=None):
    """Overflow 僅縮小 loss scale；重算同一 macro，還原 BN，不能跳過資料。"""
    buffers = dict(model.named_buffers())
    saved = {name: buffers[name].detach().clone() for name in bn_buffers(model)}
    loss_factor = recipe.get('reference_batch', 0) / sum(len(item['img']) for item in pending) if recipe.get(
        'reference_batch') else 1.
    for attempt in range(recipe['overflow_attempts']):
        optimizer.zero_grad(set_to_none=True)
        if attempt:
            with torch.no_grad():
                for name, value in saved.items():
                    buffers[name].copy_(value)
        losses = []
        for item in pending:
            with torch.autocast('cuda', dtype=torch.float16, enabled=recipe['amp']):
                loss, _ = model.loss(item) if loss_fn is None else loss_fn(model, item)
                loss = loss.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss; no optimizer update')
            scaler.scale(loss * loss_factor).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        if finite:
            break
        if not scaler.is_enabled():
            raise FloatingPointError('nonfinite FP32 gradient')
        scaler.update(new_scale=scaler.get_scale() / 2)
    else:
        raise FloatingPointError('AMP overflow retry limit reached')
    names = gradient_names or [f'{site}.qkv.{part}.conv.weight' for site in SITES for part in ('q', 'k')]
    gradients = {name: float(model.get_parameter(name).grad.float().norm())
                 if model.get_parameter(name).grad is not None else 0. for name in names}
    if any(value <= 0 or not math.isfinite(value) for value in gradients.values()):
        raise AssertionError('missing/zero/nonfinite required gradient')
    qk_grad = {name: value for name, value in gradients.items() if '.qkv.' in name}
    norm = torch.nn.utils.clip_grad_norm_(parameters, recipe['gradient_clip'], error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return {'loss_macro_sum': sum(losses), 'gradient_norm': float(norm),
            'qk_gradient_norm': qk_grad, 'required_gradient_norm': gradients,
            'loss_backward_factor': loss_factor, 'amp_retries': attempt, 'loss_scale': scaler.get_scale()}


def build_model(variant, phase, parent, calibration):
    if phase == 'screening':
        if parent is not None:
            raise ValueError('screening requires the common native parent')
        model = convert(load_parent(), variant, calibration['coefficients'][variant])
        source, digest = PARENT, PARENT_SHA
    else:
        if parent is None:
            raise ValueError('recovery requires the selected screening checkpoint')
        model, manifest = checkpoint.load(parent)
        if manifest['variant'] != variant or manifest['lineage']['phase'] != 'screening':
            raise ValueError('recovery parent must be this variant screening result')
        if manifest['coefficients'] != calibration['coefficients'][variant]:
            raise ValueError('recovery cannot change calibrated score coefficients')
        source, digest = Path(parent).resolve(), sha256(parent)
    model.research_lineage = {'parent': str(source), 'parent_sha256': digest, 'phase': phase}
    return model


class Stage1Policy:
    codec = checkpoint
    verify = staticmethod(verify_reference_review)
    recipe = staticmethod(recipe_for)
    build = staticmethod(build_model)
    configure = staticmethod(train_scope)
    mutable = staticmethod(bn_buffers)
    optimizer = staticmethod(optimizer_for)

    @staticmethod
    def loss(model, batch):
        return model.loss(batch)

    @staticmethod
    def accept_candidate(report, history):
        return True

    @staticmethod
    def gradient_names(model):
        return [f'{site}.qkv.{part}.conv.weight' for site in SITES for part in ('q', 'k')]

    @staticmethod
    def set_lr(optimizer, recipe, epoch, position, step, smoke):
        lr = recipe['lr0'] if smoke else learning_rate(recipe, epoch, position)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr

    @staticmethod
    def guard_metrics(report, history, smoke):
        pass


def run(variant, output, calibration_path, *, phase='screening', parent=None, smoke=False, resume=False, policy=None):
    policy = policy or Stage1Policy()
    codec = policy.codec
    review = policy.verify()
    recipe = policy.recipe(phase)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    calibration = read_calibration(calibration_path)
    make_dataset = getattr(policy, 'dataset', dataset)
    train_ds, _, cfg = make_dataset('train', recipe['batch'], epochs=recipe['epochs'])
    val_ds, val_data, _ = make_dataset('val', 8)
    require_idle_gpu()
    seed(recipe['seed'])
    for name, value in {'amp': recipe['amp'], 'optimizer': recipe['optimizer'], 'lr0': recipe['lr0'],
                        'lrf': recipe['linear_final_fraction'], 'momentum': recipe['betas'][0],
                        'nbs': recipe['logical_batch'], 'weight_decay': recipe['weight_decay'],
                        'warmup_epochs': recipe['warmup_epochs'], 'warmup_bias_lr': recipe['warmup_bias_lr'],
                        'patience': recipe['patience'], 'seed': recipe['seed'],
                        'cos_lr': recipe.get('scheduler') == 'warmup-cosine'}.items():
        setattr(cfg, name, value)
    model = policy.build(variant, phase, parent, calibration).to('cuda:0')
    model.args = cfg
    parameters = policy.configure(model, recipe['scope'])
    mutable = policy.mutable(model)
    gradient_names = policy.gradient_names(model)
    update_names = getattr(policy, 'update_names', policy.gradient_names)(model)
    score_for = getattr(policy, 'score', lambda report: report.get('official', {}).get('AP50_95', 0.))
    frozen = frozen_hashes(model, mutable)
    manifest = codec.manifest_for(model, sha256(calibration_path))
    ema, optimizer = FrozenEMA(model, mutable), policy.optimizer(model, recipe)
    scaler = torch.amp.GradScaler('cuda', enabled=recipe['amp'], init_scale=recipe['initial_loss_scale'])
    if not resume and hasattr(policy, 'initialize_state'):
        policy.initialize_state(model, ema, optimizer, scaler, recipe)
    ema_frozen = {n: h for n, h in frozen_hashes(ema.ema).items() if n not in ema.names}
    output.mkdir(parents=True, exist_ok=resume)
    if not resume:
        write_json(output / 'recipe.json', {'recipe': recipe, 'manifest': manifest,
                   'resolved_native_args': vars(cfg), 'reference_review': review,
                   'model_parameters': sum(p.numel() for p in model.parameters()),
                   'trainable_parameters': sum(p.numel() for p in parameters),
                   'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad],
                   'mutable_bn_buffers': sorted(mutable)})
    start_epoch, best, history = 0, {'AP50_95': -1., 'epoch': None, 'checkpoint': None}, []
    if resume:
        start_epoch, best, history = restore_resume(output / 'resume.pt', model, ema, optimizer, manifest,
                                                   recipe=recipe, scaler=scaler)
        if frozen_hashes(model, mutable) != frozen:
            raise ValueError('resumed frozen tensors differ from phase parent')
        # 上次若在 best 落盤後、resume 提交前中斷，刪除該未提交 epoch 的孤立產物再重跑。
        for path in output.glob('epoch-*.pt'):
            if path.name != best['checkpoint']:
                codec.load(path)  # 僅清理本 run 中可驗證的研究 checkpoint。
                path.unlink()
                path.with_suffix('.json').unlink()
    events = (output / 'progress.jsonl').open('a', buffering=1)

    def emit(event):
        event = {'time_unix': time.time(), 'variant': variant, 'phase': phase, **event}
        events.write(json.dumps(event) + '\n')
        write_json(output / 'status.json', event)

    def evaluate_epoch(epoch):
        evaluator = getattr(policy, 'evaluate', evaluate)
        report = evaluator(ema.ema, val_ds, val_data, output / f'evaluation-{epoch:02d}',
                          limit_batches=2 if smoke else None, progress=emit)
        write_json(output / f'metrics-{epoch:02d}.json', report)
        return report

    def persist(next_epoch):
        save_resume(output / 'resume.pt', model, ema, optimizer, manifest, next_epoch, best, history,
                    recipe=recipe, scaler=scaler)

    def result(stop_reason):
        write_json(output / 'result.json', {'variant': variant, 'phase': phase, 'smoke': smoke,
                   'recipe': recipe, 'best': best, 'history': history, 'stop_reason': stop_reason})

    try:
        emit({'kind': 'starting', 'smoke': smoke, 'next_epoch': start_epoch})
        if not resume:
            initial = evaluate_epoch(0)
            policy.guard_metrics(initial, [], smoke)
            history.append({'epoch': 0, 'evaluation': initial})
            score = score_for(initial)
            best = {'AP50_95': score, 'epoch': 0, 'checkpoint': save_best(output, ema.ema, manifest, 0, codec)}
            persist(0)
        if hasattr(policy, 'initialize_criterion'):
            policy.initialize_criterion(model, start_epoch)
        else:
            model.criterion = model.init_criterion()
            for _ in range(start_epoch):
                model.criterion.update()
        stop_reason = 'smoke' if smoke else 'budget'
        for epoch in range(start_epoch, 1 if smoke else recipe['epochs']):
            stop = getattr(policy, 'should_stop', lambda cfg, count, best_epoch, history:
                           should_stop(cfg, count, best_epoch))
            if not smoke and stop(recipe, epoch, best['epoch'], history):
                stop_reason = 'patience'
                break
            seed(recipe['seed'] + epoch)
            policy.configure(model, recipe['scope'])
            batches = loader(train_ds, recipe['batch'], recipe['seed'] + epoch)
            started, seen, steps, retries = time.time(), 0, 0, 0
            trace, pending = [], []
            before = {name: model.get_parameter(name).detach().clone() for name in update_names}
            torch.cuda.reset_peak_memory_stats()
            for index, raw in enumerate(batches):
                if (output / 'STOP').exists() or (output.parent / 'STOP').exists():
                    raise InterruptedError('STOP requested; latest complete epoch retained')
                item = transfer(raw, 'cuda:0')
                if index == 0:
                    trace = [Path(p).name for p in raw['im_file']]
                    if smoke:
                        from ultralytics.utils.plotting import plot_images
                        labels = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                                  for k, v in item.items() if k != 'img'}
                        plot_images(images=item['img'].detach().cpu().clone(), labels=labels,
                                    paths=item['im_file'], fname=output / 'train-targets.jpg', names=model.names)
                pending.append(item)
                seen += len(item['img'])
                if len(pending) < recipe['logical_batch'] // recipe['batch'] and index + 1 != len(batches):
                    continue
                lr = policy.set_lr(optimizer, recipe, epoch, (index + 1) / len(batches), steps, smoke)
                step = macro_step(model, pending, optimizer, scaler, parameters, recipe, gradient_names, policy.loss)
                pending.clear()
                ema.update(model)
                steps += 1
                retries += step['amp_retries']
                if steps == 1 or steps % 20 == 0:
                    elapsed = time.time() - started
                    emit({'kind': 'training', 'epoch': epoch + 1, 'images': seen, 'steps': steps,
                          **step, 'lr': lr, 'elapsed_seconds': elapsed,
                          'epoch_eta_seconds': elapsed / seen * (len(train_ds) - seen)})
                if smoke and steps == recipe.get('smoke_steps', 8):
                    break
            train_seconds = time.time() - started
            peak = torch.cuda.max_memory_allocated()
            if not smoke and seen != getattr(policy, 'expected_train_images', 118287):
                raise AssertionError('training epoch did not cover canonical full split')
            if frozen_hashes(model, mutable) != frozen:
                raise AssertionError('live frozen tensor drift')
            if {n: h for n, h in frozen_hashes(ema.ema).items() if n not in ema.names} != ema_frozen:
                raise AssertionError('EMA frozen tensor drift')
            deltas = {name: float((model.get_parameter(name) - old).abs().max().detach()) for name, old in before.items()}
            if any(value <= 0 for value in deltas.values()):
                raise AssertionError('a required Q/K/scale/bias parameter did not update')
            report = evaluate_epoch(epoch + 1)
            score = score_for(report)
            previous_best = None
            if (score > best['AP50_95'] and policy.accept_candidate(report, history)) or smoke:
                previous_best = best['checkpoint']
                best = {'AP50_95': score, 'epoch': epoch + 1,
                        'checkpoint': save_best(output, ema.ema, manifest, epoch + 1, codec)}
            history.append({'epoch': epoch + 1, 'images': seen, 'optimizer_steps': steps,
                            'seconds': time.time() - started, 'training_seconds': train_seconds,
                            'peak_allocated_bytes': peak, 'amp_retries': retries,
                            'first_batch_images': trace, 'required_update_abs_max': deltas,
                            'qk_update_abs_max': {n: v for n, v in deltas.items() if '.qkv.' in n},
                            'evaluation': report})
            model.criterion.update()
            persist(epoch + 1)
            policy.guard_metrics(report, history, smoke)
            if previous_best is not None:
                (output / previous_best).unlink()
                (output / previous_best).with_suffix('.json').unlink()
            if smoke:
                reference = copy.deepcopy(optimizer.state_dict())
                scale_state = copy.deepcopy(scaler.state_dict())
                restored, _, _ = restore_resume(output / 'resume.pt', model, ema, optimizer, manifest,
                                                recipe=recipe, scaler=scaler)
                if restored != epoch + 1 or not _same(reference, optimizer.state_dict()) or scale_state != scaler.state_dict():
                    raise AssertionError('resume optimizer/scaler state mismatch')
            result('running')
            emit({'kind': 'epoch-complete', 'epoch': epoch + 1, 'best': best})
            if not smoke and stop(recipe, epoch + 1, best['epoch'], history):
                stop_reason = 'patience'
                break
        result(stop_reason)
        emit({'kind': 'completed', 'best': best, 'smoke': smoke, 'stop_reason': stop_reason})
    except BaseException as error:
        emit({'kind': 'stopped' if isinstance(error, InterruptedError) else 'failed',
              'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        events.close()


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=BASES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('screening', 'recovery'), default='screening')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--calibration', type=Path, default=ROOT / 'yolo_binaryqk/artifacts/stage1/calibration.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        run(args.variant, args.output, args.calibration, phase=args.phase, parent=args.parent,
            smoke=args.smoke, resume=args.resume)
    except GPUInUseError as error:
        print(str(error), flush=True)
        raise SystemExit(75) from error


if __name__ == '__main__':
    main()
