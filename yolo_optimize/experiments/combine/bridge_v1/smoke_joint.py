"""完整 loader 兩個實際 joint macro：256 Detect＋16 Pose／macro。"""
from dataclasses import asdict
import json
import math
from safe_source import HERE
from adapted_source import AdaptedBridgeSource
from runtime import install, Session, ActiveEMA, formal, impl
import torch
from yolo_combine.factory import FusionModelFactory
from yolo_combine.stage_policy import build_joint_optimizer
from yolo_combine.hardware_contract import HardwareContractGuard
from yolo_combine.joint_loss import MacroStepEngine, NativeTaskLossRouter
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
from yolo_combine.contracts import Task


def main():
    config, stage = install()
    formal.seed_everything(config.seed)
    session = Session(config, device='0', run_name='j1-smoke-v1')
    session.run_dir.mkdir(parents=True, exist_ok=False)
    built = FusionModelFactory(AdaptedBridgeSource(), detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint, checkpoint_kind='float')
    model = built.model.to('cuda:0')
    dl, pl, _ = session._loaders(model)
    assert len(dl.loader.dataset) == 118287 and len(pl.loader.dataset) == 5964
    optimizer, grouping = build_joint_optimizer(model, stage, optimizer_name='AdamW',
        weight_decay=config.weight_decay, beta1=config.beta1, beta2=config.beta2)
    formal._apply_shared_bn_affine(model, trainable=config.shared_bn_affine_trainable)
    active = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    fixed = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert 'graph.model.23.detect_head.p3_masf.alpha' in active
    assert any(n.startswith('graph.model.0.') for n in active) == (stage.name == 'j3')
    guard = HardwareContractGuard.capture(model)
    masf_bn = {n: t.detach().cpu().clone() for n, t in model.detect_head.p3_masf.named_buffers()}
    ema = ActiveEMA(model, decay=.9999, tau=2000)
    router = NativeTaskLossRouter(model, epochs=stage.epochs, imgsz=640)
    losses = impl._AutocastTaskLossRouter(router, device=torch.device('cuda:0'), enabled=True)
    gradient_records = []
    # 在 optimizer step 前（unscale／clip 後）觀測，避免讀到引擎清零後的 .grad。
    def capture(opt, args, kwargs):
        gradients = {n: float(p.grad.abs().max()) for n, p in model.named_parameters()
                     if '.p3_masf.' in n and p.grad is not None}
        assert gradients and all(math.isfinite(v) for v in gradients.values())
        assert any(v > 0 for n, v in gradients.items() if n.endswith('.alpha'))
        assert any(v > 0 for n, v in gradients.items() if not n.endswith('.alpha'))
        gradient_records.append(gradients)
    hook = optimizer.register_step_pre_hook(capture)
    engine = MacroStepEngine(model=model, losses=losses, optimizer=optimizer,
        reference_batch_size=config.reference_batch_size,
        task_weights={Task.DETECT: config.detect_weight, Task.POSE: config.pose_weight},
        gradient_groups=impl._gradient_groups(model), scaler=torch.amp.GradScaler('cuda', init_scale=1024),
        ema=ema, max_grad_norm=10, max_amp_retries=16,
        preprocess=lambda task, batch: dl.preprocess(batch) if task is Task.DETECT else pl.preprocess(batch))
    count = session.detect_microbatches_per_macro
    assert count * session.detect_microbatch_size == 256
    schedule = StageWarmupCosineScheduler(optimizer, stage=stage.name, epochs=stage.epochs,
        steps_per_epoch=math.ceil(len(dl.loader)/count), warmup_epochs=1,
        warmup_start_factor=.1, final_lr_factor=.5)
    detect_iter, pose_iter = iter(dl.loader), iter(pl.loader)
    reports = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        schedule.prepare_step()
        report = engine.run(detect_batches=tuple(next(detect_iter) for _ in range(count)),
            pose_batches=(next(pose_iter),), record_gradient_statistics=True)
        schedule.advance()
        reports.append(asdict(report))
    hook.remove()
    assert len(gradient_records) == ema.updates == 2
    for candidate in (model, ema.ema):
        guard.assert_unchanged(candidate)
        params = dict(candidate.named_parameters())
        assert all(torch.equal(v, params[n].detach().cpu()) for n, v in fixed.items())
        buffers = dict(candidate.detect_head.p3_masf.named_buffers())
        assert all(torch.equal(v, buffers[n].detach().cpu()) for n, v in masf_bn.items())
    params = dict(model.named_parameters())
    if stage.name == 'j3':
        assert any(float((params[n].detach()-v).abs().max()) > 0
                   for n,v in active.items() if n.startswith('graph.model.0.'))
        assert any(float((params[n].detach()-v).abs().max()) > 0
                   for n,v in active.items() if '.attn.' in n)
    if stage.name == 'j2':
        assert any(n.startswith('graph.model.9.') for n in active)
        assert any(float((params[n].detach()-v).abs().max()) > 0
                   for n,v in active.items() if n.startswith('graph.model.9.'))
    updates = {n: float((params[n].detach()-v).abs().max()) for n, v in active.items() if '.p3_masf.' in n}
    assert updates['graph.model.23.detect_head.p3_masf.alpha'] > 0
    with (session.run_dir / 'summary.json').open('x') as handle:
        json.dump({'status': 'passed', 'detect_images': 512, 'pose_images': 32,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'masf_gradients_before_step': gradient_records, 'masf_updates': updates,
            'fixed_live_and_ema_exact': True, 'masf_bn_exact': True,
            'optimizer': asdict(grouping), 'reports': reports}, handle, indent=2, default=str)
    print('JOB_DONE: J1 真實混合更新、alpha 梯度、固定 BN／硬體契約通過。', flush=True)


if __name__ == '__main__':
    main()
