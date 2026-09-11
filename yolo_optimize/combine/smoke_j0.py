"""完整 canonical loader 的前兩個 batch；只驗證更新，不作精度結論。"""
from dataclasses import asdict
import json
from j0_runtime import install, formal, impl, PoseOnlyEMA
from local_source import HERE, LocalSource
import torch
from yolo_combine.factory import FusionModelFactory
from yolo_combine.stage_policy import JOINT_STAGES, build_joint_optimizer
from yolo_combine.hardware_contract import HardwareContractGuard
from yolo_combine.joint_loss import MacroStepEngine, NativeTaskLossRouter
from yolo_combine.joint_trainer import StageWarmupCosineScheduler


def main(run_name='j0-smoke-v1', source_class=LocalSource):
    from j0_runtime import require_training_enabled
    require_training_enabled()
    config = install()
    formal.seed_everything(config.seed)
    session = formal.FormalJointTrainingSession(config, device='0', run_name=run_name)
    session.run_dir.mkdir(parents=True, exist_ok=False)
    source = source_class()
    built = FusionModelFactory(source, detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint, checkpoint_kind='float')
    model = built.model.to('cuda:0')
    detect_loader, pose_loader, _ = session._loaders(model)
    assert len(pose_loader.loader.dataset) == 5964
    optimizer, grouping = build_joint_optimizer(model, JOINT_STAGES['j0'],
        optimizer_name=config.optimizer, weight_decay=config.weight_decay, beta1=config.beta1, beta2=config.beta2)
    active = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert active and all('.pose_head.' in n for n in active)
    fixed = {n: t.detach().cpu().clone() for n, t in model.state_dict().items() if '.pose_head.' not in n}
    guard = HardwareContractGuard.capture(model)
    ema = PoseOnlyEMA(model, decay=0.9999, tau=2000)
    router = NativeTaskLossRouter(model, epochs=8, imgsz=640, detect_overrides={'epochs': 0}, pose_overrides={'epochs': 8})
    losses = impl._AutocastTaskLossRouter(router, device=torch.device('cuda:0'), enabled=True)
    engine = MacroStepEngine(model=model, losses=losses, optimizer=optimizer,
        reference_batch_size=config.reference_batch_size,
        task_weights={'detect': config.detect_weight, 'pose': config.pose_weight},
        gradient_groups=impl._gradient_groups(model), scaler=torch.amp.GradScaler('cuda'), ema=ema,
        max_grad_norm=config.gradient_clip_norm, max_amp_retries=config.amp_max_overflow_retries,
        preprocess=lambda task, batch: pose_loader.preprocess(batch))
    scheduler = StageWarmupCosineScheduler(optimizer, stage='j0', epochs=8,
        steps_per_epoch=len(pose_loader.loader), warmup_epochs=1,
        warmup_start_factor=config.warmup_start_factor, final_lr_factor=config.cosine_final_lr_factor)
    iterator = iter(pose_loader.loader)
    reports = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        scheduler.prepare_step()
        report = engine.run(detect_batches=(), pose_batches=(next(iterator),), record_gradient_statistics=True)
        scheduler.advance()
        reports.append(asdict(report))
    guard.assert_unchanged(model)
    for candidate in (model, ema.ema):
        state = candidate.state_dict()
        assert all(torch.equal(value, state[n].detach().cpu()) for n, value in fixed.items())
    params = dict(model.named_parameters())
    update = sum(float((params[n].detach() - p).square().sum()) for n, p in active.items()) ** .5
    assert 0 < update < float('inf') and ema.updates == 2
    masf_updates = {n: float((params[n].detach() - p).abs().max())
                    for n, p in active.items() if '.p3_masf.' in n}
    if masf_updates:
        assert any(v > 0 for n, v in masf_updates.items() if n.endswith('.alpha'))
        assert any(v > 0 for n, v in masf_updates.items() if not n.endswith('.alpha'))
    masf_gradients = {n: float(p.grad.detach().abs().max())
                      for n, p in params.items() if '.p3_masf.' in n and p.grad is not None}
    if masf_updates:
        assert all(torch.isfinite(p.grad).all() for n, p in params.items()
                   if '.p3_masf.' in n and p.grad is not None)
        assert any(v > 0 for n, v in masf_gradients.items() if n.endswith('.alpha'))
        assert any(v > 0 for n, v in masf_gradients.items() if not n.endswith('.alpha'))
    with (session.run_dir / 'summary.json').open('x') as handle:
        json.dump({'status': 'passed', 'pose_train_images': 5964, 'smoke_images': 32,
            'fixed_live_and_ema_exact': True, 'pose_parameter_update_l2': update,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'optimizer': asdict(grouping), 'reports': reports,
            'masf_parameter_max_updates': masf_updates,
            'masf_gradient_max_abs': masf_gradients}, handle, indent=2, default=str)
    print('JOB_DONE: J0 真實 Pose 更新及固定 Detect／共享權重驗證通過', flush=True)


if __name__ == '__main__':
    main()
