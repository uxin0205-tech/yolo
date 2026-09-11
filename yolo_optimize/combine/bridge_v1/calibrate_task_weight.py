"""固定 checkpoint 的 train-only 梯度幅度校準，不挑 validation 權重。"""
import json
import statistics
from dataclasses import asdict
import torch
from safe_source import HERE
from merge_joint import install, JointSource
import runtime
from yolo_combine.factory import FusionModelFactory
from yolo_combine.joint_loss import MacroStepEngine, NativeTaskLossRouter
from yolo_combine.stage_policy import build_joint_optimizer
from yolo_combine.contracts import Task
from yolo_combine.hardware_contract import HardwareContractGuard

NAME = 'task-weight-calibration-v1'


def main():
    config, stage = install()
    runtime.formal.seed_everything(config.seed)
    out = HERE / 'artifacts' / NAME
    out.mkdir(exist_ok=False)
    session = runtime.formal.FormalJointTrainingSession(config, device='0', run_name=NAME)
    session.run_dir.mkdir(parents=True, exist_ok=False)
    built = FusionModelFactory(JointSource(), detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint, checkpoint_kind='float')
    model = built.model.to('cuda:0')
    dl, pl, _ = session._loaders(model)
    assert len(dl.loader.dataset) == 118287 and len(pl.loader.dataset) == 5964
    optimizer, _ = build_joint_optimizer(model, stage, optimizer_name='AdamW',
        weight_decay=config.weight_decay, beta1=config.beta1, beta2=config.beta2)
    runtime.formal._apply_shared_bn_affine(model, trainable=False)
    for group in optimizer.param_groups:
        group['lr'] = 0.
    initial = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    guard = HardwareContractGuard.capture(model)
    router = NativeTaskLossRouter(model, epochs=stage.epochs, imgsz=640)
    engine = MacroStepEngine(model=model,
        losses=runtime.impl._AutocastTaskLossRouter(router, device=torch.device('cuda:0'), enabled=True),
        optimizer=optimizer, reference_batch_size=config.reference_batch_size,
        task_weights={Task.DETECT:1., Task.POSE:.25}, gradient_groups=runtime.impl._gradient_groups(model),
        scaler=torch.amp.GradScaler('cuda', init_scale=1024), ema=None,
        max_grad_norm=config.gradient_clip_norm, max_amp_retries=16,
        preprocess=lambda task,batch: dl.preprocess(batch) if task is Task.DETECT else pl.preprocess(batch))
    di, pi = iter(dl.loader), iter(pl.loader)
    results = []
    selected = None
    for i in range(24):
        model.load_state_dict(initial, strict=True)
        if i == 16:
            median = statistics.median(r['ratio'] for r in results)
            selected = max(.01, min(.25, round((.25 / median)/.005)*.005))
            engine.task_weights[Task.POSE] = selected
        report = engine.run(detect_batches=tuple(next(di) for _ in range(8)),
            pose_batches=(next(pi),), record_gradient_statistics=True)
        stats = asdict(report.gradient_statistics)
        assert stats['detect_norm'] > 0 and stats['pose_norm'] > 0
        results.append({'index':i, 'phase':'calibration' if i<16 else 'confirmation',
            'pose_weight':engine.task_weights[Task.POSE], **stats,
            'ratio':stats['pose_norm']/stats['detect_norm'], 'amp_retries':report.amp_overflow_retries})
    model.load_state_dict(initial, strict=True)
    guard.assert_unchanged(model)
    assert all(torch.equal(v, model.state_dict()[n].detach().cpu()) for n,v in initial.items())
    calibration = [r for r in results if r['phase']=='calibration']
    confirmation = [r for r in results if r['phase']=='confirmation']
    observed = statistics.median(r['ratio'] for r in confirmation)
    summary = {'status':'passed' if .5<=observed<=2. else 'needs_review',
        'selected_pose_weight':selected, 'target_ratio':1., 'confirmation_band':[.5,2.],
        'calibration_median_ratio':statistics.median(r['ratio'] for r in calibration),
        'confirmation_median_ratio':observed, 'source':JointSource().provenance(),
        'detect_images':6144, 'pose_images':384, 'validation_used':False,
        'checkpoint_state_restored_each_macro':True, 'optimizer_lr_zero':True, 'results':results}
    with (out/'summary.json').open('x') as f:
        json.dump(summary,f,indent=2)
    print('JOB_DONE: training-only task-weight calibration '+summary['status'],flush=True)


if __name__ == '__main__':
    main()
