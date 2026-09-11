"""只延長 Pose head 適應；絕不自動解凍或執行 J1。"""
from dataclasses import asdict, fields, replace
from types import MappingProxyType
import json
import sys
import torch
from safe_source import HERE, initialize
from adapted_source import AdaptedBridgeSource
from runtime import JointConfig
from j0_runtime import J0Config, PoseOnlyEMA, J0Gate
import yolo_combine._formal_training_impl as impl
import yolo_combine.formal_training as formal
import yolo_combine.stage_policy as policy
import yolo_combine.validation as validation
from validate import InternalValidator
from yolo_combine.factory import FusionModelFactory
from yolo_combine.joint_loss import MacroStepEngine, NativeTaskLossRouter
from yolo_combine.hardware_contract import HardwareContractGuard


def install():
    initialize()
    torch.set_num_threads(4)
    original = JointConfig.load(HERE / 'full35/configs/j1.yaml')
    values = {f.name: getattr(original, f.name) for f in fields(original) if f.init}
    values['stages'] = ('j0',)
    config = J0Config(**values)
    config._validate()
    assert config.preflight().ready
    stage = replace(policy.JOINT_STAGES['j0'], epochs=40, patience=0)
    impl.JOINT_STAGES = MappingProxyType({**dict(policy.JOINT_STAGES), 'j0': stage})
    impl.SourceBundle = AdaptedBridgeSource
    impl.ModelEMA = PoseOnlyEMA
    impl.AccuracyGate = J0Gate
    validation.DetectionValidator = InternalValidator
    return config, stage


class PosePlateau(RuntimeError):
    pass


class Session(formal.FormalJointTrainingSession):
    def _resolved_config(self):
        result = super()._resolved_config()
        result['pose_first_extension'] = {'epochs': 40, 'patience': 10, 'min_delta': .0001,
            'monitor': 'mean of six BBAT AP metrics', 'fresh_optimizer': True,
            'source': 'pre-J1 J0 best Pose head; original P3 bridge Detect',
            'next_joint_requires_review': True}
        result['stage_policies']['j0']['epochs'] = 40
        return result

    def _save_selected(self, labels, **kwargs):
        saved = super()._save_selected(labels, **kwargs)
        metrics = kwargs['metrics']
        names = [k for k in self.config.load_baseline() if k.startswith('bbat/')]
        score = sum(metrics[k] for k in names) / len(names)
        best = getattr(self, '_pose_best', float('-inf'))
        self._pose_stale = 0 if score > best + .0001 else getattr(self, '_pose_stale', 0) + 1
        self._pose_best = max(best, score)
        if self._pose_stale >= 10:
            with (self.run_dir / 'pose-plateau.json').open('x') as f:
                json.dump({'status': 'POSE_PLATEAU', 'best_score': self._pose_best,
                    'saved': {k: str(v) for k, v in saved.items()}, 'joint_authorized_automatically': False}, f, indent=2)
            raise PosePlateau()
        return saved


def smoke(config, stage):
    session = Session(config, device='0', run_name='j0-extend-smoke-v1')
    session.run_dir.mkdir(parents=True, exist_ok=False)
    built = FusionModelFactory(AdaptedBridgeSource(), detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint, checkpoint_kind='float')
    model = built.model.to('cuda:0')
    _, loader, _ = session._loaders(model)
    assert len(loader.loader.dataset) == 5964
    optimizer, grouping = policy.build_joint_optimizer(model, stage, optimizer_name=config.optimizer,
        weight_decay=config.weight_decay, beta1=config.beta1, beta2=config.beta2)
    assert all('.pose_head.' in n for n, p in model.named_parameters() if p.requires_grad)
    fixed = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if '.pose_head.' not in n}
    active = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    ema = PoseOnlyEMA(model)
    guard = HardwareContractGuard.capture(model)
    router = NativeTaskLossRouter(model, epochs=stage.epochs, imgsz=640, detect_overrides={'epochs': 0}, pose_overrides={'epochs': stage.epochs})
    engine = MacroStepEngine(model=model,
        losses=impl._AutocastTaskLossRouter(router, device=torch.device('cuda:0'), enabled=True),
        optimizer=optimizer, reference_batch_size=config.reference_batch_size,
        task_weights={'detect': config.detect_weight, 'pose': config.pose_weight},
        gradient_groups=impl._gradient_groups(model), scaler=torch.amp.GradScaler('cuda'), ema=ema,
        max_grad_norm=config.gradient_clip_norm, max_amp_retries=config.amp_max_overflow_retries,
        preprocess=lambda task, batch: loader.preprocess(batch))
    iterator = iter(loader.loader)
    for _ in range(2):
        engine.run(detect_batches=(), pose_batches=(next(iterator),), record_gradient_statistics=True)
    for candidate in (model, ema.ema):
        guard.assert_unchanged(candidate)
        state = candidate.state_dict()
        assert all(torch.equal(v, state[n].detach().cpu()) for n, v in fixed.items())
    params = dict(model.named_parameters())
    assert any(not torch.equal(params[n].detach(), v) for n, v in active.items())
    with (session.run_dir / 'summary.json').open('x') as f:
        json.dump({'status': 'passed', 'pose_images': 32, 'fixed_live_and_ema_exact': True,
            'optimizer': asdict(grouping)}, f, indent=2, default=str)
    print('JOB_DONE: Pose-only extension smoke passed', flush=True)


def main():
    config, stage = install()
    formal.seed_everything(config.seed)
    if '--smoke' in sys.argv:
        smoke(config, stage)
        return
    assert json.loads((HERE / 'artifacts/fusion/j0-extend-smoke-v1/summary.json').read_text())['status'] == 'passed'
    session = Session(config, device='0', run_name='j0-pose-extend-v1')
    try:
        report = session.run()
    except PosePlateau:
        print('JOB_DONE: Pose plateau saved; review before fusion', flush=True)
        return
    with (session.run_dir / 'summary.json').open('x') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print('JOB_DONE: Pose-only adaptation complete; review before fusion', flush=True)


if __name__ == '__main__':
    main()
