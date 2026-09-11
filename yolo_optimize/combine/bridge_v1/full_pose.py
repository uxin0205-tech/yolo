"""独立完整 Pose 適應；保留 Detect 錨點，不是已接受融合模型。"""
from dataclasses import asdict, replace
from types import MappingProxyType
import json
import sys
import torch
from safe_source import HERE, sha256
from adapted_source import AdaptedBridgeSource
import pose_first
from ultralytics.utils.torch_utils import ModelEMA
from yolo_combine.metrics import AccuracyGate

START = HERE / 'artifacts/fusion/j0-pose-extend-v1/inference/best_pose.pt'
START_SHA = '1b9803b6cfaacc0dad59ecd3fcffaea976bc50c45d828be5981323c941047b05'


class PoseModelEMA(ModelEMA):
    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay(self.updates)
        active = {n for n, p in model.named_parameters() if p.requires_grad}
        live = model.state_dict()
        for name, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point and (name in active or '.pose_head.' in name):
                value.mul_(decay).add_(live[name].detach(), alpha=1-decay)


class FullPoseSource(AdaptedBridgeSource):
    def provenance(self, kind='float'):
        return {**super().provenance(kind), 'pose_extension': str(START),
            'pose_extension_sha256': START_SHA, 'independent_pose_training_not_fused_acceptance': True}

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        assert sha256(START) == START_SHA
        payload = torch.load(START, map_location='cpu', weights_only=True)
        assert payload['metadata']['epoch'] == 36 and payload['metadata']['stage'] == 'j0'
        prefix = 'graph.model.23.pose_head.'
        head = {n[len(prefix):]: v for n, v in payload['state_dict'].items() if n.startswith(prefix)}
        pair.pose.model[23].load_state_dict(head, strict=True)
        return pair


def install():
    config, stage = pose_first.install()
    stage = replace(stage, epochs=60, backbone_start_layer=0,
        learning_rates=MappingProxyType({**dict(stage.learning_rates), 'backbone': 1.5e-5, 'neck': 7.5e-5}))
    pose_first.impl.JOINT_STAGES = MappingProxyType({**dict(pose_first.impl.JOINT_STAGES), 'j0': stage})
    pose_first.impl.SourceBundle = FullPoseSource
    pose_first.impl.ModelEMA = PoseModelEMA
    pose_first.impl.AccuracyGate = AccuracyGate
    return config, stage


class Session(pose_first.formal.FormalJointTrainingSession):
    def _resolved_config(self):
        result = super()._resolved_config()
        result['stage_policies']['j0'].update(epochs=60, backbone_start_layer=0,
            learning_rates=dict(pose_first.impl.JOINT_STAGES['j0'].learning_rates))
        result['full_pose_policy'] = {'independent_pose_not_accepted_fusion': True,
            'detect_checkpoint_untouched': True, 'coco_metrics': 'diagnostic: frozen Detect head on adapted Pose trunk',
            'head_lr': 2e-4, 'neck_lr': 7.5e-5, 'backbone_lr': 1.5e-5,
            'shared_bn_stats_and_affine_fixed': True, 'attention_fixed': True,
            'patience': 12, 'min_delta': .0001, 'warmup_epochs': 1,
            'source_sha256': START_SHA}
        return result

    def _save_selected(self, labels, **kwargs):
        saved = super()._save_selected(labels, **kwargs)
        metrics = kwargs['metrics']
        start = torch.load(START, map_location='cpu', weights_only=True)['metadata']['metrics']
        keys = [k for k in self.config.load_baseline() if k.startswith('bbat/')]
        score = sum(metrics[k] for k in keys) / len(keys)
        best = getattr(self, '_best_pose', float('-inf'))
        self._stale = 0 if score > best + .0001 else getattr(self, '_stale', 0) + 1
        self._best_pose = max(best, score)
        failures = {k: metrics[k]-start[k] for k in keys if metrics[k] < start[k]-.03}
        if failures or self._stale >= 12:
            with (self.run_dir / 'pose-stop.json').open('x') as f:
                json.dump({'status': 'POSE_SAFETY_STOP' if failures else 'POSE_PLATEAU',
                    'delta': failures, 'best_score': self._best_pose,
                    'saved': {k: str(v) for k, v in saved.items()}}, f, indent=2)
            raise pose_first.PosePlateau()
        return saved


def smoke(config, stage):
    session = Session(config, device='0', run_name='full-pose-smoke-v2')
    session.run_dir.mkdir(parents=True, exist_ok=False)
    built = pose_first.FusionModelFactory(FullPoseSource(), detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint, checkpoint_kind='float')
    model = built.model.to('cuda:0')
    _, loader, _ = session._loaders(model)
    assert len(loader.loader.dataset) == 5964
    optimizer, grouping = pose_first.policy.build_joint_optimizer(model, stage, optimizer_name=config.optimizer,
        weight_decay=config.weight_decay, beta1=config.beta1, beta2=config.beta2)
    pose_first.formal._apply_shared_bn_affine(model, trainable=False)
    active = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith('graph.model.0.') for n in active)
    assert any(n.startswith('graph.model.16.') for n in active)
    assert any('.pose_head.' in n for n in active)
    assert not any('.detect_head.' in n for n in active)
    fixed = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    detect_state = {n: v.detach().cpu().clone() for n, v in model.detect_head.state_dict().items()}
    guard = pose_first.HardwareContractGuard.capture(model)
    ema = PoseModelEMA(model)
    router = pose_first.NativeTaskLossRouter(model, epochs=60, imgsz=640,
        detect_overrides={'epochs': 0}, pose_overrides={'epochs': 60})
    engine = pose_first.MacroStepEngine(model=model,
        losses=pose_first.impl._AutocastTaskLossRouter(router, device=torch.device('cuda:0'), enabled=True),
        optimizer=optimizer, reference_batch_size=config.reference_batch_size,
        task_weights={'detect': config.detect_weight, 'pose': config.pose_weight},
        gradient_groups=pose_first.impl._gradient_groups(model), scaler=torch.amp.GradScaler('cuda'), ema=ema,
        max_grad_norm=config.gradient_clip_norm, max_amp_retries=config.amp_max_overflow_retries,
        preprocess=lambda task, batch: loader.preprocess(batch))
    iterator = iter(loader.loader)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        engine.run(detect_batches=(), pose_batches=(next(iterator),), record_gradient_statistics=True)
    # 同時檢查長期 EMA 衰减區間，固定 Detect 不參與浮點混合。
    ema.updates = 4000
    ema.update(model)
    for candidate in (model, ema.ema):
        guard.assert_unchanged(candidate)
        params = dict(candidate.named_parameters())
        assert all(torch.equal(v, params[n].detach().cpu()) for n, v in fixed.items())
        assert all(torch.equal(v, candidate.detect_head.state_dict()[n].detach().cpu()) for n,v in detect_state.items())
    params = dict(model.named_parameters())
    for prefix in ('graph.model.0.', 'graph.model.16.', 'graph.model.23.pose_head.'):
        assert any(not torch.equal(params[n].detach(), v) for n,v in active.items() if n.startswith(prefix))
    with (session.run_dir / 'summary.json').open('x') as f:
        json.dump({'status': 'passed', 'backbone_neck_pose_updates': True, 'fixed_live_ema_exact': True,
            'detect_head_exact': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'optimizer': asdict(grouping)}, f, indent=2, default=str)
    print('JOB_DONE: full Pose smoke passed', flush=True)


def main():
    config, stage = install()
    pose_first.formal.seed_everything(config.seed)
    if '--smoke' in sys.argv:
        smoke(config, stage)
        return
    assert json.loads((HERE / 'artifacts/fusion/full-pose-smoke-v2/summary.json').read_text())['status'] == 'passed'
    session = Session(config, device='0', run_name='full-pose-adamw-v1')
    try:
        report = session.run()
    except pose_first.PosePlateau:
        print('JOB_DONE: full Pose saved stop; review required', flush=True)
        return
    with (session.run_dir / 'summary.json').open('x') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print('JOB_DONE: full Pose adaptation complete; fusion not yet accepted', flush=True)


if __name__ == '__main__':
    main()
