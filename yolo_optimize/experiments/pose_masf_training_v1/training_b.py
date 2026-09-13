"""B 組：固定 E2，共享／Detect 全固定，僅訓 Pose head＋獨立 P3 MASF。"""
import argparse
import copy
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch import nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pose_masf_priority_v1'))
import pose_candidate as parent
from pose_candidate import initialize, save, sha256, verify_pwl, tensors
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.joint_loss import MacroStepEngine, NativeTaskLossRouter
from yolo_combine.joint_data import TaskLoaderSettings, build_task_loader
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
from yolo_combine._formal_training_impl import _AutocastTaskLossRouter, seed_everything, reseed_loader_for_epoch
from yolo_combine.resume import TrainingProgress, save_training_snapshot, save_inference_weights, load_training_snapshot
from yolo_combine.contracts import Task
from yolo_combine.data import prepare_bbt5_view
from ultralytics.utils.torch_utils import ModelEMA

PREFIX = 'graph.model.23.pose_head.'
CONFIG = HERE / 'execution-config.json'
RUN = HERE / 'artifacts/b-e5-seed1-v1'

class PoseTrainingMASF(parent.PoseP3BridgeMASF):
    def forward(self, x):
        if not self.training:
            return super().forward(x)
        assert self.end2end and self.bridge_coefficient == 1.0
        assert not any(m.training for m in self.p3_masf.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
        many = self.forward_head([self.p3_masf(x[0]), x[1], x[2]], **self.one2many)
        # detach 在 MASF 前；beta=1 前／反向均 identity，不再壓小部署分支梯度。
        one = self.forward_head([self.p3_masf(x[0].detach()), x[1].detach(), x[2].detach()], **self.one2one)
        return {'one2many': many, 'one2one': one}

class TrainingSource(parent.PoseSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enabled=True, **kwargs)
    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        pair.pose.model[-1].__class__ = PoseTrainingMASF
        pair.pose.model[-1].bridge_coefficient = 1.0
        return pair
    def provenance(self, kind='float'):
        result = super().provenance(kind)
        result.update(experiment='pose_masf_training_v1', pose_masf_initialization='Detect context copy; alpha=0',
                      new_pose_masf_training_performed='see checkpoint epoch metadata', pose_bridge_beta=1.0,
                      canceled_arm='A: Pose head-only additional training', active_arm='B: Pose head + P3 MASF')
        return result

def build():
    source = TrainingSource()
    pair = source.build_task_models('float')
    model, report = assemble_graph_shared_model(pair.detect, pair.pose)
    assert report.complete
    original = parent.payload()['state_dict']
    state = model.state_dict()
    assert set(original) <= set(state)
    assert all(n.startswith(PREFIX + 'p3_masf.') for n in set(state) - set(original))
    state.update(original)
    model.load_state_dict(state, strict=True)
    model.pose_head.p3_masf.load_state_dict(model.detect_head.p3_masf.state_dict(), strict=True)
    with torch.no_grad(): model.pose_head.p3_masf.alpha.zero_()
    for n, p in model.named_parameters(): p.requires_grad_(n.startswith(PREFIX))
    set_training(model)
    return model, source

def set_training(model):
    model.eval()
    model.pose_head.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()

def move(model, device):
    model.to(device)
    for head in (model.detect_head, model.pose_head):
        for name in ('stride', 'anchors', 'strides'):
            setattr(head, name, getattr(head, name).to(device))
        head.shape = None
    return model

class FrozenGuard:
    def __init__(self, model):
        self.active = {n for n, p in model.named_parameters() if p.requires_grad}
        assert self.active and all(n.startswith(PREFIX) for n in self.active)
        self.fixed = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n not in self.active}
    def check(self, model):
        current = model.state_dict()
        for n, v in self.fixed.items():
            assert torch.equal(current[n].detach().cpu(), v), '固定 state 改變：' + n
        assert not any(m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))

class PoseEMA(ModelEMA):
    def __init__(self, model):
        self.active = {n for n, p in model.named_parameters() if p.requires_grad}
        super().__init__(model, decay=.9999, tau=2000)
    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        current = model.state_dict()
        for n, v in self.ema.state_dict().items():
            if n in self.active:
                assert v.dtype.is_floating_point
                v.mul_(d).add_(current[n].detach(), alpha=1-d)

def optimizer_for(model, cfg):
    groups = {}
    modules = dict(model.named_modules())
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        role = 'pose_head'
        if '.p3_masf.' in n: role = 'pose_masf_alpha' if n.endswith('.alpha') else 'pose_masf_context'
        owner = modules[n.rsplit('.', 1)[0]]
        no_decay = n.endswith(('.bias', '.alpha')) or isinstance(owner, nn.modules.batchnorm._BatchNorm)
        key = role + ('/no_decay' if no_decay else '/decay')
        group = groups.setdefault(key, dict(params=[], param_names=[], group_name=key, role=role,
            lr=cfg['peak_lr'][role], weight_decay=0.0 if no_decay else cfg['weight_decay']))
        group['params'].append(p); group['param_names'].append(n)
    return torch.optim.AdamW(list(groups.values()), betas=tuple(cfg['betas']))

def batches_of(loader, count):
    iterator = iter(loader)
    while True:
        group = tuple(itertools.islice(iterator, count))
        if not group: return
        yield group

def state_bundle(model, cfg, device):
    optimizer = optimizer_for(model, cfg)
    ema = PoseEMA(model)
    losses = _AutocastTaskLossRouter(NativeTaskLossRouter(model, epochs=cfg['epochs'], imgsz=640),
                                    device=device, enabled=device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=1024)
    scheduler = StageWarmupCosineScheduler(optimizer, stage='pose_b', epochs=5, steps_per_epoch=47,
        warmup_epochs=1, warmup_start_factor=.1, final_lr_factor=.5)
    return dict(optimizer=optimizer, ema=ema, criteria=losses, scaler=scaler, scheduler=scheduler)

def validate(model, source, output, epoch, kind):
    from common import prepare_coco
    from validate import InternalValidator
    import yolo_combine.validation as validation
    validation.DetectionValidator = InternalValidator
    original_pose = validation.PoseValidator
    class CountPose(original_pose):
        last_instance = None
        def init_metrics(self, m):
            super().init_metrics(m); type(self).last_instance = self
    validation.PoseValidator = CountPose
    try:
        view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml', HERE/'artifacts/datasets/bbat5-v1-runtime')
        validator = validation.JointValidator(source, detect_data_yaml=prepare_coco(), pose_data_yaml=view.yaml,
            output_root=output, settings=validation.ValidationSettings(imgsz=640, detect_batch_size=32,
                pose_batch_size=16, detect_workers=4, pose_workers=4, device='0', plots=False, save_coco_json=False))
        result = validator.validate(model.eval(), epoch=epoch, kind=kind)
        assert len(InternalValidator.last_instance.dataloader.dataset) == 5000
        assert len(CountPose.last_instance.dataloader.dataset) == 683
        metrics = dict(result.metrics)
        assert all(math.isfinite(v) for v in metrics.values())
        if kind == 'bittrue':
            expected = parent.payload()['metadata']['metrics']
            assert all(abs(metrics[k]-v) <= 1e-8 for k, v in expected.items() if k.startswith('coco/'))
        return metrics
    finally:
        validation.PoseValidator = original_pose
        InternalValidator.last_instance = None
        CountPose.last_instance = None

def run(mode):
    cfg = json.loads(CONFIG.read_text())
    initialize(); torch.set_num_threads(4); seed_everything(cfg['seed'])
    from activation_lab.training.full35 import _fp32_bbox_iou
    import ultralytics.utils.loss as loss_module
    loss_module.bbox_iou = _fp32_bbox_iou(loss_module.bbox_iou)
    assert torch.cuda.is_available()
    out = HERE/'artifacts/smoke-v1' if mode == 'smoke' else RUN
    if (out/'summary.json').exists():
        assert json.loads((out/'summary.json').read_text())['status'] == 'completed'
        print('JOB_DONE already_completed ' + mode, flush=True); return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    model, source = build(); guard = FrozenGuard(model); move(model, device)
    bundle = state_bundle(model, cfg, device)
    view = prepare_bbt5_view(cfg['dataset_registry'], HERE/'artifacts/datasets/bbat5-v1-runtime')
    loader = build_task_loader(model, data_yaml=view.yaml, settings=TaskLoaderSettings.for_pose(
        batch_size=16, workers=4, imgsz=640, seed=cfg['seed']), device=device)
    assert len(loader.dataset) == 5964 and len(loader.loader) == 373
    engine = MacroStepEngine(model=model, losses=bundle['criteria'], optimizer=bundle['optimizer'],
        reference_batch_size=64, task_weights={Task.POSE:1, Task.DETECT:0}, scaler=bundle['scaler'],
        ema=bundle['ema'], max_grad_norm=10, max_amp_retries=16, preprocess=lambda task, b: loader.preprocess(b))
    scheduler = bundle['scheduler']; start = 0
    # 每個 epoch 一個不可覆寫的續訓點；只從最高已保存邊界恢復。
    snapshots = sorted((out/'checkpoints').glob('epoch-*.pt'))
    if snapshots:
        restored = load_training_snapshot(snapshots[-1], model=model, **bundle)
        assert restored.resolved_config == cfg
        start = restored.progress.next_epoch
        guard.check(model); guard.check(bundle['ema'].ema)
    observed = []
    if mode == 'smoke':
        def observe(opt, args, kwargs):
            a = model.pose_head.p3_masf.alpha
            grads = [p.grad for n,p in model.pose_head.p3_masf.named_parameters() if n != 'alpha' and p.grad is not None]
            observed.append({'alpha':float(a.detach()), 'alpha_gradient':float(a.grad),
                'context_gradient_max':max(float(g.abs().max()) for g in grads)})
        hook = bundle['optimizer'].register_step_pre_hook(observe)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, 1 if mode == 'smoke' else 5):
        set_training(model)
        loader_seed = reseed_loader_for_epoch(loader.loader, seed=cfg['seed'], epoch=epoch, offset=1)
        reports = []; begin = time.monotonic()
        groups = batches_of(loader.loader, 8)
        if mode == 'smoke': groups = itertools.islice(groups, 2)
        for group in groups:
            scheduler.prepare_step()
            reports.append(asdict(engine.run(detect_batches=(), pose_batches=group)))
            scheduler.advance()
            # 只寫心跳；背景 monitor 每 600 秒才讀，不產生每步模型訊息。
            (out/'heartbeat').touch()
        guard.check(model); guard.check(bundle['ema'].ema)
        if mode == 'smoke':
            hook.remove()
            assert len(observed) == 2 and observed[0]['alpha'] == 0
            assert observed[0]['alpha_gradient'] != 0 and observed[0]['context_gradient_max'] == 0
            assert observed[1]['alpha'] != 0 and observed[1]['context_gradient_max'] > 0
            assert sum(r['pose_images'] for r in reports) == 256
            save(out/'summary.json', {'status':'completed', 'formal_epoch':False, 'observed':observed,
                'reports':reports, 'frozen_live_ema_exact':True, 'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            print('JOB_DONE smoke', flush=True); return
        assert len(reports) == 47 and sum(r['pose_images'] for r in reports) == 5964
        assert reports[-1]['pose_images'] == 76
        engine.advance_epoch((Task.POSE,))
        ep = out/f'epochs/e{epoch+1}'
        ep.mkdir(parents=True, exist_ok=True)
        # 在耗時驗證前先保存完整邊界；若驗證失敗，重試只補驗證，不重訓。
        checkpoint = out/f'checkpoints/epoch-{epoch+1:02d}.pt'
        assert not checkpoint.exists()
        saved = save_training_snapshot(checkpoint, model=model, **bundle,
            progress=TrainingProgress('pose_b', epoch+1, scheduler.current_step, epoch+1), resolved_config=cfg,
            provenance=source.provenance(), loader_state={'epoch_end':True,'loader_seed':loader_seed,'reports':reports}, best_state={})
        save_inference_weights(ep/'ema.pt', model=model, ema=bundle['ema'], metadata={
            'epoch':epoch,'parent_sha256':cfg['parent_inference_sha256'],'trained_pose_masf':True,'config':cfg,
            'full_resume_sha256':saved.sha256,'source_module':'training_b.TrainingSource'})
        save(ep/'training.json', {'epoch':epoch+1,'reports':reports,'train_seconds':time.monotonic()-begin,
            'alpha_live':float(model.pose_head.p3_masf.alpha.detach()),
            'alpha_ema':float(bundle['ema'].ema.pose_head.p3_masf.alpha),'frozen_live_ema_exact':True})
        metrics = validate(bundle['ema'].ema, source, ep/'validation', epoch, 'bittrue')
        save(ep/'metrics.json', metrics)
    # 恢復時：补齊已訓回合的 export／驗證（不重跑正常 epoch）。
    for epoch in range(5):
        ep = out/f'epochs/e{epoch+1}'; ep.mkdir(parents=True, exist_ok=True)
        snapshot = out/f'checkpoints/epoch-{epoch+1:02d}.pt'
        if not (ep/'ema.pt').exists():
            data = torch.load(snapshot, map_location='cpu', weights_only=True)
            model.load_state_dict(data['ema_state'], strict=True)
            save_inference_weights(ep/'ema.pt', model=model, use_ema=False, metadata={'epoch':epoch,
                'parent_sha256':cfg['parent_inference_sha256'], 'trained_pose_masf':True,'full_resume_sha256':sha256(snapshot)})
        if not (ep/'metrics.json').exists():
            data = torch.load(ep/'ema.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(data['state_dict'], strict=True)
            save(ep/'metrics.json', validate(model, source, ep/'validation-recovery', epoch, 'bittrue'))
    metrics_by_epoch = {str(i):json.loads((out/f'epochs/e{i}/metrics.json').read_text()) for i in range(1,6)}
    best = max(range(1,6), key=lambda i:metrics_by_epoch[str(i)]['bbat/pose/map50_95'])
    save(out/'summary.json', {'status':'completed','epochs':5,'metrics_by_epoch':metrics_by_epoch,
        'best_pose_epoch':best,'comparison_epoch':5,'auto_promoted':False,'A_canceled':True,
        'parent_sha256':cfg['parent_inference_sha256'],'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    print('JOB_DONE train_B', flush=True)

def analyze():
    initialize(); torch.set_num_threads(4)
    summary = json.loads((RUN/'summary.json').read_text()); assert summary['status'] == 'completed'
    output = RUN/'analysis'
    output.mkdir(parents=True, exist_ok=True)
    model, source = build()
    artifact = RUN/'epochs/e5/ema.pt'
    model.load_state_dict(torch.load(artifact,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    results = {}
    for case, kind in (('e5_float','float'), ('e5_alpha_off','bittrue')):
        result_path = output/(case+'.json')
        if result_path.exists(): results[case] = json.loads(result_path.read_text()); continue
        if case == 'e5_alpha_off':
            with torch.no_grad(): model.pose_head.p3_masf.alpha.zero_()
        results[case] = validate(model, source, output/case, 4, kind)
        save(result_path, results[case])
    base = parent.payload()['metadata']['metrics']; e5 = summary['metrics_by_epoch']['5']
    keys = [k for k in base if k.endswith('/map50_95')]
    lines = ['# B 組 Pose MASF 專項訓練結果', '',
        'B 組 5 epoch 已完成；A 組依使用者指示取消。沒有自動替換正式模型。', '',
        'E2→B5 同時包含額外訓練與 MASF 的效果，不能宣稱是 MASF 的獨立收益。B5 關閉 α 只測試已訓模型的分支依賴，不等於無 MASF 重新訓練對照。', '',
        '| AP50–95 (%) | 原 E2 | B5 | B5 關閉 MASF | B5−E2 (pp) |', '| --- | ---: | ---: | ---: | ---: |']
    for k in keys:
        lines.append(f"| {k} | {100*base[k]:.4f} | {100*e5[k]:.4f} | {100*results['e5_alpha_off'][k]:.4f} | {100*(e5[k]-base[k]):+.4f} |")
    lines += ['', '來源、超參數及架構見 README 與 execution-config.json；完整數據／續訓權重保存在 artifacts/b-e5-seed1-v1。',
              '', '此表為完整 COCO val 5000 與 canonical BBAT5 v1 val 683；未改 split。Float 與 BitTrue 分開保存。',
              '', '本次未重新量測推論延遲、目標硬體或能耗；相同結構的先前 benchmark 僅可作參考。']
    # 報告檔案編輯一律透過 apply_patch。
    sys.path.insert(0,str(HERE.parents[1]))
    from tools.restructure_layout import edit
    edit(HERE/'RESULTS.md','\n'.join(lines))
    save(output/'summary.json', {'status':'completed','results':results,'e5':e5,'baseline':base,
        'checkpoint':str(artifact),'sha256':sha256(artifact),'A_canceled':True,'auto_promoted':False})
    print('ALL_DONE B 組與 alpha-off 分析完成',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['smoke','train','analyze'])
    mode=parser.parse_args().mode
    if mode == 'analyze': analyze()
    else: run(mode)
