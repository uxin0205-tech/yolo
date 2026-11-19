"""執行單一階段，保存可續訓 checkpoint 與實際配置。"""
import json
import shutil
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer

ROOT = Path(__file__).resolve().parents[1]

class StagePoseTrainer(PoseTrainer):
    """明確沿用本階段 YAML 的 epochs，支援使用者授權的續訓延長。"""
    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume and 'epochs' in overrides:
            requested = int(overrides['epochs'])
            if requested < self.args.epochs:
                raise ValueError('續訓不可縮短 checkpoint 的原始 epoch 上限')
            self.args.epochs = requested


def preserve_resume(trainer):
    if trainer.last.is_file():
        target = trainer.last.with_name('last-resumable.pt')
        temp = target.with_suffix('.tmp')
        shutil.copy2(trainer.last, temp)
        temp.replace(target)


def main():
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    cfg['batch'] = int(sys.argv[2]) if len(sys.argv) > 2 else cfg['batch']
    run = ROOT / 'runs' / cfg['name']
    checkpoint = run / 'weights/last-resumable.pt'
    source = cfg.pop('model')
    if checkpoint.is_file():
        source = str(checkpoint)
        cfg['resume'] = str(checkpoint)
    cfg['exist_ok'] = True
    model = YOLO(source)
    # 固定使用者指定 batch；OOM 時停止，避免官方自動降低 batch。
    model.add_callback('on_train_epoch_start', lambda trainer: setattr(trainer, '_oom_retries', 3))
    def record_batch(trainer):
        (run / 'batch-policy.json').write_text(json.dumps({
            'requested_batch':cfg['batch'], 'actual_batch':trainer.batch_size,
            'policy':'fixed', 'resume':str(checkpoint) if cfg.get('resume') else None
        }, indent=2))
    model.add_callback('on_pretrain_routine_end', record_batch)
    model.add_callback('on_model_save', preserve_resume)
    model.train(trainer=StagePoseTrainer, **cfg)
    run = ROOT / 'runs' / cfg['name']
    assert (run / 'weights/best.pt').is_file()
    assert (run / 'weights/last.pt').is_file()
    assert (run / 'weights/last-resumable.pt').is_file()
    (run / 'completed.json').write_text(json.dumps({'status':'completed','name':cfg['name']},indent=2))

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or any(text in str(exc) for text in (
            'CUDA out of memory', 'CUDNN_STATUS_INTERNAL_ERROR', 'unable to find an engine'
        )):
            print(f'顯存不足，保留進度並停止：{exc}', flush=True)
            sys.exit(42)
        raise
