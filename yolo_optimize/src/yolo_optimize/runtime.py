"""第一輪實驗的唯讀來源與獨立輸出邊界。"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
FINAL_ROOT = Path('/home/uxin/yolo/yolo_combine/final/full35')
sys.dont_write_bytecode = True
sys.path.insert(0, str(FINAL_ROOT / 'code/project/src'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def output_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(WORKSPACE) or resolved == WORKSPACE:
        raise ValueError(f'輸出必須位於優化專案的子目錄：{resolved}')
    return resolved


def write_json(path: Path, payload: object) -> None:
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def load_config(run_root: str | Path):
    from .data_safety import install_readonly_data_guard
    install_readonly_data_guard()
    from yolo_combine.joint_config import JointExperimentConfig

    config = JointExperimentConfig.load(FINAL_ROOT / 'configs/joint.yaml')
    return replace(config, run_root=output_path(run_root), warmup_epochs=1)


def prepare_data(config, root: str | Path) -> tuple[Path, Path]:
    """保留完整影像清單；僅將 labels.cache 放入 runtime View。"""
    import yaml
    from yolo_combine.data import prepare_bbt5_view

    root = output_path(root)
    for parent in (root / 'bbat5-v1-runtime', root / 'coco2017-runtime',
                   root / 'coco2017-runtime/images', root / 'coco2017-runtime/labels'):
        if parent.is_symlink():
            raise ValueError(f'runtime View 的父目錄不得是 symlink：{parent}')
    pose = prepare_bbt5_view(config.registry, root / 'bbat5-v1-runtime')
    coco = root / 'coco2017-runtime'
    coco.mkdir(parents=True, exist_ok=True)
    # 原 YAML 的 path 是相對 datasets_dir；此專案的固定來源已核對如下。
    source = Path('/home/uxin/yolo/coco2017')
    payload = yaml.safe_load(config.detect_data.read_text(encoding='utf-8'))
    if (payload.get('train'), payload.get('val')) != ('train2017.txt', 'val2017.txt'):
        raise ValueError('COCO 清單設定已改變，須重新稽核，禁止猜測來源')
    lineage = {'source_yaml': str(config.detect_data), 'source_yaml_sha256': sha256(config.detect_data), 'splits': {}}
    for split in ('train2017', 'val2017'):
        for kind in ('images', 'labels'):
            target = source / kind / split
            if not target.is_dir():
                raise FileNotFoundError(target)
            link = coco / kind / split
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != target.resolve():
                    raise ValueError(f'既有 runtime View 不符合來源：{link}')
            else:
                link.symlink_to(target, target_is_directory=True)
        original = source / f'{split}.txt'
        contents = original.read_bytes()
        entries = contents.decode('utf-8').splitlines()
        if not entries or any(not line.startswith(f'./images/{split}/') or '..' in Path(line).parts for line in entries):
            raise ValueError(f'COCO 清單格式不符合核對內容：{original}')
        destination = coco / original.name
        if destination.exists() and destination.read_bytes() != contents:
            raise ValueError(f'既有 COCO runtime 清單不符：{destination}')
        if not destination.exists():
            destination.write_bytes(contents)
        lineage['splits'][split] = {'count': len(entries), 'list_sha256': sha256(original)}
    # 官方 validator 會對 COCO 自動附加 JSON 評估；原始 annotations 只供讀取。
    annotation_source = source / 'annotations'
    annotation_view = coco / 'annotations'
    if not (annotation_source / 'instances_val2017.json').is_file():
        raise FileNotFoundError(annotation_source / 'instances_val2017.json')
    if annotation_view.exists() or annotation_view.is_symlink():
        if not annotation_view.is_symlink() or annotation_view.resolve() != annotation_source.resolve():
            raise ValueError('既有 COCO annotation View 不符')
    else:
        annotation_view.symlink_to(annotation_source, target_is_directory=True)
    payload['path'] = str(coco)
    detect_yaml = coco / 'coco2017.yaml'
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    if detect_yaml.exists() and detect_yaml.read_text(encoding='utf-8') != rendered:
        raise ValueError('既有 COCO runtime YAML 不符')
    if not detect_yaml.exists():
        detect_yaml.write_text(rendered, encoding='utf-8')
    write_json(coco / 'lineage.json', lineage)
    return detect_yaml, pose.yaml


def load_model(config, checkpoint: str | Path, device):
    from yolo_combine.factory import FusionModelFactory
    from yolo_combine.inference import load_combined_weights
    from yolo_combine.source import SourceBundle
    from yolo_combine.xnor import XNORExecutionConfig

    source = SourceBundle(config.source_bundle, architecture=config.architecture)
    built = FusionModelFactory(
        source, detect_data_yaml=config.detect_data, pose_data_yaml=config.pose_data,
        xnor=XNORExecutionConfig(token_tile=config.xnor_token_tile),
    ).build(checkpoint_kind='float', allow_untrained_pose_head=True)
    model = built.model.to(device)
    loaded = load_combined_weights(model, checkpoint, prefer_ema=True)
    if loaded.state_source != 'ema':
        raise ValueError(f'本轮要求 EMA 起點，但權重來源是 {loaded.state_source}')
    return source, model, built.report, loaded
