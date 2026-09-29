"""融合前獨立研究的唯讀來源與輸出界線。"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SOURCE = Path('/home/uxin/yolo/yolo_achitechure/achitechure_1/final')
sys.path.insert(0, str(WORKSPACE / 'src'))
sys.path.insert(0, str(SOURCE / 'code'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('產物不可離開本研究 artifacts')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def registry():
    return next(x for x in json.loads((SOURCE / 'models.json').read_text())['models']
                if x['id'] == 'full35-b-f10')


def setup():
    from yolo_optimize.data_safety import install_readonly_data_guard
    install_readonly_data_guard()
    import importlib.util
    path = Path('/home/uxin/yolo/yolo_combine/final/full35/code/project/src/yolo_combine/xnor.py')
    spec = importlib.util.spec_from_file_location('prefusion_exact_xnor', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.install_xnor_backend(module.XNORExecutionConfig(token_tile=32))


def prepare_coco():
    import yaml
    source = Path('/home/uxin/yolo/coco2017')
    root = ROOT / 'artifacts/datasets/coco2017-runtime'
    root.mkdir(parents=True, exist_ok=True)
    lineage = {}
    for split, count in [('train2017', 118287), ('val2017', 5000)]:
        contents = (source / f'{split}.txt').read_bytes()
        entries = contents.decode().splitlines()
        if len(entries) != count or any(not x.startswith(f'./images/{split}/') or '..' in Path(x).parts for x in entries):
            raise ValueError('COCO 原始清單不符；禁止重切或略過')
        for kind in ('images', 'labels'):
            link = root / kind / split
            target = source / kind / split
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != target.resolve():
                    raise ValueError(f'來源不符：{link}')
            else:
                link.symlink_to(target, target_is_directory=True)
        destination = root / f'{split}.txt'
        if destination.exists() and destination.read_bytes() != contents:
            raise ValueError('既有 runtime 清單不符')
        if not destination.exists():
            destination.write_bytes(contents)
        lineage[split] = {'count': count, 'sha256': sha256(destination)}
    annotations = root / 'annotations'
    if not annotations.exists():
        annotations.symlink_to(source / 'annotations', target_is_directory=True)
    if annotations.resolve() != (source / 'annotations').resolve():
        raise ValueError('annotations 來源不符')
    payload = yaml.safe_load(Path('/home/uxin/yolo/coco2017.yaml').read_text())
    payload['path'] = str(root)
    rendered = yaml.safe_dump(payload, sort_keys=False)
    destination = root / 'coco2017.yaml'
    if destination.exists() and destination.read_text() != rendered:
        raise ValueError('runtime YAML 不符')
    if not destination.exists():
        destination.write_text(rendered)
    write_json(root / 'lineage.json', lineage)
    return destination
