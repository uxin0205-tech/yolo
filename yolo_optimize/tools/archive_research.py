"""只新增封存：保存研究檔案並逐檔驗證，不載入 checkpoint 或刪除來源。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
SKIP = {'__pycache__', '.git', '.pytest_cache', '.venv', 'node_modules', 'archives'}
WEIGHTS = {'.pt', '.pth', '.onnx', '.engine', '.safetensors'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def walk(root, code_only=False):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and
                         not (Path(directory) / d).is_symlink() and
                         (not code_only or d not in {'artifacts', 'weights', 'runs', 'dataset', 'original'}))
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in {'.pyc', '.cache'}:
                continue
            # runtime images/labels 不複製；其設定與 manifest 仍保存。
            if 'datasets' in path.relative_to(root).parts and path.suffix not in {'.json', '.yaml', '.yml', '.md', '.csv'}:
                continue
            if code_only and path.suffix not in {'.py', '.yaml', '.yml', '.json', '.toml', '.txt', '.md', '.sh'}:
                continue
            if path.is_file() and not path.is_symlink():
                yield path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    args = p.parse_args()
    assert args.name.replace('-', '').isalnum()
    out = ROOT / 'archives' / args.name
    out.mkdir(parents=True, exist_ok=False)
    sources = {x: Path('snapshot/optimize') / x.relative_to(ROOT) for x in walk(ROOT)}
    for rel in ('yolo_combine/src', 'yolo_combine/configs', 'yolo_attention_final/src',
                'yolo_attention_final/configs', 'yolo_activation/src',
                'yolo_achitechure/achitechure_1/final/code',
                'yolo_achitechure/achitechure_1/final/configs'):
        for x in walk(BASE / rel, code_only=True):
            sources[x] = Path('snapshot/dependencies') / x.relative_to(BASE)
    for rel in ('yolo_achitechure/achitechure_1/final', 'yolo_combine/final/full35',
                'yolo_attention_final/final'):
        folder = BASE / rel
        if folder.exists():
            for x in walk(folder):
                sources[x] = Path('snapshot/dependencies') / x.relative_to(BASE)
    pose = BASE / 'yolo_combine/variants/full35/artifacts/pose/p0-full35-p3-b32a4-e100max-seed0/weights/best.pt'
    sources[pose] = Path('snapshot/dependencies') / pose.relative_to(BASE)
    for rel in ('configs/datasets/bbat5-v1.yaml', 'coco2017.yaml', 'AGENTS.md', 'CONTEXT.md',
                'original/pose/derived/bbat5-v1/configs/pose.yaml',
                'original/pose/derived/bbat5-v1/configs/detect.yaml'):
        x = BASE / rel
        sources[x] = Path('snapshot/dependencies') / rel
    total = sum(x.stat().st_size for x in sources)
    assert shutil.disk_usage(out).free > total + 10 * 1024**3
    print(f'ARCHIVE_STARTED files={len(sources)} bytes={total}', flush=True)
    records, checkpoints, summaries, metrics, anomalies = [], [], [], [], []
    with (out / 'manifest.jsonl').open('x') as mf:
        for number, (src, rel) in enumerate(sorted(sources.items()), 1):
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            assert not dst.exists()
            before = src.stat()
            sha = digest(src)
            subprocess.run(['cp', '--reflink=auto', '--preserve=timestamps', '--', str(src), str(dst)], check=True)
            copied_sha = digest(dst)
            after = src.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(src)
            assert sha == copied_sha, str(src)
            row = dict(source=str(src), archive=str(rel), bytes=before.st_size, sha256=sha,
                       mtime_ns=before.st_mtime_ns, copy_verified=True)
            mf.write(json.dumps(row, ensure_ascii=False) + '\n')
            mf.flush()
            records.append(row)
            if src.suffix.lower() in WEIGHTS:
                checkpoints.append({**row, 'role_hint': 'resume_candidate' if 'checkpoints' in src.parts or 'resume' in src.name
                                    else 'inference_candidate' if 'inference' in src.parts else 'unclassified',
                                    'load_test': 'not_loaded_by_archive; byte-integrity-only'})
            if src.suffix == '.json' and src.is_relative_to(ROOT):
                try:
                    data = json.loads(dst.read_text())
                except (ValueError, UnicodeError) as exc:
                    anomalies.append(dict(path=str(src), issue=str(exc)))
                    continue
                if src.name in {'summary.json', 'safety-stop.json', 'pose-stop.json', 'recovery-stop.json'}:
                    summaries.append(dict(path=str(src), status=data.get('status', 'unspecified') if isinstance(data, dict) else 'array',
                                          archive=str(rel)))
                def collect(obj, ptr=''):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            here = ptr + '/' + k.replace('~', '~0').replace('/', '~1')
                            if isinstance(v, (int, float)) and ('map50_95' in k or k in {'overall_ap', 'person_ap'}):
                                metrics.append(dict(source=str(src.relative_to(ROOT)), pointer=here, metric=k, value=v))
                                if not math.isfinite(v):
                                    anomalies.append(dict(path=str(src), pointer=here, issue='nonfinite metric'))
                            else:
                                collect(v, here)
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            collect(v, ptr + '/' + str(i))
                collect(data)
            if number % 1000 == 0:
                print(f'ARCHIVE_PROGRESS verified={number}/{len(sources)}', flush=True)
    for name, data in [('checkpoints.json', checkpoints), ('run-index.json', summaries), ('audit-anomalies.json', anomalies)]:
        with (out / name).open('x') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    with (out / 'metrics-long.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'pointer', 'metric', 'value'])
        writer.writeheader()
        writer.writerows(metrics)
    env = subprocess.run(['/home/uxin/yolo/yolo_combine/.venv/bin/python', '-m', 'pip', 'freeze'],
                         capture_output=True, text=True, check=True).stdout
    with (out / 'requirements-freeze.txt').open('x') as f:
        f.write(env)
    summary = dict(status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
                   files=len(records), bytes=total, checkpoints=len(checkpoints),
                   checkpoint_bytes=sum(x['bytes'] for x in checkpoints), metric_records=len(metrics),
                   run_summaries=len(summaries), anomalies=len(anomalies),
                   checkpoint_loading_performed=False, all_copies_sha256_verified=True,
                   exclusions=['archives', 'cache', 'runtime images/labels', 'symlinks and symlink directories'],
                   limitation='same-disk snapshot, not off-device backup; datasets and Python environment remain external')
    with (out / 'summary.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('ARCHIVE_DONE ' + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
