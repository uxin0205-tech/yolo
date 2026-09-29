"""建立限定報告發布樹：不碰來源權重、不改實驗數字、不執行 Git push。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
ALLOWED = {'.md', '.csv', '.json', '.yaml', '.yml', '.py', '.toml', '.sh', '.svg'}
SKIP = {'archives', '__pycache__', '.git', '.venv', '.pytest_cache', 'datasets', 'node_modules', 'originals'}
LINK = re.compile(r'(?<!!)\[([^\]\n]+)\]\((<[^>]+>|[^)\n]+)\)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worktree', type=Path, required=True)
    args = parser.parse_args()
    wt = args.worktree.resolve()
    assert (wt/'.git').is_file() and not (wt/'yolo_optimize').exists()
    files = {}
    excluded = []
    for folder, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(folder)/d).is_symlink())
        for name in sorted(names):
            src = Path(folder)/name
            if src.name in {'publication-manifest.json', 'publication-check.json', 'PUBLISHED.md'}:
                continue
            if src.is_symlink() or src.suffix not in ALLOWED:
                continue
            if src.stat().st_size > 2_000_000:
                excluded.append(str(src.relative_to(ROOT)))
                continue
            files[src] = wt/'yolo_optimize'/src.relative_to(ROOT)
    # 上層 MASＦ 稽核是總報告直接引用的依據；不覆蓋遠端其他人的既有文件。
    external = BASE/'docs/worklogs/2026-09-10-masf-results-audit.md'
    if external.exists() and not (wt/external.relative_to(BASE)).exists():
        files[external] = wt/external.relative_to(BASE)
    for name in ('audit.json', 'paired-deltas.csv', 'recalculate.py'):
        attachment = BASE/'docs/worklogs/assets/2026-09-10-masf-results-audit'/name
        if attachment.exists() and not (wt/attachment.relative_to(BASE)).exists():
            files[attachment] = wt/attachment.relative_to(BASE)
    edits = []
    for src, dst in sorted(files.items()):
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists(), dst
        text = src.read_text()
        if src.suffix == '.md':
            def rewrite(match):
                label, raw = match.groups()
                target = raw.strip('<>')
                if '://' in target or target.startswith('#'):
                    return match.group(0)
                path, sep, anchor = target.partition('#')
                path = re.sub(r':\d+$', '', unquote(path))
                local = Path(os.path.normpath(src.parent/path))
                if local.is_relative_to(BASE):
                    published = wt/local.relative_to(BASE)
                    available = local in files or published.exists()
                else:
                    available = False
                if available:
                    relative = os.path.relpath(published, dst.parent)
                    result = '['+label+'](<'+relative+(('#'+anchor) if sep else '')+'>)'
                    if result != match.group(0):
                        edits.append(dict(file=str(dst.relative_to(wt)), target=target, action='relative_link'))
                    return result
                edits.append(dict(file=str(dst.relative_to(wt)), target=target, action='local_only_reference'))
                return label+'（本機／歷史參照：`'+target+'`；未隨本次報告發布）'
            text = LINK.sub(rewrite, text)
        # 格式化僅移除行尾空白；數值資料與Python內容逐字保留。
        if src.suffix == '.md':
            text = '\n'.join(line.rstrip() for line in text.splitlines())+'\n'
        with dst.open('x') as f:
            f.write(text)
    manifest = dict(scope='reports, numerical evidence, configs and research source; no weights/dataset/archive',
        file_count=len(files), bytes=sum(d.stat().st_size for d in files.values()),
        excluded_large_text=excluded, link_rewrites=edits,
        files=[dict(source=str(s.relative_to(BASE)), published=str(d.relative_to(wt)),
                    sha256=hashlib.sha256(d.read_bytes()).hexdigest()) for s,d in sorted(files.items())])
    path = wt/'yolo_optimize/reports/publication/publication-manifest.json'
    with path.open('x') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(files=len(files), bytes=manifest['bytes'], excluded_large_text=excluded,
                          local_only_links=sum(x['action']=='local_only_reference' for x in edits))))


if __name__ == '__main__':
    main()
