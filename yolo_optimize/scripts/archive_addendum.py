"""為完成後報告／推論追加獨立封存，不修改第一包。"""
import json
from pathlib import Path
import subprocess
from archive_research import ROOT, BASE, walk, digest


def main():
    out = ROOT / 'archives/research-through-20260911-addendum-v1'
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for rel in ('reports/consolidated-20260911', 'inference/routing_v1', 'docs/worklogs'):
        for x in walk(ROOT / rel):
            sources[x] = Path('snapshot/optimize') / x.relative_to(ROOT)
    for rel in ('README.md', 'scripts/archive_research.py', 'scripts/archive_addendum.py',
                'combine/bridge_v1/plan.json', 'combine/bridge_v1/README.md',
                'kd/pose_focus_v1/README.md', 'kd/dual_task_v1/README.md',
                'reports/direction1-20260910/README.md', 'studies/pre-fusion-full35-b100/README.md'):
        sources[ROOT/rel] = Path('snapshot/optimize') / rel
    # 本機 Ultralytics 有客製修改，pip 版本號不足以重建；保存實際原始碼。
    package = BASE / 'yolo_combine/.venv/lib/python3.12/site-packages/ultralytics'
    for x in walk(package, code_only=True):
        sources[x] = Path('snapshot/installed-ultralytics') / x.relative_to(package)
    for x in walk(BASE / 'docs/worklogs', code_only=True):
        sources[x] = Path('snapshot/dependencies/docs/worklogs') / x.relative_to(BASE/'docs/worklogs')
    for x in walk(BASE / 'docs/agents', code_only=True):
        sources[x] = Path('snapshot/dependencies/docs/agents') / x.relative_to(BASE/'docs/agents')
    events = ROOT/'combine/artifacts/logs/pose-routing-one2many-v1.events.jsonl'
    log = events.with_name('pose-routing-one2many-v1.log')
    for x in (events, log):
        if x.exists():
            sources[x] = Path('snapshot/optimize') / x.relative_to(ROOT)
    rows = []
    for src, rel in sorted(sources.items()):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists()
        before = src.stat()
        sha = digest(src)
        subprocess.run(['cp', '--reflink=auto', '--preserve=timestamps', '--', str(src), str(dst)], check=True)
        assert digest(dst) == sha
        after = src.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        rows.append(dict(source=str(src), archive=str(rel), bytes=before.st_size, sha256=sha, copy_verified=True))
    with (out/'manifest.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = dict(status='completed', files=len(rows), bytes=sum(x['bytes'] for x in rows),
                   all_copies_sha256_verified=True, purpose='final report, additional inference, actual installed Ultralytics source')
    with (out/'summary.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('ADDENDUM_DONE '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
