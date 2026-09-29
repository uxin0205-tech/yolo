"""只準備 0913 報告增量、來源清單與驗證；不 commit、不 push、不刪資料。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from restructure_layout import edit

ROOT=Path(__file__).resolve().parents[1]
FILES='''README.md
reports/README.md
reports/final/README.md
reports/performance/PUBLISHED-4f4131e.md
experiments/README.md
experiments/inference/README.md
experiments/attention_recovery_v1/README.md
docs/research/README.md
docs/research/2026-09-13-attention-scale-bias-alternatives.md
docs/worklogs/README.md
docs/worklogs/2026-09-12-performance-publish-confirmed.md
docs/worklogs/2026-09-13-current-model-attention-recovery.md
docs/worklogs/2026-09-13-report-update-0913.md
tools/publish_update_0913.py'''.splitlines()
TREES=('reports/current-model','experiments/inference/component_switch_v1')
LINK=re.compile(r'(!?\[[^\]\n]+\])\((<[^>]+>|[^)\n]+)\)')


def selected():
    result={ROOT/p for p in FILES}
    for folder in TREES:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
                if p.suffix in {'.md','.py','.json','.svg'} or p.name=='STOP_AFTER_CURRENT':
                    result.add(p)
    return sorted(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('worktree',type=Path);args=parser.parse_args()
    wt=args.worktree.resolve();assert (wt/'.git').is_file()
    sources=selected();mapped={p:wt/'yolo_optimize'/p.relative_to(ROOT) for p in sources}
    rewrites=[]
    for src,dst in mapped.items():
        assert src.is_file() and not src.is_symlink()
        assert src.stat().st_size<2_000_000 and src.suffix not in {'.pt','.pth','.onnx','.zip'}
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix=='.md':
            def rewrite(m):
                label,target=m.groups();target=target.strip('<>')
                if '://' in target or target.startswith('#'):return m[0]
                path,sep,anchor=target.partition('#');path=re.sub(r':\d+$','',unquote(path))
                local=Path(os.path.normpath(src.parent/path))
                pub=wt/local.relative_to(ROOT.parent) if local.is_relative_to(ROOT.parent) else None
                if pub is not None and (local in mapped or pub.exists()):
                    return label+'(<'+os.path.relpath(pub,dst.parent)+(('#'+anchor) if sep else '')+'>)'
                assert not label.startswith('!'),'圖不可被排除：'+str(local)
                rewrites.append({'file':str(src.relative_to(ROOT)),'local_only':target})
                return label.strip('[]')+'（本機檔案：`'+target+'`；本次未上傳）'
            edit(dst,LINK.sub(rewrite,src.read_text()))
        else:shutil.copy2(src,dst)
    counts={'markdown':0,'links':0,'json':0,'python':0,'svg':0}
    for src,dst in mapped.items():
        if dst.suffix=='.py':ast.parse(dst.read_text());counts['python']+=1
        if dst.suffix=='.json':
            assert json.loads(src.read_text())==json.loads(dst.read_text());counts['json']+=1
        if dst.suffix=='.svg':ET.parse(dst);counts['svg']+=1
        if dst.suffix=='.md':
            counts['markdown']+=1
            for m in LINK.finditer(dst.read_text()):
                target=m[2].strip('<>')
                if '://' in target or target.startswith('#'):continue
                assert (dst.parent/unquote(target.split('#')[0])).exists(),(dst,target)
                counts['links']+=1
    report={'status':'passed','scope':'0913 報告、診斷證據與程式；無權重／資料集',
        'counts':counts,'files':[{'path':str(d.relative_to(wt)),'bytes':d.stat().st_size,
        'sha256':hashlib.sha256(d.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(s.read_bytes()).hexdigest()} for s,d in mapped.items()],
        'local_only_link_rewrites':rewrites}
    for p in (ROOT/'reports/current-model/publication-check-0913.json',wt/'yolo_optimize/reports/current-model/publication-check-0913.json'):
        assert not p.exists(),'保留既有發布稽核，禁止覆寫'
        edit(p,json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'files':len(mapped),'bytes':sum(d.stat().st_size for d in mapped.values()),
        'checks':counts,'local_only_links':len(rewrites)},ensure_ascii=False))


if __name__=='__main__':main()
