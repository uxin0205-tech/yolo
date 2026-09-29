"""準備 Pose MASF 圖／分析的限範圍 Git 副本；不啟動 GPU、不 commit／push。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from restructure_layout import edit
ROOT=Path(__file__).resolve().parents[1]
FILES=['README.md','experiments/README.md','experiments/attention_recovery_v1/README.md',
 'experiments/attention_recovery_v1/modeling.py','docs/worklogs/README.md',
 'docs/worklogs/2026-09-13-attention-epoch-pause.md','docs/worklogs/2026-09-13-pose-masf-priority.md',
 'docs/worklogs/2026-09-13-pose-masf-training-derivation.md','docs/worklogs/2026-09-13-pose-masf-b-publication.md',
 'tools/publish_pose_0913.py']
TREES=['experiments/pose_masf_priority_v1','experiments/pose_masf_training_v1']
LINK=re.compile(r'(!?\[[^\]\n]+\])\((<[^>]+>|[^)\n]+)\)')
ALLOWED={'.md','.py','.json','.csv','.dot','.svg','.png'}

def selected():
    sources={ROOT/p for p in FILES}
    for folder in TREES:
        for p in (ROOT/folder).rglob('*'):
            rel=p.relative_to(ROOT/folder)
            if any(x in rel.parts for x in ('datasets','__pycache__','queue-v1','queue-b-v1')):continue
            if p.is_file() and not p.is_symlink() and p.suffix in ALLOWED:sources.add(p)
    return sorted(sources)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('worktree',type=Path);args=parser.parse_args()
    wt=args.worktree.resolve();assert (wt/'.git').is_file()
    mapping={p:wt/'yolo_optimize'/p.relative_to(ROOT) for p in selected()}
    rewrites=[]
    for src,dst in mapping.items():
        assert src.is_file() and not src.is_symlink() and src.stat().st_size<5_000_000
        assert src.suffix in ALLOWED
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix=='.md':
            def rewrite(m):
                label,target=m.groups();target=target.strip('<>')
                if '://' in target or target.startswith('#'):return m[0]
                path,sep,anchor=target.partition('#');path=re.sub(r':\d+$','',unquote(path))
                local=Path(os.path.normpath(src.parent/path))
                pub=wt/local.relative_to(ROOT.parent) if local.is_relative_to(ROOT.parent) else None
                if pub is not None and (local in mapping or pub.exists()):
                    return label+'(<'+os.path.relpath(pub,dst.parent)+(('#'+anchor) if sep else '')+'>)'
                assert not label.startswith('!'),'必要圖未納入：'+str(local)
                rewrites.append({'file':str(src.relative_to(ROOT)),'local_only':target})
                return label.strip('[]')+'（本機保存：`'+target+'`；本次未上傳）'
            edit(dst,LINK.sub(rewrite,src.read_text()))
        else:shutil.copy2(src,dst)
    counts=dict(markdown=0,links=0,python=0,json=0,svg=0,png=0)
    for src,dst in mapping.items():
        ext=dst.suffix
        if ext=='.py':ast.parse(dst.read_text());counts['python']+=1
        if ext=='.json':
            def invalid(value):raise ValueError('非有限 JSON 常數：'+value)
            assert json.loads(src.read_text(),parse_constant=invalid)==json.loads(dst.read_text(),parse_constant=invalid)
            counts['json']+=1
        if ext=='.svg':ET.parse(dst);counts['svg']+=1
        if ext=='.png':
            blob=dst.read_bytes();assert blob[:8]==b'\x89PNG\r\n\x1a\n'
            assert min(struct.unpack('>II',blob[16:24]))>0;counts['png']+=1
        if ext=='.md':
            counts['markdown']+=1
            for m in LINK.finditer(dst.read_text()):
                target=m[2].strip('<>')
                if '://' in target or target.startswith('#'):continue
                assert (dst.parent/unquote(target.split('#')[0])).exists(),(dst,target)
                counts['links']+=1
    result={'status':'passed','commit_subject':'5090 Done 0913','scope':'Pose MASF 圖／既有分析／CPU 通過的 B 程式；GPU 與 queue 未啟動',
       'counts':counts,'files':[{'path':str(d.relative_to(wt)),'bytes':d.stat().st_size,
       'sha256':hashlib.sha256(d.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(s.read_bytes()).hexdigest()}
       for s,d in mapping.items()],'local_only_links':rewrites,'no_weights_or_datasets':True}
    for base in (ROOT,wt/'yolo_optimize'):
        edit(base/'reports/current-model/publication-pose-0913.json',json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'files':len(mapping),'bytes':sum(d.stat().st_size for d in mapping.values()),'counts':counts,'local_only_links':len(rewrites)},ensure_ascii=False))

if __name__=='__main__':main()
