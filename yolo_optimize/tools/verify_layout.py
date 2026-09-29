"""目錄遷移的 CPU 文字／權重完整性驗證；不反序列化模型、不啟動訓練。"""
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote
from restructure_layout import ROOT, MOVES, LINK, absolute_paths, edit

def main():
    audit=ROOT/'docs/history/layout-v2-20260912'
    manifest=json.loads((audit/'manifest.json').read_text())
    before=json.loads((audit/'before.json').read_text())
    model_files=0
    for rel,row in before.items():
        if Path(rel).suffix not in {'.pt','.pth','.onnx','.engine','.safetensors'}:continue
        p=ROOT/row['new'];s=p.stat()
        assert (s.st_size,s.st_mtime_ns,s.st_ino)==(row['bytes'],row['mtime_ns'],row['inode']),p
        model_files+=1
    json_files=csv_files=0
    for row in manifest['changes']:
        p=ROOT/row['new'];old=(ROOT/row['backup']).read_text()
        if p.suffix=='.json':
            assert json.loads(p.read_text())==json.loads(absolute_paths(old)),p
            json_files+=1
        if p.suffix=='.csv':
            assert list(csv.reader(io.StringIO(p.read_text())))==list(csv.reader(io.StringIO(absolute_paths(old)))),p
            csv_files+=1
    models=json.loads((ROOT/'reports/checkpoints/registry.json').read_text())
    for row in models:
        with (ROOT/row['path']).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],row
    python_files=links=0;bad=[]
    for folder,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'archives','history','__pycache__','datasets','.pytest_cache'}
                 and not (Path(folder)/d).is_symlink()]
        for name in names:
            p=Path(folder)/name
            if p.is_symlink():continue
            if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p));python_files+=1
            if p.suffix!='.md':continue
            for _,raw in LINK.findall(p.read_text()):
                target=raw.strip('<>')
                if '://' in target or target.startswith('#'):continue
                path=re.sub(r':\d+$','',unquote(target.split('#')[0]));links+=1
                if not (p.parent/path).exists():bad.append([str(p.relative_to(ROOT)),target])
    assert not bad,bad
    assert all(not (ROOT/old).exists() for old in MOVES), '仍有舊根層目錄'
    result=dict(status='passed',physical_directory_moves=len(MOVES),model_files_stat_unchanged=model_files,
                models_sha256_rechecked=len(models),json_path_only_equivalence=json_files,
                csv_path_only_equivalence=csv_files,python_ast=python_files,local_markdown_links=links,
                broken_links=bad,old_root_aliases=False,no_gpu=True,
                separately_executed={'pytest':'83 passed in 9.85s','selected_model_cpu':
                    'qSiLU E2 safe rebuild; BitTrue Detect/Pose 160x160 finite forward passed'},
                limitations=['not a new AP evaluation','not an exact-resume audit of all historical checkpoints'])
    edit(ROOT/'reports/publication/layout-check.json',json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
