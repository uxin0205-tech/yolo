"""一次性實體目錄遷移；只改路徑文字，保留原文快照及所有模型 bytes。"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'archives').is_dir() and (p/'reports').is_dir())
AUDIT = ROOT/'docs/history/layout-v2-20260912'
MOVES = {
    'experiments/scripts/maintenance': 'tools',
    **{x: 'experiments/'+x for x in ('artifacts','studies','combine','activation','kd','inference','src','tests','scripts')},
    'optimizations': 'proposals',
    'reports/consolidated-20260911': 'reports/final',
    'reports/5090-done-0912': 'reports/publication',
    'reports/direction1-20260910': 'reports/direction1',
}
LINK = re.compile(r'(?<!!)\[([^\]\n]+)\]\((<[^>]+>|[^)\n]+)\)')
SKIP = {'archives','history','__pycache__','.pytest_cache','.git','.venv','node_modules'}
TEXT = {'.md','.py','.json','.csv','.yaml','.yml','.sh','.toml'}

def mapped(rel):
    for old,new in MOVES.items():
        if rel==old or rel.startswith(old+'/'):
            return new+rel[len(old):]
    return rel

def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def edit(p,text):
    text=text.rstrip('\n')+'\n'
    if p.exists():
        old=p.read_text()
        patch='*** Begin Patch\n*** Update File: '+str(p)+'\n@@\n'
        patch+=''.join('-'+s+'\n' for s in old.splitlines())
    else:
        patch='*** Begin Patch\n*** Add File: '+str(p)+'\n'
    patch+=''.join('+'+s+'\n' for s in text.splitlines())+'*** End Patch\n'
    subprocess.run(['apply_patch'],input=patch,text=True,check=True,stdout=subprocess.DEVNULL)

def absolute_paths(text):
    # 一次替換，避免 scripts/maintenance 被 scripts 第二次處理。
    prefixes='|'.join(re.escape(str(ROOT/old)) for old in MOVES)
    pattern=re.compile('('+prefixes+r')(?=/|[\s\x22\x27<>),;`]|$)')
    return pattern.sub(lambda m: str(ROOT/mapped(str(Path(m[0]).relative_to(ROOT)))),text)

def main():
    assert not AUDIT.exists(), '遷移已開始或完成；禁止覆寫，應依紀錄恢復'
    for old,new in MOVES.items():
        assert (ROOT/old).is_dir() and not (ROOT/new).exists(),(old,new)
    files={};links=[]
    for folder,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=[d for d in dirs if d not in SKIP]
        for name in [*dirs,*names]:
            p=Path(folder)/name
            if p.is_symlink():
                rel=str(p.relative_to(ROOT));target=os.readlink(p)
                resolved=Path(os.path.normpath(p.parent/target))
                newtarget=ROOT/mapped(str(resolved.relative_to(ROOT))) if resolved.is_relative_to(ROOT) else resolved
                newp=ROOT/mapped(rel)
                rendered=str(newtarget) if Path(target).is_absolute() else os.path.relpath(newtarget,newp.parent)
                links.append(dict(old=rel,new=mapped(rel),target=target,new_target=rendered,exists=p.exists()))
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in names:
            p=Path(folder)/name
            if p.is_symlink():continue
            st=p.stat();rel=str(p.relative_to(ROOT))
            files[rel]=dict(new=mapped(rel),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
    AUDIT.mkdir(parents=True)
    edit(AUDIT/'before.json',json.dumps(files,ensure_ascii=False,indent=2))
    changes=[]
    for rel,meta in files.items():
        p=ROOT/rel
        if p.suffix not in TEXT or meta['bytes']>20_000_000:continue
        old=p.read_text();new=old
        if p.suffix=='.md':
            dst=ROOT/meta['new']
            def rewrite(m):
                label,target=m.groups();target=target.strip('<>')
                if '://' in target or target.startswith('#'):return m[0]
                path,sep,anchor=target.partition('#')
                line=re.search(r':\d+$',path);suffix=line[0] if line else ''
                path=path[:-len(suffix)] if suffix else path
                local=Path(os.path.normpath(p.parent/unquote(path)))
                if local.is_relative_to(ROOT):local=ROOT/mapped(str(local.relative_to(ROOT)))
                target=os.path.relpath(local,dst.parent)+suffix+(('#'+anchor) if sep else '')
                return '['+label+'](<'+target+'>)'
            new=LINK.sub(rewrite,new)
        new=absolute_paths(new)
        # 對執行中的相對路徑，只修正根語意確實改變的程式。
        if rel in ('experiments/scripts/audit_accuracy_regressions.py','experiments/scripts/audit_repconv_seams.py'):
            new=new.replace('YOLO_ROOT = Path(__file__).resolve().parents[2]','YOLO_ROOT = Path(__file__).resolve().parents[3]')
        if rel=='experiments/scripts/build_round1_evidence.py':
            new=new.replace("ROOT/'proposals/integrated-roadmap/results'","ROOT.parent/'proposals/integrated-roadmap/results'")
        if rel.startswith('tools/') and not rel.endswith('restructure_layout.py'):
            new=new.replace('ROOT = Path(__file__).resolve().parents[2]','ROOT = Path(__file__).resolve().parents[1]')
        if rel.startswith('tools/') or rel=='reports/final/build_indexes.py':
            # 這些工具的 ROOT 仍是 optimize，不是 experiments。
            for oldprefix,newprefix in MOVES.items():
                new=re.sub(r"(['\"])"+re.escape(oldprefix)+r"(?=/)",lambda m: m[1]+newprefix,new)
        if p.suffix=='.py':ast.parse(new,filename=meta['new'])
        if p.suffix=='.json' and new!=old:
            assert json.loads(new)==json.loads(absolute_paths(old)),rel
        if new!=old:
            backup=AUDIT/'originals'/(rel+'.snapshot')
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
            changes.append(dict(old=rel,new=meta['new'],backup=str(backup.relative_to(ROOT)),
                before_sha256=sha(p),kind='path_text_only'))
            edit(p,new)
    # 先移出維護工具，再搬整個研究 scripts；同磁碟 rename 不複製 100 GB 權重。
    for old,new in MOVES.items():
        dst=ROOT/new;dst.parent.mkdir(parents=True,exist_ok=True)
        (ROOT/old).rename(dst)
    adjusted_links=0
    for row in links:
        p=ROOT/row['new']
        if row['target']!=row['new_target']:
            p.unlink();p.symlink_to(row['new_target']);adjusted_links+=1
        if row['exists']:assert p.exists(),row
    changed={r['old'] for r in changes}
    for rel,meta in files.items():
        p=ROOT/meta['new'];assert p.is_file(),meta['new']
        if rel not in changed:
            st=p.stat()
            assert (st.st_size,st.st_mtime_ns,st.st_ino)==(meta['bytes'],meta['mtime_ns'],meta['inode']),rel
    for row in changes:
        row['after_sha256']=sha(ROOT/row['new'])
        assert sha(ROOT/row['backup'])==row['before_sha256']
    result=dict(status='moved',directory_moves=MOVES,files=len(files),text_path_edits=len(changes),
        unchanged_files=len(files)-len(changes),symlinks=len(links),symlinks_retargeted=adjusted_links,
        changes=changes,links=links,no_weight_content_changes=True,no_gpu=True,no_data_deletions=True,
        archived_packages_untouched=True)
    edit(AUDIT/'manifest.json',json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {'changes','links'}},ensure_ascii=False))

if __name__=='__main__':main()
