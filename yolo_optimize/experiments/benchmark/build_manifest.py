"""從既有正式指標選定各階段代表與配對，不重新計算 AP。"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREF=ROOT/'experiments/studies/pre-fusion-full35-b100/artifacts'
COM=ROOT/'experiments/combine/bridge_v1/artifacts'
ACT=ROOT/'experiments/activation/bridge_v1/artifacts'
KD=ROOT/'experiments/kd'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    rows=[]
    def add(id,stage,path,metrics,source,**kw):
        p=Path(path);assert p.is_file(),p
        rows.append(dict(id=id,stage=stage,checkpoint=str(p),sha256=sha(p),
            metrics={k:v for k,v in metrics.items() if k.endswith('map50_95')},
            metrics_source=str(source),metrics_source_sha256=sha(source),task='detect',loader='pickle',**kw))
    for key,label in [('fp','BinaryQK'),('a0','BinaryQK'),('bittrue-v2','BinaryQK')]:
        p=PREF/('baseline-'+key)/'summary.json';s=json.loads(p.read_text())
        add('fp' if key=='fp' else 'a0' if key=='a0' else 'b100',label,s['checkpoint'],s['metrics'],p,backend='float' if key=='fp' else 'bittrue')
    choices=[('hog_control_e3','HOG','prefusion-hog-control-v1',3),('hog_e3','HOG','prefusion-hog-hog-v1',3),
        ('rep_control_e4','RepConv','rep17-control-v1',4),('rep_folded_e4','RepConv','rep17-rep-v1',4),
        ('p3_control_e5','MASF','masf-p3-control-v1',5),('p3_shared_e5','MASF','masf-p3-shared-v1',5),
        ('p3_fork_e5','MASF','masf-p3-fork-v1',5),('p3_control_e8','MASF','masf-head-control-v1',8),
        ('p3_bridge_e8','MASF','masf-task-bridge-v1',8),('p2_control_e5','MASF','masf-p2-control-v1',5),
        ('p2_e5','MASF','masf-p2-p2-v1',5)]
    bbat=json.loads((ROOT/'experiments/combine/artifacts/existing-masf-bbat-v1/summary.json').read_text())
    for id,stage,run,epoch in choices:
        p=PREF/run/'summary.json';s=json.loads(p.read_text());record=next(e for e in s['epochs'] if e['epoch']==epoch)
        path=PREF/run/f'epoch-{epoch:02d}-resume.pt';ap=dict(record['ema'])
        for b in bbat['rows']:
            if Path(b['checkpoint'])==path:ap.update(b['bbat'])
        add(id,stage,path,ap,p,backend='bittrue',state='ema',epoch=epoch,
            checkpoint_is_training_snapshot=True,bbat_secondary_source=str(ROOT/'experiments/combine/artifacts/existing-masf-bbat-v1/summary.json'))
    joint=[('old_joint','融合',Path('/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt'),'old','silu',COM/'old-combine-comparison-v1/old-on-bittrue.json','metrics'),
      ('j3_joint','融合',COM/'fusion/balanced-j3-v1/inference/best_pose.pt','new','silu',COM/'old-combine-comparison-v1/new-on-bittrue.json','metrics'),
      ('pose_recovery','融合',COM/'fusion/j3-pose-head-recovery-v1/inference/best_pose.pt','new','silu',ACT/'zero-shot-v1/summary.json',None),
      ('silu_e9','Activation',ACT/'runs/silu-short-e10-seed1-v1/inference/best_joint.pt','new','silu',ACT/'runs/silu-short-e10-seed1-v1/summary.json',None),
      ('qsilu_e2','Activation',ACT/'runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt','new','qsilu_pq',KD/'dual_task_v1/artifacts/student-baseline-v1.json','metrics'),
      ('dual_kd_e4','KD',KD/'dual_task_v1/artifacts/runs/spatial-e5-seed1-v1/inference/best_pose.pt','new','qsilu_pq',KD/'dual_task_v1/artifacts/runs/spatial-e5-seed1-v1/summary.json',None),
      ('head_kd_e2','KD',KD/'pose_focus_v1/artifacts/runs/kd-e5-seed1-v1/inference/best_pose.pt','new','qsilu_pq',KD/'pose_focus_v1/artifacts/direct-kd-result-v1.json',None),
      ('mixed_keypoint','推論',ROOT/'experiments/inference/pose_branch_v1/artifacts/branch-v2/candidate.pt','new','qsilu_pq',ROOT/'experiments/inference/pose_branch_v1/artifacts/branch-v2/metrics.json',None)]
    for id,stage,path,template,activation,source,key in joint:
        s=json.loads(source.read_text())
        if key:metrics=s[key]
        elif id=='pose_recovery':metrics=s['results']['silu']['bittrue']
        elif id=='silu_e9':metrics=s['best_state']['best_joint']['metrics']
        elif id in ('dual_kd_e4','head_kd_e2'):metrics=s['best_state']['best_pose']['metrics']
        else:metrics=s['bittrue']
        add(id,stage,path,metrics,source,backend='bittrue',template=template,activation=activation)
        rows[-1].update(loader='state_dict',task='both')
    zero=ACT/'zero-shot-v1/summary.json';z=json.loads(zero.read_text())
    for arm in ['hardswish','poly_shift']:
        add(arm+'_zero','Activation',COM/'fusion/j3-pose-head-recovery-v1/inference/best_pose.pt',z['results'][arm]['bittrue'],zero,backend='bittrue',template='new',activation=arm,zero_shot=True)
        rows[-1].update(loader='state_dict',task='both')
    base=next(r for r in rows if r['id']=='qsilu_e2')
    for route in ['one2one','one2many']:
        r={**base,'id':'qsilu_pose_'+route,'stage':'推論','task':'pose','route':route}
        r['metrics']={k:v for k,v in base['metrics'].items() if k.startswith('bbat/')}
        if route=='one2many':
            p=ROOT/'experiments/inference/routing_v1/artifacts/one2many-v1/summary.json';s=json.loads(p.read_text())
            r.update(metrics={k:v for k,v in s['results']['bittrue']['metrics'].items() if k.endswith('map50_95')},metrics_source=str(p),metrics_source_sha256=sha(p))
        rows.append(r)
    out=HERE/'manifest.json';assert not out.exists()
    out.write_text(json.dumps({'cases':rows,'precision':'FP32','batch':1,'imgsz':640,'cpu_threads':4,
        'seed':50900912,'synthetic_input':True,'AP_reused_not_remeasured':True},ensure_ascii=False,indent=2)+'\n')
    print('MANIFEST_READY',len(rows))

if __name__=='__main__':main()
