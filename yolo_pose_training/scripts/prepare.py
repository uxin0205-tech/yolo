from pathlib import Path
import yaml,json,hashlib
root=Path('/home/uxin/yolo/yolo_pose_training')
canonical=Path('/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml')
data=yaml.safe_load(canonical.read_text()); source=Path(data['path'])
view=root/'artifacts/datasets/bbat5-v1/pose'
audit={'dataset_id':'bbat5-v1','canonical_yaml':str(canonical),'registry':'/home/uxin/yolo/configs/datasets/bbat5-v1.yaml','assignment_unchanged':True,'splits':{}}
for split,expected in [('train',5964),('val',683)]:
    h=hashlib.sha256()
    for kind in ('images','labels'):
        src=source/kind/split; dst=view/kind/split; dst.mkdir(parents=True,exist_ok=True)
        files=sorted(p for p in src.iterdir() if p.is_file())
        assert len(files)==expected,(split,kind,len(files))
        for p in files:
            target=dst/p.name
            if not target.exists(): target.symlink_to(p)
            assert target.resolve()==p.resolve()
            if kind=='labels': h.update(p.name.encode()+p.read_bytes())
        audit['splits'][split+'_'+kind]=len(files)
    audit['splits'][split+'_labels_sha256']=h.hexdigest()
data['path']=str(view)
(root/'configs/data.runtime.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
(root/'artifacts/dataset-audit.json').write_text(json.dumps(audit,indent=2))
base=dict(data=str(root/'configs/data.runtime.yaml'),task='pose',epochs=20,imgsz=640,batch=32,freeze=10,optimizer='AdamW',lr0=1e-3,lrf=.1,weight_decay=5e-4,warmup_epochs=3,warmup_bias_lr=0.0,cos_lr=True,mosaic=.5,close_mosaic=5,mixup=0.,cutmix=0.,copy_paste=0.,degrees=5.,translate=.1,scale=.4,fliplr=0.,flipud=0.,box=7.5,cls=.5,dfl=1.5,pose=12.,kobj=1.,rle=1.,seed=0,deterministic=True,amp=True,device=0,workers=8,cache=False,save=True,save_period=10,patience=0,val=True,plots=True,project=str(root/'runs'),exist_ok=False,pretrained=True,resume=False)
names=['stage1_adamw_640','stage2_adamw_640','stage3_adamw_960','stage4_musgd_960']
updates=[{},dict(epochs=50,freeze=None,lr0=3e-4,lrf=.05,warmup_epochs=2,close_mosaic=10),dict(epochs=40,imgsz=960,batch=16,freeze=None,lr0=1e-4,warmup_epochs=1,mosaic=0.,close_mosaic=0,degrees=3.,translate=.05,scale=.2),dict(epochs=100,imgsz=960,batch=16,freeze=None,optimizer='MuSGD',lr0=1e-3,lrf=.05,momentum=.9,warmup_epochs=3,mosaic=0.,close_mosaic=0,degrees=3.,translate=.05,scale=.2,patience=30)]
for i,(name,update) in enumerate(zip(names,updates)):
    cfg=base|update|{'name':name,'model':str(root/'weights/yolo26m-pose.pt') if i==0 else str(root/'runs'/names[i-1]/'weights/best.pt')}
    (root/f'configs/{name}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(root/'configs/queue.yaml').write_text(yaml.safe_dump({'stages':[f'configs/{n}.yaml' for n in names],'monitor_seconds':600,'comparison_imgsz':960},sort_keys=False))
