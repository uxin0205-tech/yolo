"""固定驗證集的操作點診斷：框匹配、漏檢與可見 keypoint 誤差，不計算 AP。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0:'ball',1:'bat'}


def iou_matrix(a, b):
    a=np.asarray(a).reshape(-1,4); b=np.asarray(b).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(hi-lo,0),axis=2)
    aa=np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=1)
    bb=np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)


def match_boxes(gt_boxes,gt_cls,pred_boxes,pred_cls,scores):
    """依信心排序、同類別且 IoU>=0.5 的一對一匹配。"""
    overlaps=iou_matrix(pred_boxes,gt_boxes);pairs={};used=set()
    for p in np.argsort(-scores,kind='stable'):
        eligible=[g for g in range(len(gt_cls)) if g not in used and gt_cls[g]==pred_cls[p] and overlaps[p,g]>=.5]
        if eligible:
            g=max(eligible,key=lambda g:overlaps[p,g]);pairs[g]=int(p);used.add(g)
    return pairs,overlaps


def distribution(values):
    if not values:return {'count':0,'mean':None,'median':None,'p90':None}
    a=np.asarray(values,dtype=float)
    return {'count':len(values),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9))}


def write_csv(path,rows):
    if rows:
        with path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(weights,output,sizes):
    from ultralytics import YOLO
    output.mkdir(parents=True,exist_ok=True)
    data=yaml.safe_load((ROOT/'configs/data.runtime.yaml').read_text());view=Path(data['path'])
    files=sorted((view/'images/val').glob('*'));assert len(files)==683
    digest=hashlib.sha256()
    for image in files:
        label=view/'labels/val'/f'{image.stem}.txt';digest.update(label.name.encode()+label.read_bytes())
    audit=json.loads((ROOT/'artifacts/dataset-audit.json').read_text())
    assert digest.hexdigest()==audit['splits']['val_labels_sha256']
    model=YOLO(weights);summaries=[]
    for size in sizes:
        objects=[];points=[];fps={0:0,1:0};visible={0:0,1:0}
        for start in range(0,len(files),8):
            paths=files[start:start+8]
            predictions=model.predict(source=[str(p) for p in paths],imgsz=size,device=0,batch=8,
                                      conf=.25,iou=.7,max_det=300,rect=False,half=False,
                                      verbose=False,save=False,stream=False)
            for path,result in zip(paths,predictions):
                h,w=result.orig_shape;image=result.orig_img
                label=view/'labels/val'/f'{path.stem}.txt'
                rows=np.array([[float(x) for x in line.split()] for line in label.read_text().splitlines() if line.strip()]).reshape(-1,11)
                gc=rows[:,0].astype(int);xywh=rows[:,1:5]*[w,h,w,h]
                gb=np.column_stack((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2))
                gk=rows[:,5:].reshape(-1,2,3).copy();gk[:,:,:2]*=[w,h]
                pb=result.boxes.xyxy.cpu().numpy();pc=result.boxes.cls.cpu().numpy().astype(int);scores=result.boxes.conf.cpu().numpy()
                pk=result.keypoints.data.cpu().numpy()
                pairs,overlaps=match_boxes(gb,gc,pb,pc,scores)
                for pi in set(range(len(pb)))-set(pairs.values()):fps[int(pc[pi])]+=1
                for g,cls in enumerate(gc):
                    cls=int(cls);visible[cls]+=int((gk[g,:,2]>0).sum())
                    matched=g in pairs;diag=max(float(np.linalg.norm(xywh[g,2:])),1e-9)
                    side=float(np.sqrt(np.prod(xywh[g,2:]))*640/max(h,w))
                    size_bin='tiny_<16' if side<16 else 'small_16_32' if side<32 else 'medium_32_96' if side<96 else 'large_96+'
                    same=overlaps[pc==cls,g];other=overlaps[pc!=cls,g]
                    maxsame=float(same.max()) if len(same) else 0.;maxother=float(other.max()) if len(other) else 0.
                    reason='matched' if matched else ('class_confusion' if maxother>=.5 else 'assignment_conflict' if maxsame>=.5 else 'poor_box_overlap' if maxsame>=.1 else 'no_same_class_overlap')
                    x1,y1,x2,y2=gb[g];crop=image[max(0,int(y1)):min(h,int(np.ceil(y2))),max(0,int(x1)):min(w,int(np.ceil(x2)))]
                    blur=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if min(crop.shape[:2],default=0)>=3 else None
                    obj={'image':path.name,'gt_index':g,'class':NAMES[cls],'matched':int(matched),'reason':reason,
                         'size_bin':size_bin,'sqrt_area_at640_px':side,'laplacian_variance':blur,
                         'best_same_class_iou':maxsame,'matched_iou':None,'center_error_px':None,'center_error_bbox_diag':None}
                    if matched:
                        pi=pairs[g];obj['matched_iou']=float(overlaps[pi,g]);center=(pb[pi,:2]+pb[pi,2:])/2
                        error=float(np.linalg.norm(center-xywh[g,:2]));obj['center_error_px']=error;obj['center_error_bbox_diag']=error/diag
                        for k in range(2):
                            if gk[g,k,2]<=0:continue
                            distance=float(np.linalg.norm(pk[pi,k,:2]-gk[g,k,:2]));normal=distance/diag
                            points.append({'image':path.name,'gt_index':g,'class':NAMES[cls],'keypoint':k,
                                           'error_px':distance,'error_bbox_diag':normal,
                                           'predicted_visibility_conf':float(pk[pi,k,2]) if pk.shape[2]>2 else None})
                    objects.append(obj)
            if start%80==0:print(f'診斷 imgsz={size}：{min(start+8,len(files))}/{len(files)}',flush=True)
        classes={}
        for cls,name in NAMES.items():
            obs=[o for o in objects if o['class']==name];pts=[p for p in points if p['class']==name];tp=sum(o['matched'] for o in obs)
            groups={}
            for key in sorted(set(o['size_bin'] for o in obs)):
                g=[o for o in obs if o['size_bin']==key];groups[key]={'gt':len(g),'matched':sum(o['matched'] for o in g),'recall':sum(o['matched'] for o in g)/len(g)}
            norm=[p['error_bbox_diag'] for p in pts]
            classes[name]={'gt':len(obs),'tp':tp,'fn':len(obs)-tp,'fp':fps[cls],
                           'precision':tp/max(tp+fps[cls],1),'recall':tp/max(len(obs),1),
                           'unmatched_reasons':{key:sum(o['reason']==key for o in obs) for key in sorted(set(o['reason'] for o in obs)) if key!='matched'},
                           'by_size':groups,'matched_iou':distribution([o['matched_iou'] for o in obs if o['matched']]),
                           'box_center_error_px':distribution([o['center_error_px'] for o in obs if o['matched']]),
                           'visible_gt_keypoints':visible[cls],'matched_keypoints':len(pts),
                           'keypoint_error_px':distribution([p['error_px'] for p in pts]),
                           'keypoint_error_bbox_diag':distribution(norm),
                           'pck_at_0p1_matched_only':sum(x<=.1 for x in norm)/len(norm) if norm else None,
                           'pck_at_0p1_all_visible_gt':sum(x<=.1 for x in norm)/visible[cls] if visible[cls] else None,
                           'by_keypoint':{str(k):distribution([p['error_px'] for p in pts if p['keypoint']==k]) for k in range(2)}}
        summaries.append({'imgsz':size,'classes':classes})
        write_csv(output/f'objects_{size}.csv',objects);write_csv(output/f'keypoints_{size}.csv',points)
    report={'weights':str(weights),'weight_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            'dataset':'bbat5-v1 formal val','images':len(files),'label_sha256':digest.hexdigest(),
            'protocol':{'conf':.25,'matching_iou':.5,'class_aware':True,'matching':'confidence-ordered greedy one-to-one',
                        'rect':False,'batch':8,'half':False,'keypoints':'GT visibility>0; predicted keypoint confidence is reported, not filtered',
                        'normalized_error':'distance / GT bbox diagonal','size_bins':'sqrt area after letterbox scaling to 640; invariant across evaluation resolutions',
                        'caveat':'fixed-threshold diagnostic, not AP; matched-only errors have selection bias; all-visible-GT PCK counts missed detections as failures; blur is size/texture-confounded, no occlusion or motion-blur ground truth'},
            'results':summaries}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# 框與關鍵點誤差診斷','', '固定conf=0.25、同類別IoU≥0.5匹配，非AP，也非validator自動選擇的最佳F1操作點。',
           '關鍵點像素誤差只計成功匹配且GT可見的點；全體PCK將漏檢算失敗。未篩掉低預測visibility confidence。', '',
           '| 解析度 | 類別 | GT | 漏檢 | 額外預測 | Recall | 關鍵點中位像素誤差 | 全體PCK@0.1框對角線 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for res in summaries:
        for name,c in res['classes'].items():
            lines.append(f"| {res['imgsz']} | {name} | {c['gt']} | {c['fn']} | {c['fp']} | {c['recall']:.4f} | {c['keypoint_error_px']['median']} | {c['pck_at_0p1_all_visible_gt']} |")
    lines += ['', '逐物件明細含小物件尺寸、IoU、中心誤差、Laplacian variance；模糊量受尺寸與紋理影響，不作因果歸因。資料沒有遮擋或動態模糊真值，不能自動可靠判定這兩類原因。',
              '此固定閾值診斷與正式AP評估分開；資料分配與labels不修改。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[640,960]);args=parser.parse_args();analyze(args.weights,args.output,args.sizes)
