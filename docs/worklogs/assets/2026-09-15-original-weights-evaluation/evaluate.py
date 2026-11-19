"""原始權重評估；資料、標註與 split 維持不變，停用資料 cache 寫入。"""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data import dataset
from ultralytics.models.yolo.detect.val import DetectionValidator

ROOT = Path('/home/uxin/yolo')
OUT = Path(__file__).resolve().parent
WEIGHTS = Path('/home/uxin/single_view_pipeline/single_view_pipeline/weights')
def keep_cache_in_memory(prefix, path, data, version):
    data["version"] = version
dataset.save_dataset_cache_file = keep_cache_in_memory
torch.set_num_threads(8)

class SaveOnlyValidator(DetectionValidator):
    def eval_json(self, stats):
        return stats

def extract(m):
    data = {}
    for kind in ['box', 'pose']:
        v = getattr(m, kind, None)
        if v is None:
            continue
        data[kind] = {'overall': dict(precision=float(v.mp), recall=float(v.mr), f1=float(np.mean(v.f1)), map50=float(v.map50), map75=float(v.map75), map50_95=float(v.map)), 'classes': {}}
        for i, c in enumerate(v.ap_class_index):
            data[kind]['classes'][m.names[int(c)]] = dict(precision=float(v.p[i]), recall=float(v.r[i]), f1=float(v.f1[i]), map50=float(v.ap50[i]), map75=float(v.all_ap[i,5]), map50_95=float(v.ap[i]))
    data['speed_ms'] = m.speed
    data['instances_per_class'] = m.nt_per_class.tolist()
    data['images_per_class'] = m.nt_per_image.tolist()
    return data

common = dict(imgsz=640, batch=16, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, half=False, rect=True, augment=False, plots=False, save_txt=False, project=str(OUT), exist_ok=True, verbose=True)
results = {'ultralytics':ultralytics.__version__, 'settings': common, 'dataset_id':'bbat5-v1', 'registry':str(ROOT/'configs/datasets/bbat5-v1.yaml'), 'split':'val', 'dataset_unchanged':True}
if (OUT/'results.json').exists():
    results = json.loads((OUT/'results.json').read_text())
for name, weight, yaml in [('coco','yolo11m.pt',ROOT/'coco2017.yaml'),('bbat','yolo11m_bat.pt',ROOT/'original/pose/derived/bbat5-v1/configs/pose.yaml')]:
    if name in results:
        continue
    model = YOLO(WEIGHTS/weight)
    extra = dict(validator=SaveOnlyValidator, save_json=True) if name == 'coco' else dict(save_json=False)
    m = model.val(data=str(yaml), name=name, **common, **extra)
    results[name] = {'weight':str(WEIGHTS/weight), 'sha256':hashlib.sha256((WEIGHTS/weight).read_bytes()).hexdigest(), 'yaml':str(yaml), **extract(m)}
    (OUT/'results.json').write_text(json.dumps(results, indent=2))
    print('COMPLETED', name, flush=True)

from faster_coco_eval import COCO, COCOeval_faster as COCOeval
gt = COCO(str(ROOT/'coco2017/annotations/instances_val2017.json'))
pred = gt.loadRes(str(OUT/'coco/predictions.json'))
e = COCOeval(gt,pred,'bbox')
e.evaluate(); e.accumulate(); e.summarize()
p = e.eval['precision']
idx = list(e.params.catIds).index(1)
x = p[:,:,idx,0,-1]
results['coco']['official_coco'] = {'map50_95':float(e.stats[0]), 'map50':float(e.stats[1]), 'person_map50_95':float(x[x>-1].mean()), 'max_dets':100}
(OUT/'results.json').write_text(json.dumps(results, indent=2))
print('ALL_COMPLETE', flush=True)
