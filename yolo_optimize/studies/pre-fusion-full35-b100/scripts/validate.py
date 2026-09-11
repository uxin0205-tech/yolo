"""B100 同口徑完整 COCO 驗證；不混入 Pose 或歷史棒球 split。"""
import argparse
import math
import json
from pathlib import Path
from common import ROOT, SOURCE, registry, sha256, write_json, setup, prepare_coco
from ultralytics import YOLO
from achitechure_1.evaluation import COCO2017Validator


class InternalValidator(COCO2017Validator):
    last_instance = None

    def init_metrics(self, model):
        super().init_metrics(model)
        type(self).last_instance = self
        self.args.save_json = False

    def eval_json(self, stats):
        return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['float','bittrue'],required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--candidate', choices=['b100','a0','fp'], default='b100')
    args=parser.parse_args()
    if not args.name.replace('-','').isalnum():raise ValueError('name 格式不符')
    output=ROOT/'artifacts'/args.name
    output.mkdir(parents=True,exist_ok=False)
    setup()
    data=prepare_coco()
    record=registry()
    checkpoint=SOURCE/record[args.backend]
    expected=record[args.backend+'_sha256']
    if args.candidate=='a0':
        if args.backend!='bittrue':raise ValueError('A0 只有 Bit-True checkpoint')
        record=next(x for x in json.loads((SOURCE/'models.json').read_text())['models'] if x['id']=='a0')
        checkpoint=SOURCE/record['bittrue'];expected=record['bittrue_sha256']
    elif args.candidate=='fp':
        if args.backend!='float':raise ValueError('官方 FP 不是 Bit-True 模型')
        checkpoint=Path('/home/uxin/yolo/yolo_attention/weights/yolo26m.pt')
        expected='401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'
    if sha256(checkpoint)!=expected:
        raise ValueError('來源權重雜湊不符')
    model=YOLO(str(checkpoint))
    results=model.val(validator=InternalValidator,data=str(data),imgsz=640,batch=32,
        device='0',workers=4,split='val',save_json=False,plots=False,
        project=str(output),name='validation',exist_ok=False,half=False)
    maps=[float(x) for x in results.box.maps]
    metrics={'coco/box/map50_95':float(results.box.map),
             'coco/person/box/map50_95':maps[0],
             'coco/ball/box/map50_95':maps[32],
             'coco/bat/box/map50_95':maps[34]}
    if len(maps)!=80 or not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('驗證指標不完整')
    write_json(output/'summary.json',{'status':'complete','backend':args.backend,'candidate':args.candidate,
        'checkpoint':str(checkpoint),'sha256':sha256(checkpoint),'metrics':metrics,
        'per_class_ap':maps,'images':len(InternalValidator.last_instance.dataloader.dataset),
        'training_performed':False,'evaluator':'Ultralytics internal; imgsz640; halfFalse; batch32'})
    print('JOB_DONE',metrics,flush=True)


if __name__=='__main__': main()
