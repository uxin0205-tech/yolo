"""研究模型獨立推論入口；只接受實際匯出的 Bit-True inference checkpoint。"""
import argparse
from pathlib import Path
from common import ROOT,setup
from ultralytics import YOLO


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--conf',type=float,default=0.25)
    parser.add_argument('--person-only-output',action='store_true')
    args=parser.parse_args()
    if not args.name.replace('-','').isalnum():raise ValueError('name 格式不符')
    if not args.weights.is_file() or not args.source.exists():raise FileNotFoundError('權重或輸入不存在')
    if not 0<args.conf<1:raise ValueError('conf 必須介於 0 與 1')
    output=ROOT/'artifacts/inference'/args.name
    if output.exists():raise FileExistsError('禁止覆寫推論結果')
    setup();model=YOLO(str(args.weights.resolve()))
    sites=[m for m in model.model.modules() if m.__class__.__name__=='HardwareFriendlyAttention']
    if len(sites)!=2 or any(m.config.normalization.value!='bit_true_pwl' for m in sites):
        raise ValueError('僅接受兩 site Bit-True inference model')
    if any(m.score.__class__.__name__!='BinaryScore' for m in sites):
        raise ValueError('先移除 training-only surrogate adapter 再推論')
    for _ in model.predict(source=str(args.source.resolve()),imgsz=640,device=0,half=False,
        conf=args.conf,classes=[0] if args.person_only_output else None,stream=True,
        save=True,save_txt=True,save_conf=True,project=str(output.parent),name=args.name,exist_ok=False):
        pass
    print('JOB_DONE inference',output)


if __name__=='__main__':main()
