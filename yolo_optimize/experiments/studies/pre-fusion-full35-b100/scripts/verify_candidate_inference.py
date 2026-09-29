"""late E8 匯出、完整 COCO 重驗與固定 person 畫面的獨立推論。"""
import json
import subprocess
import sys
from common import ROOT,SOURCE,setup,prepare_coco,write_json,sha256
import torch
from ultralytics import YOLO
from export_model import export
from continue_a0 import validate


def main():
    output=ROOT/'artifacts/late-e8-inference-verification'
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    snapshot=ROOT/'artifacts/a0-scope-late-v1/epoch-08-resume.pt'
    destination=export(snapshot,output/'late-e8-bittrue.pt')
    setup();data=prepare_coco()
    expected=json.loads((snapshot.parent/'summary.json').read_text())['epochs'][-1]
    assert expected['epoch']==8
    metrics=validate(YOLO(str(destination)).model,'bittrue',data,output/'full-coco')
    tracked=('coco/box/map50_95','coco/person/box/map50_95')
    deltas={k:metrics[k]-expected['ema'][k] for k in tracked}
    if any(abs(v)>1e-6 for v in deltas.values()):raise AssertionError(f'匯出 AP 改變：{deltas}')
    runtime=data.parent
    images=[]
    for relative in (runtime/'val2017.txt').read_text().splitlines():
        path=runtime/relative
        label=runtime/'labels/val2017'/f'{path.stem}.txt'
        if any(line.split()[0]=='0' for line in label.read_text().splitlines() if line.strip()):
            images.append(str(path.resolve()))
        if len(images)==8:break
    if len(images)!=8:raise ValueError('固定 person 推論樣本不足')
    image_list=output/'person-first8.txt'
    image_list.write_text('\n'.join(images)+'\n')
    for name,weights in [('a0-person-first8',SOURCE/'weights/bittrue/a0.pt'),('late-e8-person-first8',destination)]:
        subprocess.run([sys.executable,str(ROOT/'scripts/infer.py'),'--weights',str(weights),
            '--source',str(image_list),'--name',name,'--conf','0.25'],check=True)
    report={'status':'passed','snapshot':str(snapshot),'inference':str(destination),
        'inference_sha256':sha256(destination),'metrics':{k:metrics[k] for k in tracked},
        'export_ap_delta':deltas,'full_coco_images':5000,'inference_images_per_model':len(images),
        'sample_rule':'COCO val 原始清單順序，前8張標註含 person 的影像；選樣不依預測效果',
        'inference_classes':'all80','confidence':0.25,'inference_imgsz':640,
        'qualitative_judgement_done':False,'independent_scene_validation_done':False,
        'hardware_latency_measured':False,'accuracy_winner':False}
    write_json(output/'summary.json',report)
    print('JOB_DONE: late E8 匯出 AP 保留與獨立推論通過；不代表真實場景或硬體已驗收',flush=True)


if __name__=='__main__':main()
