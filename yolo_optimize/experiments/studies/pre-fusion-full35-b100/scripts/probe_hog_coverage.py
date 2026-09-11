"""固定 1024 張增強後 train prefix：HOG 每個 GT 自身框的 stride8 有效 cell。"""
from common import ROOT,SOURCE,setup,prepare_coco,write_json
from train_hog import Harness
from yolo_optimize.hog import build_hog_targets
import torch


def main():
    output=ROOT/'artifacts/hog-class-coverage-v1';output.mkdir(parents=True,exist_ok=False)
    setup();data=prepare_coco();torch.set_num_threads(8)
    trainer=Harness(overrides={'model':str(SOURCE/'weights/bittrue/a0.pt'),'data':str(data),
        'epochs':10,'batch':32,'nbs':128,'imgsz':640,'device':'cpu','workers':4,'amp':False,
        'optimizer':'AdamW','project':str(output),'name':'setup','exist_ok':False,'seed':20260919,
        'deterministic':True,'mosaic':0.0,'mixup':0.0,'cutmix':0.0,'copy_paste':0.0,'fliplr':0.5,
        'cache':False,'fraction':1.0,'plots':False,'save_json':False,'warmup_epochs':1.0,
        'save':False,'patience':4,'cos_lr':True,'close_mosaic':0})
    trainer._setup_train()
    categories={'all80':None,'person':0,'sports_ball':32,'baseball_bat':34}
    records={k:{'instances':0,'no_cell_center':0,'no_valid_energy_cell':0,'short_side_lt8px':0,
        'valid_cells':0,'images':set()} for k in categories}
    observed=0
    with torch.inference_mode():
        for index,batch in enumerate(trainer.train_loader):
            if index==32:break
            batch=trainer.preprocess_batch(batch);images=batch['img'];boxes=batch['bboxes'];ids=batch['batch_idx'].long().flatten()
            target=build_hog_targets(images,boxes,ids)
            h,w=target.energy.shape[-2:]
            xs=(torch.arange(w)+.5)/w;ys=(torch.arange(h)+.5)/h
            lo=boxes[:,:2]-boxes[:,2:]/2;hi=boxes[:,:2]+boxes[:,2:]/2
            mask=(xs[None,None,:]>=lo[:,0,None,None])&(xs[None,None,:]<hi[:,0,None,None])
            mask=mask&(ys[None,:,None]>=lo[:,1,None,None])&(ys[None,:,None]<hi[:,1,None,None])
            cells=mask.sum((1,2));valid=(mask&(target.energy[ids]>1e-6)).sum((1,2))
            small=(boxes[:,2]*images.shape[-1]<8)|(boxes[:,3]*images.shape[-2]<8)
            for name,category in categories.items():
                select=torch.ones(len(boxes),dtype=torch.bool) if category is None else batch['cls'].flatten()==category
                record=records[name];record['instances']+=int(select.sum())
                record['no_cell_center']+=int(((cells==0)&select).sum())
                record['no_valid_energy_cell']+=int(((valid==0)&select).sum())
                record['short_side_lt8px']+=int((small&select).sum());record['valid_cells']+=int(valid[select].sum())
                record['images'].update(observed+int(i) for i in ids[select].unique())
            observed+=len(images)
    assert observed==1024
    for record in records.values():
        record['images']=len(record['images'])
        record['zero_valid_cell_fraction']=record['no_valid_energy_cell']/max(record['instances'],1)
    write_json(output/'summary.json',{'status':'passed','images':observed,'device':'cpu','optimizer_steps':0,
        'sample_rule':'固定 seed20260919 的前1024張增強後 train 影像；不依 AP 選樣',
        'categories':records,'limitation':'自身框無有效cell不代表完全無梯度，仍可能受其他重疊框與原生loss監督；prefix並非全資料估計'})
    print('JOB_DONE HOG_COVERAGE',records,flush=True)


if __name__=='__main__':main()
