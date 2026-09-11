"""只用 canonical train、只改 Pose head BN buffers 的可丟棄診斷。"""
import copy
import json
from dataclasses import replace
from types import MappingProxyType
import torch
from safe_source import HERE, initialize, sha256
from recover_pose_head import RecoverySource
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.joint_data import TaskLoaderSettings, build_task_loader
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import JointValidator, ValidationSettings
from yolo_combine.metrics import GATE_METRICS
import yolo_combine.validation as validation
from validate import InternalValidator
from common import prepare_coco

START = HERE/'artifacts/fusion/j3-pose-head-recovery-v1/inference/best_pose.pt'
SHA = 'd5b2083b2b736ded790578a2779d223d82413ce0c6cbac1468601105c094f9a6'


def main():
    initialize(); torch.set_num_threads(4); torch.manual_seed(20261003)
    out = HERE/'artifacts/j3-pose-bn-calibration-v1'
    out.mkdir(exist_ok=False)
    assert sha256(START) == SHA
    payload = torch.load(START,map_location='cpu',weights_only=True)
    source = RecoverySource()
    pair = source.build_task_models('float')
    model, report = assemble_graph_shared_model(pair.detect,pair.pose)
    assert report.complete
    model.load_state_dict(payload['state_dict'],strict=True)
    model = model.cuda().eval().requires_grad_(False)
    modules = [(n,m) for n,m in model.named_modules() if '.pose_head.' in n and isinstance(m,torch.nn.BatchNorm2d)]
    assert modules
    allowed = {n+'.'+suffix for n,m in modules for suffix in ('running_mean','running_var','num_batches_tracked')}
    fixed = {k:v.clone() for k,v in payload['state_dict'].items() if k not in allowed}
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',out/'datasets/bbat5-v1-runtime')
    settings = TaskLoaderSettings.for_pose(batch_size=128,workers=4,seed=20261003)
    settings = replace(settings,augmentation=MappingProxyType({k:0. for k in settings.augmentation}))
    loader = build_task_loader(model,data_yaml=view.yaml,settings=settings,device=torch.device('cuda:0'),
        registry='/home/uxin/yolo/configs/datasets/bbat5-v1.yaml')
    assert len(loader.dataset) == 5964
    original_momentum = {n:m.momentum for n,m in modules}
    for n,m in modules:
        m.reset_running_stats(); m.train()
    seen = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader.loader:
            image = loader.preprocess(batch)['img']
            n = image.shape[0]
            for _,m in modules: m.momentum = n/(seen+n)
            with torch.autocast('cuda',dtype=torch.float16): model(image,task='pose')
            seen += n
    assert seen == 5964
    peak = torch.cuda.max_memory_allocated()
    for n,m in modules: m.momentum = original_momentum[n]
    model.cpu().eval()
    state = model.state_dict()
    assert all(torch.equal(v,state[k]) for k,v in fixed.items())
    validation.DetectionValidator = InternalValidator
    validator = JointValidator(source,detect_data_yaml=prepare_coco(),pose_data_yaml=view.yaml,
        output_root=out/'validation',settings=ValidationSettings(imgsz=640,detect_batch_size=32,
            pose_batch_size=16,detect_workers=4,pose_workers=4,device='0',plots=False,save_coco_json=False))
    result = validator.validate(model,epoch=0,kind='bittrue')
    delta = {k:result.metrics[k]-payload['metadata']['metrics'][k] for k in GATE_METRICS}
    assert all(abs(v)<1e-8 for k,v in delta.items() if k.startswith('coco/'))
    candidate = copy.deepcopy(payload)
    candidate['state_dict'] = state
    candidate['metadata'] = {'metrics':dict(result.metrics),'source_checkpoint':str(START),
        'source_sha256':SHA,'calibration':'train-only Pose head BN; candidate not accepted',
        'physical_batch':128,'images':seen}
    torch.save(candidate,out/'candidate.pt')
    with (out/'summary.json').open('x') as f:
        json.dump({'status':'completed','metrics':dict(result.metrics),'delta':delta,'images':seen,
            'physical_batch':128,'peak_memory_allocated':peak,'bn_layers':len(modules),
            'all_parameters_and_other_buffers_exact':True,'accepted':False},f,indent=2)
    print('JOB_DONE: Pose head BN train-only 診斷完成，候選待分析。',flush=True)


if __name__ == '__main__': main()
