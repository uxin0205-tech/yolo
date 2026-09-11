#!/usr/bin/env python3
"""雙向交換 Pose one-to-one 分類分支，僅作完整驗證的責任定位。"""
import sys
import argparse
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from yolo_optimize import runtime
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.validation import JointValidator, ValidationSettings, extract_pose_metrics, PoseValidator
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('classifier','pose_head'),default='classifier')
    scope = parser.parse_args().scope
    root = ROOT/'artifacts/direction1-20260909'/f'pose-{scope}-probe-v2'
    root.mkdir(parents=True,exist_ok=False)
    data_root = ROOT/'artifacts/direction1-20260908'
    config = runtime.load_config(data_root)
    detect,pose = runtime.prepare_data(config,data_root/'datasets')
    paths = {'parent':runtime.FINAL_ROOT/'weights/combined/inference/best_joint.pt',
        'native_e5':data_root/'native-parent-ema-control-adopted/inference/epoch-0005.pt'}
    results = {}
    for recipient,donor in (('native_e5','parent'),('parent','native_e5')):
        name = f'{recipient}_with_{donor}_{scope}'
        source,model,_,_ = runtime.load_model(config,paths[recipient],torch.device('cuda:0'))
        donor_state = torch.load(paths[donor],map_location='cpu',weights_only=True,mmap=True)['state_dict']
        before = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        seam = '.pose_head.one2one_cv3.' if scope=='classifier' else '.pose_head.'
        keys = [key for key in before if seam in key]
        if not keys: raise ValueError('Pose one-to-one classifier mapping 為空')
        if before.keys() != donor_state.keys(): raise ValueError('state schema 不同')
        with torch.no_grad():
            for key,value in model.state_dict().items():
                if key in keys: value.copy_(donor_state[key].to(value))
        changed=[]
        for key,value in model.state_dict().items():
            if not torch.equal(before[key],value.cpu()):
                if key not in keys: raise AssertionError('越界改動')
                changed.append(key)
            if key in keys: assert torch.equal(value.cpu(),donor_state[key])
        assert changed
        del before,donor_state
        models = build_graph_validation_models(model,source,kind='bittrue')
        builder = JointValidator(source,detect_data_yaml=detect,pose_data_yaml=pose,
            output_root=root/name,settings=ValidationSettings(device='cuda:0',plots=False))
        validator = PoseValidator(save_dir=root/name,
            args=builder._args(task='pose',data=Path(pose),batch=16,workers=8))
        validator(model=models.pose)
        assert validator.seen==683
        results[name] = {'recipient':str(paths[recipient]),'donor':str(paths[donor]),
            'recipient_sha256':runtime.sha256(paths[recipient]),'donor_sha256':runtime.sha256(paths[donor]),
            'replaced_state_keys':keys,'changed_state_keys':changed,
            'metrics':extract_pose_metrics(validator.metrics,names=models.pose.names),
            'classification_bn_included':True,'scope':scope,'training_performed':False,'deployable_candidate':False}
        runtime.write_json(root/f'{name}.json',results[name])
        del validator,models,model
        torch.cuda.empty_cache()
    runtime.write_json(root/'summary.json',{'status':'complete','cases':results,
        'scope':'full canonical BBAT5 val683, bidirectional eval-only branch swap; not training causal proof'})
    print('JOB_DONE: 雙向 Pose classifier 交換診斷完成',flush=True)


if __name__=='__main__': main()
