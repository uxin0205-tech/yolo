"""固定 checkpoint 的單組回退；完整驗證，不修改任何來源。"""
import copy
import json
import torch
from safe_source import HERE, initialize
from adapted_source import AdaptedBridgeSource
from common import prepare_coco
from yolo_combine.factory import FusionModelFactory
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import JointValidator, ValidationSettings
import yolo_combine.validation as validation
from validate import InternalValidator


def main():
    initialize()
    torch.set_num_threads(4)
    out = HERE / 'artifacts/j1-diagnosis-v1'
    out.mkdir(exist_ok=False)
    source = AdaptedBridgeSource()
    coco = prepare_coco()
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml', HERE / 'artifacts/datasets/bbat5-v1-runtime')
    built = FusionModelFactory(source, detect_data_yaml=coco, pose_data_yaml=view.yaml).build(
        pose_head_checkpoint=source.record['pose_checkpoint'], checkpoint_kind='float')
    initial = copy.deepcopy(built.model.state_dict())
    trained = torch.load(HERE / 'artifacts/fusion/j1-bridge-v1/inference/last.pt', map_location='cpu', weights_only=True)
    validation.DetectionValidator = InternalValidator
    validator = JointValidator(source, detect_data_yaml=coco, pose_data_yaml=view.yaml,
        output_root=out / 'validation', settings=ValidationSettings(imgsz=640, detect_batch_size=32,
        pose_batch_size=16, detect_workers=4, pose_workers=4, device='0', plots=False, save_coco_json=False))
    cases = {
        'reproduce': lambda k: False,
        'restore_detect_bn': lambda k: '.detect_head.' in k and any(k.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked')),
        'restore_neck': lambda k: k.startswith('graph.model.') and 11 <= int(k.split('.')[2]) <= 22,
        'restore_detect_parameters': lambda k: '.detect_head.' in k and k in dict(built.model.named_parameters()),
        'restore_masf': lambda k: '.detect_head.p3_masf.' in k,
    }
    results = {}
    for name, predicate in cases.items():
        state = {k: (initial[k] if predicate(k) else v) for k, v in trained['state_dict'].items()}
        built.model.load_state_dict(state, strict=True)
        result = validator.validate(built.model.eval(), epoch=len(results), kind='bittrue')
        results[name] = dict(result.metrics)
        if name == 'reproduce':
            for k in ('coco/box/map50_95', 'coco/person/box/map50_95'):
                assert abs(results[name][k] - trained['metadata']['metrics'][k]) < 1e-8
            assert results[name]['coco/box/map50_95'] < .5082119551806837 - .005
            print('REPRODUCED: saved checkpoint COCO regression', flush=True)
        with (out / (name + '.json')).open('x') as f:
            json.dump({'metrics': results[name], 'restored_tensors': sum(predicate(k) for k in state)}, f, indent=2)
        del result
        InternalValidator.last_instance = None
    with (out / 'summary.json').open('x') as f:
        json.dump({'status': 'completed', 'results': results, 'scope': 'one-group rollback diagnostic, not retraining proof'}, f, indent=2)
    print('JOB_DONE: checkpoint rollback diagnosis complete', flush=True)


if __name__ == '__main__':
    main()
