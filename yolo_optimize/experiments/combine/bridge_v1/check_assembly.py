"""完整 Pose 訓練後的共享初始化診斷；不覆寫來源，不訓練。"""
import json
import copy
import torch
from safe_source import HERE, initialize
from full_pose import FullPoseSource
from common import prepare_coco
from yolo_combine.factory import FusionModelFactory
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import JointValidator, ValidationSettings
import yolo_combine.validation as validation
from validate import InternalValidator


def main():
    initialize()
    torch.set_num_threads(4)
    out = HERE / 'artifacts/full-pose-assembly-v1'
    out.mkdir(exist_ok=False)
    source = FullPoseSource()
    coco = prepare_coco()
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml', HERE / 'artifacts/datasets/bbat5-v1-runtime')
    built = FusionModelFactory(source, detect_data_yaml=coco, pose_data_yaml=view.yaml).build(
        pose_head_checkpoint=source.record['pose_checkpoint'], checkpoint_kind='float')
    base = copy.deepcopy(built.model.state_dict())
    full = torch.load(HERE / 'artifacts/fusion/full-pose-gentle-v1/inference/best_pose.pt', map_location='cpu', weights_only=True)
    validation.DetectionValidator = InternalValidator
    validator = JointValidator(source, detect_data_yaml=coco, pose_data_yaml=view.yaml,
        output_root=out / 'validation', settings=ValidationSettings(imgsz=640, detect_batch_size=32,
            pose_batch_size=16, detect_workers=4, pose_workers=4, device='0', plots=False, save_coco_json=False))
    results = {}
    for ratio in (0., .1, .25):
        state = copy.deepcopy(base)
        changed = []
        for name, value in full['state_dict'].items():
            if '.pose_head.' in name:
                state[name] = value
            elif '.detect_head.' not in name and value.dtype.is_floating_point and not torch.equal(base[name], value):
                state[name] = base[name] + ratio * (value-base[name])
                changed.append(name)
        built.model.load_state_dict(state, strict=True)
        result = validator.validate(built.model.eval(), epoch=len(results), kind='bittrue')
        results[str(ratio)] = dict(result.metrics)
        if ratio == 0:
            assert abs(result.metrics['coco/box/map50_95']-.5082119551806837) < 1e-8
            assert abs(result.metrics['coco/person/box/map50_95']-.6276641270640566) < 1e-8
        with (out / f'ratio-{ratio}.json').open('x') as f:
            json.dump({'ratio_of_pose_trunk': ratio, 'metrics': results[str(ratio)], 'changed_names': changed}, f, indent=2)
        del result
        InternalValidator.last_instance = None
    with (out / 'summary.json').open('x') as f:
        json.dump({'status': 'completed', 'results': results,
            'scope': 'initialization diagnostic; not trained fusion or deployment acceptance'}, f, indent=2)
    print('JOB_DONE: full Pose assembly compatibility verified', flush=True)


if __name__ == '__main__':
    main()
