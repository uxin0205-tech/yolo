"""雙教師限定來源與 weights_only 載入；不將教師併入部署模型。"""
import copy
import importlib
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'activation/bridge_v1'))
from verify_selected import SelectedSource, SELECTED, SELECTED_SHA, initialize, sha256
from safe_source import BridgeSource, POSE, POSE_SHA
import torch

DETECT = HERE / 'artifacts/teachers/yolo26l.pt'
DETECT_SHA = '9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26'
DETECT_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt'
ALLOWED = {
    'ultralytics.nn.modules.head': 'Detect',
    'ultralytics.nn.modules.block': 'C3k Bottleneck SPPF C2PSA C3k2 Attention PSABlock',
    'ultralytics.nn.modules.conv': 'Conv DWConv Concat',
    'ultralytics.nn.tasks': 'DetectionModel',
    'torch.nn.modules.linear': 'Identity',
    'torch.nn.modules.activation': 'SiLU',
    'torch.nn.modules.conv': 'Conv2d',
    'torch.nn.modules.container': 'Sequential ModuleList',
    'torch.nn.modules.batchnorm': 'BatchNorm2d',
    'torch.nn.modules.upsampling': 'Upsample',
    'torch.nn.modules.pooling': 'MaxPool2d',
}

def detect_teacher():
    assert sha256(DETECT) == DETECT_SHA
    allowed = [getattr(importlib.import_module(module), name)
               for module, names in ALLOWED.items() for name in names.split()]
    expected = {f'{m}.{n}' for m, ns in ALLOWED.items() for n in ns.split()}
    assert set(torch.serialization.get_unsafe_globals_in_checkpoint(DETECT)) <= expected
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(DETECT, map_location='cpu', weights_only=True)
    model = copy.deepcopy(payload.get('ema') or payload['model']).float().eval().requires_grad_(False)
    assert model.model[-1].nc == 80 and model.names[0] == 'person'
    return model

def pose_teacher():
    model = BridgeSource().original_pose('float').eval().requires_grad_(False)
    assert model.names == {0: 'ball', 1: 'bat'}
    assert tuple(model.model[-1].kpt_shape) == (2, 3)
    return model

def main():
    import json
    from common import prepare_coco
    from validate import InternalValidator
    from yolo_combine.data import prepare_bbt5_view
    from yolo_combine.validation import extract_detect_metrics, extract_pose_metrics
    from ultralytics.models.yolo.pose import PoseValidator
    initialize(); torch.set_num_threads(4)
    out = HERE / 'artifacts/teacher-validation-v1'
    out.mkdir(exist_ok=False)
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',
                            HERE / 'artifacts/datasets/bbat5-v1-runtime')
    results = {}
    for task, build, validator_cls, data, size in (
        ('detect', detect_teacher, InternalValidator, prepare_coco(), 5000),
        ('pose', pose_teacher, PoseValidator, view.yaml, 683),
    ):
        model = build()
        validator = validator_cls(save_dir=out/task, args=dict(task=task, data=str(data),
            imgsz=640, batch=32 if task=='detect' else 16, workers=4, device='0',
            plots=False, save_json=False, compile=False, rect=True, split='val', mode='val', half=False))
        validator(model=model)
        assert len(validator.dataloader.dataset) == size
        metrics = (extract_detect_metrics if task=='detect' else extract_pose_metrics)(
            validator.metrics, names=model.names)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        results[task] = metrics
        with (out/(task+'.json')).open('x') as f:
            json.dump({'metrics': metrics, 'images': size, 'checkpoint': str(DETECT if task=='detect' else POSE),
                       'sha256': DETECT_SHA if task=='detect' else POSE_SHA}, f, indent=2)
        del model, validator
        InternalValidator.last_instance = None
        torch.cuda.empty_cache()
    student = torch.load(SELECTED, map_location='cpu', weights_only=True)['metadata']['metrics']
    delta = {k:v-student[k] for task in results.values() for k,v in task.items()
             if k.endswith('map50_95') and k in student}
    with (out/'summary.json').open('x') as f:
        json.dump({'status':'completed', 'results':results, 'teacher_minus_student':delta,
                   'student_sha256':SELECTED_SHA, 'bbat_dataset':'bbat5-v1 unchanged',
                   'training_performed':False}, f, indent=2)
    print('JOB_DONE: 雙教師同口徑驗證完成 '+json.dumps(delta), flush=True)

if __name__ == '__main__': main()
