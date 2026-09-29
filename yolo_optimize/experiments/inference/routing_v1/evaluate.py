"""同一 qSiLU 權重的 BBAT one2many＋NMS 推論驗證；不訓練。"""
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'kd/dual_task_v1'))
from teachers import SelectedSource, SELECTED, SELECTED_SHA, initialize, sha256
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import extract_pose_metrics
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.nms import non_max_suppression


def main():
    initialize()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    out = HERE / 'artifacts/one2many-v1'
    out.mkdir(parents=True, exist_ok=False)
    assert sha256(SELECTED) == SELECTED_SHA
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',
                            HERE / 'artifacts/datasets/bbat5-v1-runtime')
    results = {}
    for kind in ('float', 'bittrue'):
        pair = SelectedSource().build_task_models(kind)
        model = pair.pose.eval().requires_grad_(False)
        model.end2end = False
        with torch.inference_mode():
            pred = model(torch.rand(1, 3, 160, 160))
            assert pred[0].shape == (1, 12, 525) and torch.isfinite(pred[0]).all()
            assert 'boxes' in pred[1] and 'one2one' not in pred[1]
            filtered = non_max_suppression(pred, conf_thres=.001, iou_thres=.7,
                nc=2, max_det=300, agnostic=False, end2end=False)
            assert len(filtered) == 1 and filtered[0].shape[1] == 12
        validator = PoseValidator(save_dir=out / kind, args=dict(task='pose', data=str(view.yaml),
            imgsz=640, batch=16, workers=4, device='0', plots=False, save_json=False,
            compile=False, rect=True, split='val', mode='val', half=False,
            end2end=False, conf=.001, iou=.7, agnostic_nms=False, max_det=300))
        validator(model=model)
        assert len(validator.dataloader.dataset) == 683
        assert not validator.end2end and not model.end2end
        metrics = extract_pose_metrics(validator.metrics, names=model.names)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        results[kind] = dict(metrics=metrics, images=683, end2end=False,
                             speed_observation=validator.speed,
                             speed_not_controlled_benchmark=True)
        with (out / (kind+'.json')).open('x') as f:
            json.dump(results[kind], f, indent=2)
        del validator, model, pair
        torch.cuda.empty_cache()
    baseline = json.loads((ROOT/'kd/dual_task_v1/artifacts/student-baseline-v1.json').read_text())['metrics']
    delta = {k: v-baseline[k] for k, v in results['bittrue']['metrics'].items() if k.endswith('map50_95')}
    summary = dict(status='completed', checkpoint=str(SELECTED), checkpoint_sha256=SELECTED_SHA,
        changed_weights=False, training_performed=False, coco_route_changed=False,
        coco_revalidated=False, dataset='bbat5-v1 canonical val683 unchanged',
        config=dict(end2end=False, conf=.001, iou=.7, agnostic_nms=False, max_det=300, imgsz=640),
        results=results, bittrue_minus_existing_one2one=delta,
        limitation='existing one2one baseline reused; timing is observation only; no threshold sweep or test set')
    with (out/'summary.json').open('x') as f:
        json.dump(summary, f, indent=2)
    assert sha256(SELECTED) == SELECTED_SHA
    print('JOB_DONE '+json.dumps(delta), flush=True)


if __name__ == '__main__':
    main()
