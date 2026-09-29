"""不重新訓練：固定原框分類，僅置換 KD E2 關鍵點分支。"""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'kd/dual_task_v1'))
from teachers import SelectedSource, SELECTED, SELECTED_SHA, initialize, sha256

KD = ROOT / 'kd/pose_focus_v1/artifacts/runs/kd-e5-seed1-v1/inference/best_pose.pt'
KD_SHA = 'b67a5a82c3b36ff1bc701630ab7b874b06485ca2c9c6fff42074f6f202ebe0e4'
PREFIX = 'graph.model.23.pose_head.'
BRANCHES = ('cv4.', 'cv4_kpts.', 'cv4_sigma.', 'one2one_cv4.',
            'one2one_cv4_kpts.', 'one2one_cv4_sigma.')


def load_states():
    assert sha256(SELECTED) == SELECTED_SHA and sha256(KD) == KD_SHA
    parent = torch.load(SELECTED, map_location='cpu', weights_only=True)['state_dict']
    trained = torch.load(KD, map_location='cpu', weights_only=True)['state_dict']
    assert parent.keys() == trained.keys()
    assert all(torch.equal(v, trained[k]) for k, v in parent.items() if not k.startswith(PREFIX))
    keys = [k for k in parent if k.startswith(tuple(PREFIX + b for b in BRANCHES))]
    assert keys and all(any(k.startswith(PREFIX+b) for k in keys) for b in BRANCHES)
    mixed = {k: trained[k] if k in keys else v for k, v in parent.items()}
    return parent, trained, mixed, keys


def set_pose(model, state):
    # 共享 state 已先驗證相同；保留後端轉換建立的 PWL 常數表。
    model.load_state_dict({n: state[PREFIX + n[len('model.23.'):]] if n.startswith('model.23.')
                           else v for n, v in model.state_dict().items()}, strict=True)
    return model.eval().requires_grad_(False)


class MixedSource(SelectedSource):
    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        set_pose(pair.pose, load_states()[2])
        return pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--run', default='branch-v2')
    args = parser.parse_args()
    initialize()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    parent, trained, mixed, keys = load_states()
    assert args.run.replace('-', '').isalnum()
    out = HERE / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    for kind in ('float', 'bittrue'):
        pair = SelectedSource().build_task_models(kind)
        model = pair.pose.eval().requires_grad_(False)
        x = torch.rand(1, 3, 160, 160)
        with torch.inference_mode():
            outputs = []
            for state in (parent, trained, mixed):
                set_pose(model, state)
                outputs.append(model(x))
        p, k, m = outputs
        assert model.end2end
        assert torch.equal(p[0][..., :6], m[0][..., :6])
        for branch in ('one2many', 'one2one'):
            for field in ('boxes', 'scores'):
                assert torch.equal(p[1][branch][field], m[1][branch][field])
            assert torch.equal(k[1][branch]['kpts'], m[1][branch]['kpts'])
        assert torch.isfinite(m[0]).all()
        checks[kind] = dict(parent_boxes_scores_exact=True, kd_raw_keypoints_exact=True,
                            end2end=bool(model.end2end), output_shape=list(m[0].shape),
                            pose_parameters=sum(p.numel() for p in model.parameters()))
        del pair, model, outputs, p, k, m
    manifest = dict(status='cpu_probe_passed', parent=str(SELECTED), parent_sha256=SELECTED_SHA,
                    keypoint_source=str(KD), keypoint_sha256=KD_SHA, swapped_keys=keys,
                    checks=checks, added_parameters=0, added_operators=0,
                    scope='CPU synthetic 160x160; not accuracy or hardware latency validation')
    with (out / 'probe.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    with (out / 'candidate.pt').open('xb') as f:
        torch.save({'state_dict': mixed, 'metadata': manifest}, f)
    reloaded = torch.load(out / 'candidate.pt', map_location='cpu', weights_only=True)['state_dict']
    assert all(torch.equal(v, reloaded[k]) for k, v in mixed.items())
    print('CPU_PROBE_DONE: 原框與分數精確保留，KD 原始關鍵點精確接上；Float/BitTrue 皆通過', flush=True)
    if args.validate:
        from yolo_combine.fusion_model import assemble_graph_shared_model
        from yolo_combine.validation import JointValidator, ValidationSettings
        from yolo_combine.data import prepare_bbt5_view
        import yolo_combine.validation as validation
        from validate import InternalValidator
        from common import prepare_coco
        # 驗證的是重新載入的獨立匯出，而非僅驗證記憶體中的組合。
        source = MixedSource()
        pair = source.build_task_models()
        model, report = assemble_graph_shared_model(pair.detect, pair.pose)
        assert report.complete
        model.load_state_dict(reloaded, strict=True)
        view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',
                                HERE / 'artifacts/datasets/bbat5-v1-runtime')
        validation.DetectionValidator = InternalValidator
        validator = JointValidator(source, detect_data_yaml=prepare_coco(), pose_data_yaml=view.yaml,
            output_root=out / 'validation', settings=ValidationSettings(imgsz=640,
                detect_batch_size=32, pose_batch_size=16, detect_workers=4, pose_workers=4,
                device='0', plots=False, save_coco_json=False))
        results = {}
        for kind in ('float', 'bittrue'):
            result = validator.validate(model.eval(), epoch=0, kind=kind)
            results[kind] = dict(result.metrics)
            assert len(InternalValidator.last_instance.dataloader.dataset) == 5000
            InternalValidator.last_instance = None
        with (out / 'metrics.json').open('x') as f:
            json.dump(results, f, indent=2)
        print('JOB_DONE: 推論分支組合與 COCO/BBAT 完整驗證完成', flush=True)


if __name__ == '__main__':
    main()
