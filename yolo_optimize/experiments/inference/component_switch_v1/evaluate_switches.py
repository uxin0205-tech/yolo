"""目前選用權重的推論開關消融；不訓練、不改來源、不改 split。"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/activation/bridge_v1'))
from verify_selected import SelectedSource, SELECTED, SELECTED_SHA, initialize, sha256
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.validation import JointValidator, ValidationSettings
from yolo_combine.metrics import GATE_METRICS
import yolo_combine.validation as validation
from validate import InternalValidator
from common import prepare_coco
from yolo_combine.data import prepare_bbt5_view
from pwl_contract import verify_pwl
import torch

OUT = HERE / 'artifacts'
OLD = ROOT / 'experiments/activation/bridge_v1/artifacts/selected-and-teacher-probe-v1/summary.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


class SwitchSource(SelectedSource):
    def __init__(self, *, fp=False, off=False):
        super().__init__(teacher=fp)
        self.off = off

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        if self.off:
            with torch.no_grad():
                pair.detect.model[23].p3_masf.alpha.zero_()
        assert not any('masf' in n for n, _ in pair.pose.named_modules())
        return pair

    def provenance(self, kind='float'):
        return {**super().provenance(kind), 'masf_off': self.off,
                'experiment': '同權重即時切換，非重新訓練的架構比較'}


def graph(source):
    pair = source.build_task_models()
    model, report = assemble_graph_shared_model(pair.detect, pair.pose)
    assert report.complete
    return model.eval()


def leaves(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, (tuple, list)):
        for v in x:
            yield from leaves(v)


def preflight():
    torch.manual_seed(50900913)
    x = torch.rand(1, 3, 160, 160)
    records = {}
    for fp in (False, True):
        on = graph(SwitchSource(fp=fp))
        off_source = SwitchSource(fp=fp, off=True)
        off = graph(off_source)
        changed = [n for n, v in on.state_dict().items() if not torch.equal(v, off.state_dict()[n])]
        assert changed == ['graph.model.23.detect_head.p3_masf.alpha'], changed
        mats = build_graph_validation_models(off, off_source, kind='bittrue')
        assert float(mats.detect.model[23].p3_masf.alpha) == 0
        verify_pwl(mats.detect); verify_pwl(mats.pose)
        with torch.inference_mode():
            a = list(leaves(on(x, task='pose')))
            b = list(leaves(off(x, task='pose')))
            assert len(a) == len(b) and all(torch.equal(u, v) for u, v in zip(a, b))
            masf = off.detect_head.p3_masf
            z = torch.rand(1, masf.channels, 20, 20)
            assert torch.equal(masf(z), z)
            off_pred = list(leaves(off(x, task='detect')))
            bypass = copy.deepcopy(off)
            bypass.detect_head.p3_masf = torch.nn.Identity()
            bypass_pred = list(leaves(bypass(x, task='detect')))
            assert len(off_pred) == len(bypass_pred)
            assert all(torch.equal(u, v) for u, v in zip(off_pred, bypass_pred))
            assert all(torch.isfinite(v).all() for v in off_pred + a)
        sites = []
        for name, m in on.named_modules():
            if type(m).__name__ == 'HardwareFriendlyAttention':
                q = torch.randn(1, m.num_heads, m.key_dim, 9)
                k = torch.randn_like(q)
                if fp:
                    torch.testing.assert_close(m.score(q,k), (q * (m.key_dim**-.5)).transpose(-2,-1) @ k, rtol=0, atol=0)
                sites.append({'path': name, 'basis': str(m.score.basis),
                              'scale_mode': str(m.score.scale_mode),
                              'fixed_scales': m.score.fixed_coefficients.tolist()})
        assert len(sites) == 2
        records['fp' if fp else 'binary'] = {'only_changed_state': changed,
            'pose_outputs_exact_equal': True, 'alpha_zero_equals_identity': True,
            'detect_bypass_exact_equal': True, 'materialized_alpha_zero': True,
            'alpha_on': float(on.detect_head.p3_masf.alpha.detach()),
            'bridge_training_gradient_coefficient': float(on.detect_head.bridge_coefficient),
            'attention': sites,
            'layers': [{'index': i, 'class': type(m).__name__, 'from': m.f} for i,m in enumerate(on.graph.model)],
            'activation_types': sorted(set(type(m).__name__ for m in on.modules() if 'SiLU' in type(m).__name__)),
            'hog_sites': [n for n,m in on.named_modules() if 'hog' in n.lower()],
            'repconv_sites': [n for n,m in on.named_modules() if type(m).__name__ == 'RepConv']}
    save(OUT / 'preflight.json', {'status':'passed','checkpoint':str(SELECTED),
         'sha256':sha256(SELECTED),'checks':records,'not_accuracy_evaluation':True})
    print('JOB_DONE CPU 開關、旁路等價、Pose 隔離與架構核對通過', flush=True)


def evaluate(fp):
    assert json.loads((OUT/'preflight.json').read_text())['status'] == 'passed'
    name = 'fp_masf_off' if fp else 'binary_masf_off'
    dst = OUT/(name+'.json')
    assert not dst.exists(), '已完成結果不重跑'
    source = SwitchSource(fp=fp, off=True)
    model = graph(source)
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',
        ROOT/'experiments/activation/bridge_v1/artifacts/datasets/bbat5-v1-runtime')
    validation.DetectionValidator = InternalValidator
    original_pose = validation.PoseValidator
    class CountingPose(original_pose):
        last_instance = None
        def init_metrics(self, model):
            super().init_metrics(model)
            type(self).last_instance = self
    validation.PoseValidator = CountingPose
    validator = JointValidator(source, detect_data_yaml=prepare_coco(), pose_data_yaml=view.yaml,
        output_root=OUT/name, settings=ValidationSettings(imgsz=640,detect_batch_size=32,
        pose_batch_size=16,detect_workers=4,pose_workers=4,device='0',plots=False,save_coco_json=False))
    result = validator.validate(model,epoch=0,kind='bittrue')
    assert len(InternalValidator.last_instance.dataloader.dataset) == 5000
    assert len(CountingPose.last_instance.dataloader.dataset) == 683
    metrics = dict(result.metrics)
    assert all(math.isfinite(v) for v in metrics.values()) and all(k in metrics for k in GATE_METRICS)
    old = json.loads(OLD.read_text())['results']['fp-qk-teacher-candidate' if fp else 'selected-binary-student']['bittrue']
    assert all(abs(metrics[k]-old[k])<1e-8 for k in GATE_METRICS if k.startswith('bbat/')), 'Pose 隔離未重現，需診斷'
    assert sha256(SELECTED) == SELECTED_SHA
    save(dst, {'status':'passed','checkpoint':str(SELECTED),'sha256':SELECTED_SHA,
        'metrics':metrics,'fp_qk':fp,'masf_alpha':0,'normalization':'bittrue PWL [-10,0], 20 segments',
        'coco_images':5000,'bbat_images':683,'training':False,'pose_matches_on':True,
        'settings':vars(validator.settings),'provenance':source.provenance('bittrue')})
    print('JOB_DONE '+name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job',choices=['preflight','binary_masf_off','fp_masf_off'])
    args = parser.parse_args()
    initialize(); torch.set_num_threads(4)
    if args.job == 'preflight': preflight()
    else: evaluate(args.job == 'fp_masf_off')
