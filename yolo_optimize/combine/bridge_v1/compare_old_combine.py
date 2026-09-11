"""新舊融合完整同口徑驗證；安全 state-dict 載入與同權重 MASF 消融。"""
import copy
import json
import sys
import torch
from safe_source import HERE, BridgeSource, initialize, sha256
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import initialize_weights
from common import prepare_coco
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import JointValidator, ValidationSettings
from yolo_combine.metrics import GATE_METRICS
import yolo_combine.validation as validation
from validate import InternalValidator
from pwl_contract import verify_pwl
from verify_qk_challenger import tensors
from pathlib import Path

NEW = HERE/'artifacts/fusion/balanced-j3-v1/inference/best_pose.pt'
OLD = Path('/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt')
HASHES = {NEW: '3d8d806ce9247c6df6a1b8007bc52210db041e8060c52de71079f6a1ff1a8042',
          OLD: 'd67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c'}
OUT = HERE/'artifacts/old-combine-comparison-v1'


def exact(a, b):
    a, b = tensors(a), tensors(b)
    assert len(a) == len(b) and all(torch.equal(x,y) for x,y in zip(a,b))


class OldSource(BridgeSource):
    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        pair = super().build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        pose = self.original_pose(kind)
        for i in range(23):
            pair.detect.model[i] = copy.deepcopy(pose.model[i])
            pair.pose.model[i] = copy.deepcopy(pose.model[i])
        template = pair.detect.model[23]
        head = Detect(nc=80, reg_max=1, end2end=True, ch=(256,512,512))
        # YOLO 模型建構會設定 BN eps=1e-3；這些屬性不在 state_dict。
        initialize_weights(head)
        reference_bn = template.cv2[0][0].bn
        assert head.cv2[0][0].bn.eps == reference_bn.eps
        for name in ('f', 'i', 'type', 'stride'):
            setattr(head, name, copy.deepcopy(getattr(template, name)))
        pair.detect.model[23] = head.eval()
        pair.detect.save = copy.deepcopy(pose.save)
        pair.pose.save = copy.deepcopy(pose.save)
        return pair


def load_graph(path, source):
    assert sha256(path) == HASHES[path]
    payload = torch.load(path, map_location='cpu', weights_only=True)
    pair = source.build_task_models('float')
    model, report = assemble_graph_shared_model(pair.detect, pair.pose)
    assert report.complete and report.audit.compatible
    model.load_state_dict(payload['state_dict'], strict=True)
    verify_pwl(model)
    return model.eval(), payload


def preflight():
    results = {}
    sample = torch.rand(1,3,160,160, generator=torch.Generator().manual_seed(20260911))
    for label, path, source in [('new', NEW, BridgeSource()), ('old', OLD, OldSource())]:
        model, payload = load_graph(path, source)
        modules = [(n,m) for n,m in model.named_modules() if type(m).__name__ == 'P3MASFFull35']
        assert len(modules) == 1, [n for n,m in modules]
        name, masf = modules[0]
        alpha = float(masf.alpha)
        for kind in ('float', 'bittrue'):
            pair = build_graph_validation_models(model, source, kind=kind)
            verify_pwl(pair.detect); verify_pwl(pair.pose)
        with torch.inference_mode():
            on = model(sample, task='both')
            masf.alpha.zero_()
            off = model(sample, task='both')
        if label == 'new':
            exact(on['pose'], off['pose'])
        results[label] = {'sha256': HASHES[path], 'masf_site': name, 'alpha': alpha,
            'source_epoch_zero_based': payload['metadata']['epoch'], 'pwl': [-10,0],
            'segments': 20, 'float_bittrue_materialization': 'passed'}
    return results


def main():
    initialize()
    torch.set_num_threads(4)
    if '--preflight' in sys.argv:
        result = preflight()
        out = HERE/'artifacts/old-combine-comparison-preflight-v1.json'
        with out.open('x') as f: json.dump(result, f, indent=2)
        print(json.dumps(result), flush=True)
        return
    assert (HERE/'artifacts/old-combine-comparison-preflight-v1.json').exists()
    if '--resume' in sys.argv:
        assert OUT.is_dir() and not (OUT/'summary.json').exists()
    else:
        OUT.mkdir(exist_ok=False)
    coco = prepare_coco()
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml', HERE/'artifacts/datasets/bbat5-v1-runtime')
    validation.DetectionValidator = InternalValidator
    results = {}
    cases = [('new-on-float', NEW, BridgeSource(), 'float', False),
             ('new-on-bittrue', NEW, BridgeSource(), 'bittrue', False),
             ('new-zero-bittrue', NEW, BridgeSource(), 'bittrue', True),
             ('old-on-bittrue', OLD, OldSource(), 'bittrue', False),
             ('old-zero-bittrue', OLD, OldSource(), 'bittrue', True)]
    for label, path, source, kind, zero in cases:
        saved = OUT/(label+'.json')
        if saved.exists():
            assert '--resume' in sys.argv
            record = json.loads(saved.read_text())
            assert record['sha256'] == HASHES[path]
            results[label] = record['metrics']
            continue
        model, payload = load_graph(path, source)
        if zero:
            with torch.no_grad():
                for m in model.modules():
                    if type(m).__name__ == 'P3MASFFull35': m.alpha.zero_()
        validator = JointValidator(source, detect_data_yaml=coco, pose_data_yaml=view.yaml,
            output_root=OUT/(label+'-bn-eps-fixed' if label.startswith('old-') else label), settings=ValidationSettings(imgsz=640, detect_batch_size=32,
                pose_batch_size=16, detect_workers=4, pose_workers=4, device='0', plots=False, save_coco_json=False))
        result = validator.validate(model, epoch=0, kind=kind)
        assert len(InternalValidator.last_instance.dataloader.dataset) == 5000
        results[label] = dict(result.metrics)
        if label == 'new-on-bittrue':
            assert all(abs(result.metrics[k]-payload['metadata']['metrics'][k]) < 1e-8 for k in GATE_METRICS)
        if label == 'old-on-bittrue':
            expected = json.loads((OLD.parents[3]/'analysis/SUMMARY.json').read_text())
            assert all(abs(result.metrics[r['metric']]-r['j3']) < 1e-7 for r in expected['metrics'])
        with (OUT/(label+'.json')).open('x') as f:
            json.dump({'metrics': results[label], 'checkpoint': str(path), 'sha256': HASHES[path]},f,indent=2)
        del result, model
        InternalValidator.last_instance = None
    assert all(results['new-on-bittrue'][k] == results['new-zero-bittrue'][k] for k in GATE_METRICS if k.startswith('bbat/'))
    summary = {'status':'completed', 'results':results,
        'new_minus_old':{k:results['new-on-bittrue'][k]-results['old-on-bittrue'][k] for k in GATE_METRICS},
        'on_minus_zero':{label:{k:results[label+'-on-bittrue'][k]-results[label+'-zero-bittrue'][k] for k in GATE_METRICS} for label in ('new','old')},
        'note':'同權重推論消融不是移除 MASF 後重新訓練；新 Pose 分支不經過 MASF。'}
    with (OUT/'summary.json').open('x') as f: json.dump(summary,f,indent=2)
    print('JOB_DONE: 新舊 combine 同口徑與 MASF 消融完成',flush=True)


if __name__ == '__main__':
    main()
