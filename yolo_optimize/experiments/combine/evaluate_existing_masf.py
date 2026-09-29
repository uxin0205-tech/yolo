"""直接評估已訓練權重；不建立新 Pose head、不訓練、不修改標註。"""
import copy
import json
from local_source import HERE, ROOT, SOURCE, LocalSource, initialize
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils.metrics import ConfusionMatrix
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import extract_pose_metrics, _class_values, _overall
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from common import sha256
from qk_challenger import remove
from pwl_contract import verify_pwl

STUDY = ROOT / 'studies/pre-fusion-full35-b100/artifacts'
CASES = [('p3-control-e5', 'masf-p3-control-v1', 5),
         ('p3-shared-e5', 'masf-p3-shared-v1', 5),
         ('p3-detect-only-e5', 'masf-p3-fork-v1', 5),
         ('head-control-e8', 'masf-head-control-v1', 8),
         ('p3-bridge-e8', 'masf-task-bridge-v1', 8),
         ('p2-control-e5', 'masf-p2-control-v1', 5),
         ('p2-shared-e5', 'masf-p2-p2-v1', 5)]


class CanonicalBoxValidator(DetectionValidator):
    def build_dataset(self, img_path, mode='val', batch=None):
        args = copy.copy(self.args)
        args.task = 'pose'  # 解析原 Pose labels，保留其中既有 box，不寫另一份 labels。
        return build_yolo_dataset(args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def init_metrics(self, model):
        assert model.names[32] == 'sports ball' and model.names[34] == 'baseball bat'
        super().init_metrics(model)
        self.names = {0: 'ball', 1: 'bat'}
        self.nc = 2
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=self.names)
        self.is_coco = self.is_lvis = False
        self.args.save_json = False

    @staticmethod
    def remap(pred):
        selected = (pred['cls'] == 32) | (pred['cls'] == 34)
        result = {k: v[selected] for k, v in pred.items()}
        result['cls'] = (result['cls'] == 34).to(result['cls'].dtype)
        return result

    def postprocess(self, preds):
        return [self.remap(p) for p in super().postprocess(preds)]


def main():
    initialize()
    torch.set_num_threads(4)
    out = HERE / 'artifacts/existing-masf-bbat-v1'
    out.mkdir(parents=True, exist_ok=False)
    source = LocalSource()
    source.verify_manifest()
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',
                            HERE / 'artifacts/datasets/bbat5-v1-runtime')
    # 檢查類別映射，不能把 COCO person=0 誤當 BBAT ball=0。
    synthetic = {'cls': torch.tensor([0., 32., 34., 1.]), 'conf': torch.arange(4.),
                 'bboxes': torch.arange(16.).reshape(4, 4), 'extra': torch.empty(4, 0)}
    mapped = CanonicalBoxValidator.remap(synthetic)
    assert mapped['cls'].tolist() == [0., 1.] and mapped['conf'].tolist() == [1., 2.]
    rows = []
    for name, directory, epoch in CASES:
        path = STUDY / directory / f'epoch-{epoch:02d}-resume.pt'
        summary_path = STUDY / directory / 'summary.json'
        historical = json.loads(summary_path.read_text())
        record = next(e for e in historical['epochs'] if e['epoch'] == epoch)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['epoch'] == epoch
        model = copy.deepcopy(payload['ema']).float()
        remove(model)
        if hasattr(model, 'criterion'):
            del model.criterion
        convert_yolo26_model(model, VariantConfig.from_yaml(SOURCE / 'configs/attention/bittrue-pwl-final.yaml'))
        verify_pwl(model)
        validator = CanonicalBoxValidator(save_dir=out / name, args={
            'task': 'detect', 'data': str(view.yaml), 'imgsz': 640, 'batch': 16,
            'workers': 4, 'device': '0', 'plots': False, 'save_json': False,
            'compile': False, 'rect': True, 'split': 'val', 'mode': 'val', 'half': False})
        validator(model=model.eval())
        assert len(validator.dataloader.dataset) == 683
        assert validator.dataloader.dataset.use_keypoints
        metrics = {**_overall(validator.metrics.box, prefix='bbat/box'),
            **_class_values(validator.metrics.box, class_id=0, prefix='bbat/ball/box'),
            **_class_values(validator.metrics.box, class_id=1, prefix='bbat/bat/box')}
        row = {'name': name, 'checkpoint': str(path), 'sha256': sha256(path),
            'epoch': epoch, 'coco': record['ema'], 'coco_source': str(summary_path),
            'bbat': metrics, 'keypoint_ap': None, 'reason': 'Detect 權重沒有 Pose head',
            'masf': {n: float(m.alpha) for n, m in model.named_modules() if hasattr(m, 'alpha') and 'masf' in n}}
        rows.append(row)
        with (out / (name + '.json')).open('x') as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)
        del validator, model, payload
    # 現成 Pose 的 MASF 開／關：是推論消融，不冒稱重新訓練的 no-MASF control。
    baseline = json.loads((HERE / 'artifacts/baseline-v1/summary.json').read_text())['backends']['bittrue']['baseline']
    pose = source.original_pose('bittrue')
    alphas = {n: float(m.alpha) for n, m in pose.named_modules() if hasattr(m, 'alpha') and 'masf' in n}
    assert alphas and all('model.16.' in n for n in alphas)
    with torch.no_grad():
        for n, m in pose.named_modules():
            if n in alphas:
                m.alpha.zero_()
    validator = PoseValidator(save_dir=out / 'original-pose-masf-off', args={
        'task': 'pose', 'data': str(view.yaml), 'imgsz': 640, 'batch': 16, 'workers': 4,
        'device': '0', 'plots': False, 'save_json': False, 'compile': False,
        'rect': True, 'split': 'val', 'mode': 'val', 'half': False})
    validator(model=pose.eval())
    assert len(validator.dataloader.dataset) == 683
    off = extract_pose_metrics(validator.metrics, names=pose.names)
    with (out / 'summary.json').open('x') as handle:
        json.dump({'status': 'complete', 'training_performed': False, 'images': 683,
            'dataset': str(view.yaml), 'class_mapping': {'32': 0, '34': 1},
            'rows': rows, 'pose_ablation': {'checkpoint': source.record['pose_checkpoint'],
                'sha256': source.record['pose_sha256'], 'original_alphas': alphas,
                'on': {k: v for k, v in baseline.items() if k.startswith('bbat/')}, 'off': off,
                'on_source': str(HERE / 'artifacts/baseline-v1/summary.json'),
                'limitation': '關閉已訓練 MASF 的推論消融，不是無 MASF 成對重訓'}},
            handle, ensure_ascii=False, indent=2)
    print('ALL_DONE: 現有 P2／P3 權重 BBAT box 與原 Pose MASF 消融已完成；停止等待使用者決定', flush=True)


if __name__ == '__main__':
    main()
