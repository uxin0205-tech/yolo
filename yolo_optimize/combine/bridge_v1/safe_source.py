"""使用已驗收 P3 bridge；限定 SHA256 與明確類別的 weights_only 載入。"""
import copy
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMBINE = HERE.parent
sys.path.insert(0, str(COMBINE))
from local_source import LocalSource, SOURCE, initialize, sha256
import torch
from yolo_combine.source import BuiltTaskModels, TrunkTransferReport, HeadTransferReport
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from pwl_contract import verify_pwl

DETECT = COMBINE.parent / 'studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/masf-e8-bittrue.pt'
DETECT_SHA = '73705178305e55e497ead1e1d8948114b67e1d17bf528861c9606e66cd372721'
POSE = Path('/home/uxin/yolo/yolo_combine/variants/full35/artifacts/pose/p0-full35-p3-b32a4-e100max-seed0/weights/best.pt')
POSE_SHA = '7b46c6af29723cff8ffdad96cf1220fd75975c31f99def58cfec84e80c322f99'

# 依上述兩份已驗收檔案的靜態 GLOBAL 清單明確列出，不從 pickle 任意 import。
ALLOWED = {
    'torch.nn.modules.activation': 'SiLU Tanh',
    'torch.nn.modules.batchnorm': 'BatchNorm2d',
    'torch.nn.modules.container': 'ModuleList Sequential',
    'torch.nn.modules.conv': 'Conv2d',
    'torch.nn.modules.linear': 'Identity Linear',
    'torch.nn.modules.pooling': 'MaxPool2d',
    'torch.nn.modules.upsampling': 'Upsample',
    'ultralytics.nn.modules.block': 'Bottleneck C2PSA C3k C3k2 PSABlock SPPF RealNVP',
    'ultralytics.nn.modules.conv': 'Concat Conv DWConv',
    'ultralytics.nn.modules.head': 'Pose26',
    'ultralytics.nn.tasks': 'DetectionModel PoseModel',
    'ultralytics.utils': 'IterableSimpleNamespace',
    'achitechure_1.masf': 'P3MASFFull35 _P3MASFContext',
    'achitechure_1.model': 'C3k2P3MASFFull35',
    'masf_task_bridge': 'TaskAlignedMASFDetect',
    'yolo_attention.attention': 'HardwareFriendlyAttention',
    'yolo_attention.binary_basis': 'BinaryScore',
    'yolo_attention.config': 'BasisKind BiasKind NormalizationKind RowCorrection ScaleMode VariantConfig',
    'yolo_attention.normalization': 'BitTruePiecewiseLinearSoftmax PiecewiseLinearSoftmax',
    'yolo_attention.projection': 'ModularQKVProjection',
    'yolo_attention.relative_bias': 'RelativePositionBias',
}


def load_known(path, expected):
    assert (Path(path), expected) in ((DETECT, DETECT_SHA), (POSE, POSE_SHA))
    assert sha256(path) == expected
    allowed = [set]
    for module, names in ALLOWED.items():
        imported = importlib.import_module(module)
        allowed.extend(getattr(imported, n) for n in names.split())
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


class BridgeSource(LocalSource):
    def __init__(self, root=None, architecture='full35'):
        assert architecture == 'full35'
        self.root = HERE
        self.architecture = architecture
        from yolo_combine.source import SourceBundle
        self.base = SourceBundle(SOURCE, architecture=architecture)
        self.record = {'detect_checkpoint': str(DETECT), 'detect_sha256': DETECT_SHA,
            'pose_checkpoint': str(POSE), 'pose_sha256': POSE_SHA,
            'detect_expected_metrics': {'coco/box/map50_95': .5082119551806837,
                                      'coco/person/box/map50_95': .6276641270640566}}

    def provenance(self, kind='float'):
        return {**self.record, 'source_kind': 'direction1_P3_bridge_E8_plus_trained_Pose_head',
            'normalization': kind, 'masf_location': 'Detect head P3 only',
            'loader': 'weights_only=True; explicit allowlist and SHA256',
            'plan': str(HERE / 'plan.json')}

    def original_pose(self, kind='float'):
        assert kind in ('float', 'bittrue')
        payload = load_known(POSE, POSE_SHA)
        model = copy.deepcopy(payload.get('ema') or payload['model']).float()
        if hasattr(model, 'criterion'):
            del model.criterion
        convert_yolo26_model(model, VariantConfig.from_yaml(SOURCE / f'configs/attention/{kind}-pwl-final.yaml'))
        verify_pwl(model)
        return model.eval()

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        assert kind in ('float', 'bittrue')
        if pose_head_checkpoint is not None:
            assert Path(pose_head_checkpoint).resolve() == POSE.resolve()
        payload = load_known(DETECT, DETECT_SHA)
        detect = copy.deepcopy(payload['model']).float()
        if hasattr(detect, 'criterion'):
            del detect.criterion
        convert_yolo26_model(detect, VariantConfig.from_yaml(SOURCE / f'configs/attention/{kind}-pwl-final.yaml'))
        verify_pwl(detect)
        assert len(detect.model) == 24
        assert hasattr(detect.model[23], 'bridge_coefficient')
        assert abs(float(detect.model[23].bridge_coefficient) - .012076444778011642) < 1e-15
        assert abs(float(detect.model[23].p3_masf.alpha) - .017174629494547844) < 1e-10
        assert not any('masf' in n for n, _ in detect.model[:23].named_modules())
        pose = self.original_pose(kind)
        head_state = {n: v.clone() for n, v in pose.model[23].state_dict().items()}
        for i in range(23):
            pose.model[i] = copy.deepcopy(detect.model[i])
        pose.save = copy.deepcopy(detect.save)
        assert all(torch.equal(v, pose.model[23].state_dict()[n]) for n, v in head_state.items())
        count = sum(len(m.state_dict()) for m in detect.model[:23])
        return BuiltTaskModels(detect=detect.eval(), pose=pose.eval(),
            transfer=TrunkTransferReport(source_layers=23, target_layers=23, compatible_tensors=count,
                missing_tensors=(), shape_mismatches=()),
            pose_head_transfer=HeadTransferReport(compatible_tensors=len(head_state), missing_tensors=(),
                unexpected_tensors=(), shape_mismatches=()), pose_head_checkpoint=POSE)


if __name__ == '__main__':
    initialize()
    torch.set_num_threads(4)
    models = BridgeSource().build_task_models()
    print('PASS: weights_only 安全載入；P3 bridge／alpha／PWL／完整 Pose head 保留。')
