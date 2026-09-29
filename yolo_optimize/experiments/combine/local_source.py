"""獨立無 MASF 來源適配器；重用原融合 factory／validator 的正式檢查。"""
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'studies/pre-fusion-full35-b100/scripts'))
from common import SOURCE, sha256
sys.path.insert(0, '/home/uxin/yolo/yolo_combine/src')
import torch
from yolo_combine.source import SourceBundle, BuiltTaskModels, TrunkTransferReport, HeadTransferReport, ManifestReport
from yolo_combine.xnor import install_xnor_backend, XNORExecutionConfig
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from qk_challenger import remove
from pwl_contract import verify_pwl

SELECTION = HERE / 'artifacts/selected-source-v1.json'


def initialize():
    from yolo_optimize.data_safety import install_readonly_data_guard
    install_readonly_data_guard()
    SourceBundle(SOURCE, architecture='full35').activate_code()
    install_xnor_backend(XNORExecutionConfig(backend='bool_tiled', token_tile=32))


class LocalSource:
    """Duck-typed SourceBundle，來源是新研究權重，不偽裝成原 A2 bundle。"""
    def __init__(self, root=None, architecture='full35'):
        assert architecture == 'full35'
        self.root = HERE
        self.architecture = architecture
        self.record = json.loads(SELECTION.read_text())
        self.base = SourceBundle(SOURCE, architecture='full35')

    def verify_manifest(self):
        count = size = 0
        for prefix in ('detect', 'pose'):
            path = Path(self.record[prefix + '_checkpoint'])
            assert sha256(path) == self.record[prefix + '_sha256'], path
            size += path.stat().st_size
            count += 1
        return ManifestReport(files=count, bytes=size)

    def verify_environment(self):
        return self.base.verify_environment()

    def activate_code(self):
        self.base.activate_code()

    def provenance(self, kind='float'):
        return {'source_kind': 'direction1_no_masf_detect_plus_canonical_pose_head',
                'selection': str(SELECTION), 'normalization': kind,
                'detect_sha256': self.record['detect_sha256'], 'pose_sha256': self.record['pose_sha256']}

    def original_pose(self, kind='float'):
        assert kind in ('float', 'bittrue')
        payload = torch.load(self.record['pose_checkpoint'], map_location='cpu', weights_only=False)
        pose = copy.deepcopy(payload.get('ema') or payload['model']).float()
        if hasattr(pose, 'criterion'):
            del pose.criterion
        remove(pose)
        convert_yolo26_model(pose, VariantConfig.from_yaml(SOURCE / f'configs/attention/{kind}-pwl-final.yaml'))
        verify_pwl(pose)
        return pose.eval()

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        assert kind in ('float', 'bittrue')
        if pose_head_checkpoint is not None:
            assert Path(pose_head_checkpoint).resolve() == Path(self.record['pose_checkpoint']).resolve()
        self.verify_manifest()
        payload = torch.load(self.record['detect_checkpoint'], map_location='cpu', weights_only=False)
        detect = copy.deepcopy(payload['ema']).float()
        remove(detect)
        if hasattr(detect, 'criterion'):
            del detect.criterion
        convert_yolo26_model(detect, VariantConfig.from_yaml(SOURCE / f'configs/attention/{kind}-pwl-final.yaml'))
        verify_pwl(detect)
        assert len(detect.model) == 24 and not any('masf' in n for n, _ in detect.named_modules())
        pose = self.original_pose(kind)
        original_head = {n: t.clone() for n, t in pose.model[23].state_dict().items()}
        assert pose.model[23].f == detect.model[23].f == [16, 19, 22]
        # 更換 Pose 的共享特徵提取部分，但完整保留原訓練 head，不平均兩個 trunk。
        for index in range(23):
            pose.model[index] = copy.deepcopy(detect.model[index])
        pose.save = copy.deepcopy(detect.save)
        assert all(torch.equal(t, pose.model[23].state_dict()[n]) for n, t in original_head.items())
        count = 0
        for left, right in zip(detect.model[:23], pose.model[:23]):
            assert left.state_dict().keys() == right.state_dict().keys()
            assert all(torch.equal(t, right.state_dict()[n]) for n, t in left.state_dict().items())
            count += len(left.state_dict())
        return BuiltTaskModels(detect=detect.eval(), pose=pose.eval(),
            transfer=TrunkTransferReport(source_layers=23, target_layers=23, compatible_tensors=count,
                                         missing_tensors=(), shape_mismatches=()),
            pose_head_transfer=HeadTransferReport(compatible_tensors=len(original_head), missing_tensors=(),
                                                  unexpected_tensors=(), shape_mismatches=()),
            pose_head_checkpoint=Path(self.record['pose_checkpoint']))
