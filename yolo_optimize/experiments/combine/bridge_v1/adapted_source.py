"""在逐 tensor 相同的 trunk 上，重用已完成 J0 的 Pose head。"""
import json
from safe_source import HERE, COMBINE, BridgeSource, sha256
import torch
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from safe_source import SOURCE
from pwl_contract import verify_pwl

J0 = COMBINE / 'artifacts/fusion/j0-no-masf-v1/inference/best_pose.pt'
J0_SHA = 'f483423c6618c69854ffb24779823c0a7084ffbbd3bff1446171d668967fd685'


class AdaptedBridgeSource(BridgeSource):
    def provenance(self, kind='float'):
        return {**super().provenance(kind), 'pose_head_adaptation': str(J0),
                'pose_head_adaptation_sha256': sha256(J0), 'shared_trunk_exact_reuse': True,
                'fresh_joint_optimizer_not_exact_resume': True}

    def build_task_models(self, kind='float', *, pose_head_checkpoint=None):
        assert sha256(J0) == J0_SHA
        pair = super().build_task_models('float', pose_head_checkpoint=pose_head_checkpoint)
        payload = torch.load(J0, map_location='cpu', weights_only=True)
        assert payload['metadata']['stage'] == 'j0' and payload['metadata']['epoch'] == 7
        summary = json.loads((J0.parents[1] / 'summary.json').read_text())
        assert summary['epochs_completed'] == 8 and summary['completed_stages'] == ['j0']
        state = payload['state_dict']
        trunk = {n: v for n, v in pair.detect.state_dict().items() if not n.startswith('model.23.')}
        assert len(trunk) == 568
        assert all(torch.equal(v, state['graph.'+n]) for n, v in trunk.items())
        prefix = 'graph.model.23.pose_head.'
        head = {n[len(prefix):]: v for n, v in state.items() if n.startswith(prefix)}
        pair.pose.model[23].load_state_dict(head, strict=True)
        for model in (pair.detect, pair.pose):
            convert_yolo26_model(model, VariantConfig.from_yaml(SOURCE / f'configs/attention/{kind}-pwl-final.yaml'))
            verify_pwl(model)
        return pair
