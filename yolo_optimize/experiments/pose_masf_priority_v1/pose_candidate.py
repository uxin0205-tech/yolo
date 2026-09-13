"""固定 native QK E2 起點的 Pose P3 MASF；不修改 Detect、共享 Neck 或舊權重。"""
import copy
import json
from pathlib import Path
import sys
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ATTENTION=HERE.parent/'attention_recovery_v1'
sys.path.insert(0,str(ATTENTION))
from modeling import RecoverySource,initialize,sha256,save,verify_pwl
from yolo_combine.fusion_model import assemble_graph_shared_model
from ultralytics.nn.modules.head import Pose26
from masf_task_bridge import _GradientScale

CHECKPOINT=HERE/'artifacts/parent/native-qk-e2-inference.pt'
PAUSE=ATTENTION/'artifacts/queue-v1/pause-outcome-v1.json'
PIN=HERE/'source-pin.json'

def payload():
    pin=json.loads(PIN.read_text())
    assert sha256(CHECKPOINT)==pin['inference_sha256'],'固定起點已改變，禁止混用新權重'
    result=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert result['metadata']['epoch']==1 and result['checkpoint_kind']=='inference_only'
    return result

class PoseP3BridgeMASF(Pose26):
    def forward(self,x):
        if not self.training:
            features=list(x);features[0]=self.p3_masf(features[0])
            return super().forward(features)
        assert self.end2end and 0<self.bridge_coefficient<=.25
        assert not any(m.training for m in self.p3_masf.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
        many=self.forward_head([self.p3_masf(x[0]),x[1],x[2]],**self.one2many)
        feature=self.p3_masf(x[0].detach())
        feature=_GradientScale.apply(feature,self.bridge_coefficient)
        one=self.forward_head([feature,x[1].detach(),x[2].detach()],**self.one2one)
        return {'one2many':many,'one2one':one}

class PoseSource(RecoverySource):
    def __init__(self,*args,enabled=False,**kwargs):
        super().__init__(*args,arm='native_qk',**kwargs);self.enabled=enabled
    def provenance(self,kind='float'):
        return {**super().provenance(kind),'experiment':'pose_masf_priority_v1',
            'fixed_parent':str(CHECKPOINT),'fixed_parent_sha256':json.loads(PIN.read_text())['inference_sha256'],
            'pose_masf':self.enabled,'pose_masf_initialization':'copy trained Detect MASF; not Pose-trained',
            'location':'Pose head P3 only; raw P3 still supplies Neck P4/P5',
            'detect_masf_unchanged':True,'new_pose_masf_training_performed':False}
    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair=super().build_task_models(kind,pose_head_checkpoint=pose_head_checkpoint)
        if self.enabled:
            head=pair.pose.model[-1];assert type(head) is Pose26
            donor=pair.detect.model[-1]
            assert head.cv2[0][0].conv.in_channels==donor.cv2[0][0].conv.in_channels
            head.__class__=PoseP3BridgeMASF
            head.add_module('p3_masf',copy.deepcopy(donor.p3_masf))
            head.bridge_coefficient=donor.bridge_coefficient
        return pair

def build(enabled=False,zero_alpha=False):
    source=PoseSource(enabled=enabled);pair=source.build_task_models('float')
    model,report=assemble_graph_shared_model(pair.detect,pair.pose);assert report.complete
    original=payload()['state_dict'];state=model.state_dict()
    added=set(state)-set(original)
    assert not set(original)-set(state)
    assert all(n.startswith('graph.model.23.pose_head.p3_masf.') for n in added)
    assert bool(added)==enabled
    state.update(original);model.load_state_dict(state,strict=True)
    if enabled:
        model.pose_head.p3_masf.load_state_dict(model.detect_head.p3_masf.state_dict(),strict=True)
        if zero_alpha:
            with torch.no_grad():model.pose_head.p3_masf.alpha.zero_()
    model.eval()
    return model,source

def tensors(value):
    if isinstance(value,torch.Tensor):return [value]
    if isinstance(value,dict):return [t for v in value.values() for t in tensors(v)]
    if isinstance(value,(list,tuple)):return [t for v in value for t in tensors(v)]
    return []
