"""Pose head 內的空間蒸餾，涵蓋 one-to-many／one-to-one 框與關鍵點分支。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dual_task_v1'))
from teachers import pose_teacher, POSE_SHA
from spatial_kd import spatial_loss
from contextlib import contextmanager
from dataclasses import replace
import torch
from yolo_combine.joint_loss import NativeTaskLossRouter
from yolo_combine.contracts import Task

GROUPS=('cv2','cv4','one2one_cv2','one2one_cv4')

@contextmanager
def capture_head(head):
    features={};handles=[]
    for group in GROUPS:
        for i,branch in enumerate(getattr(head,group)):
            # box 的最後1x1輸出之前；Pose26 cv4 已是 kpts/sigma 前共用隱藏特徵。
            module=branch[-2] if group.endswith('cv2') else branch[-1]
            key=(group,i)
            def hook(m,args,output,k=key):
                if k in features:raise RuntimeError('repeated Pose head tap')
                features[k]=output
            handles.append(module.register_forward_hook(hook))
    try:yield features
    finally:
        for h in handles:h.remove()
        features.clear()

class HeadRouter(NativeTaskLossRouter):
    def __init__(self,model,*,mu,**kwargs):
        super().__init__(model,**kwargs)
        self.mu=float(mu)
        assert self.mu>=0
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
            self.teacher=pose_teacher().to(self.device)

    def parts(self,batch):
        assert not self.teacher.training and not any(p.requires_grad for p in self.teacher.parameters())
        with capture_head(self.teacher.model[-1]) as tf:
            with torch.no_grad():self.teacher(batch['img'])
            with capture_head(self.model.pose_head) as sf:
                native=super().loss_for(Task.POSE,batch)
                assert len(sf)==len(tf)==12
                kd=torch.stack([spatial_loss([sf[g,i] for i in range(3)],
                                             [tf[g,i] for i in range(3)]) for g in GROUPS]).mean(0).sum()
                features=[sf[g,i] for g in GROUPS for i in range(3)]
        return native,kd,features

    def loss_for(self,task,batch):
        assert Task(task) is Task.POSE, 'Pose-only router must not receive COCO training batches'
        if self.mu==0:return super().loss_for(task,batch)
        native,kd,_=self.parts(batch)
        return replace(native,raw_total=native.raw_total+self.mu*kd)

    def state_dict(self):
        return {**super().state_dict(),'head_kd':{'mu':self.mu,'teacher_sha256':POSE_SHA,'groups':list(GROUPS)}}

    def load_state_dict(self,state):
        assert state['head_kd']==self.state_dict()['head_kd']
        super().load_state_dict(state)
