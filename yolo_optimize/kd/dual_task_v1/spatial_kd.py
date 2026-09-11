"""Training-only、跨通道數空間 attention transfer；非 QK 排名蒸餾。"""
from contextlib import contextmanager
from dataclasses import replace
import json
import torch
import torch.nn.functional as F
from teachers import HERE, detect_teacher, pose_teacher, DETECT_SHA, POSE_SHA
from yolo_combine.joint_loss import NativeTaskLossRouter
from yolo_combine.contracts import Task

SITES=(16,19,22)

def spatial_loss(student, teacher):
    """每張影像的 loss；空間總和、三尺度平均，最後由原 macro 統一正規化。"""
    if len(student)!=3 or len(teacher)!=3:raise ValueError('exactly three feature scales required')
    terms=[]
    for s,t in zip(student,teacher):
        if s.ndim!=4 or t.ndim!=4 or s.shape[0]!=t.shape[0] or s.shape[2:]!=t.shape[2:]:
            raise ValueError('teacher/student batch and spatial coordinates must match')
        # fp32 避免 AMP 平方 overflow；不同 channel 數以平均能量聚合。
        sa=F.normalize(s.float().square().mean(1).flatten(1),p=2,dim=1,eps=1e-6)
        ta=F.normalize(t.detach().float().square().mean(1).flatten(1),p=2,dim=1,eps=1e-6)
        terms.append((sa-ta).square().sum(1))
    result=torch.stack(terms).mean(0)
    if not torch.isfinite(result).all():raise FloatingPointError('non-finite spatial KD')
    return result

@contextmanager
def capture(layers):
    values={};handles=[]
    for i in SITES:
        def hook(module,args,output,index=i):
            if index in values:raise RuntimeError('unexpected repeated shared feature forward')
            values[index]=output
        handles.append(layers[i].register_forward_hook(hook))
    try:yield values
    finally:
        for handle in handles:handle.remove()
        values.clear()

class SpatialRouter(NativeTaskLossRouter):
    """教師不是學生子模組，不進 EMA／optimizer／export；每次 forward 暫存即清除。"""
    def __init__(self,model,*,mu=None,**kwargs):
        super().__init__(model,**kwargs)
        if mu is None:
            record=json.loads((HERE/'artifacts/kd-calibration-v1/summary.json').read_text())
            assert record['status']=='passed'
            mu=record['mu']
        self.mu={Task(k):float(v) for k,v in mu.items()}
        assert set(self.mu)==set(Task) and all(v>=0 for v in self.mu.values())
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
            self.teachers={Task.DETECT:detect_teacher().to(self.device),Task.POSE:pose_teacher().to(self.device)}

    def parts(self,task,batch):
        task=Task(task);teacher=self.teachers[task]
        assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
        with capture(teacher.model) as tf:
            with torch.no_grad():teacher(batch['img'])
            with capture(self.model.graph.model) as sf:
                native=super().loss_for(task,batch)
                features=[sf[i] for i in SITES]
                kd=spatial_loss(features,[tf[i] for i in SITES]).sum()
        return native,kd,features

    def loss_for(self,task,batch):
        task=Task(task)
        if self.mu[task]==0:return super().loss_for(task,batch)
        native,kd,_=self.parts(task,batch)
        return replace(native,raw_total=native.raw_total+self.mu[task]*kd)

    def state_dict(self):
        return {**super().state_dict(),'spatial_kd':{'mu':{t.value:v for t,v in self.mu.items()},
            'detect_sha256':DETECT_SHA,'pose_sha256':POSE_SHA,'sites':list(SITES),'version':1}}

    def load_state_dict(self,state):
        assert state['spatial_kd']==self.state_dict()['spatial_kd']
        super().load_state_dict(state)
