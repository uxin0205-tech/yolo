"""三組 Attention 的獨立重建；不修改原模型或外部套件。"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/activation/bridge_v1'))
from verify_selected import SelectedSource,SELECTED,SELECTED_SHA,initialize,sha256
from yolo_attention.binary_basis import clipped_ste_sign,deterministic_sign,fast_hadamard_transform
import yolo_attention.binary_basis as binary
from qk_challenger import TrainableBinaryScore,ExactDotSurrogate
from yolo_attention.projection import qkv_channel_indices
from ultralytics.nn.modules.block import Attention
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.source import ManifestReport
from pwl_contract import verify_pwl

ARMS=('binary_control','scale_bias','native_qk')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False,default=str);f.write('\n')


class StaticDyadicScore(nn.Module):
    """16 個跨圖片共用係數；訓練 STE，部署固定 m/1024，無 dynamic selector。"""
    denominator=1024
    def __init__(self,source):
        super().__init__()
        self.coefficients=nn.Parameter(source.fixed_coefficients.detach().clone())
        self.num_heads=source.num_heads
        self.register_buffer('inference_coefficients',self.quantized().detach().clone(),persistent=False)

    def quantized(self):
        return self.coefficients.detach().clamp(1/self.denominator,1).mul(self.denominator).round().div(self.denominator)

    def train(self,mode=True):
        super().train(mode)
        if not mode:self.inference_coefficients.copy_(self.quantized())
        return self

    def forward(self,q,k):
        active=self.training and torch.is_grad_enabled()
        sign=clipped_ste_sign if active else deterministic_sign
        qh=fast_hadamard_transform(q,dim=-2,normalize=active)
        kh=fast_hadamard_transform(k,dim=-2,normalize=active)
        dot=ExactDotSurrogate.apply if active else binary.xnor_popcount_dot
        z0=dot(sign(q),sign(k));z1=dot(sign(qh),sign(kh))
        if active:
            bounded=self.coefficients.clamp(1/self.denominator,1)
            c=bounded+(self.quantized()-bounded).detach()
        else:c=self.inference_coefficients
        return c[:,0].view(1,-1,1,1)*z0+c[:,1].view(1,-1,1,1)*z1


class NativePWLAttention(nn.Module):
    """原生 fused QKV 與浮點點積，保留 PWL；沒有二值 score／relative bias。"""
    def __init__(self,source):
        super().__init__()
        self.num_heads=source.num_heads;self.head_dim=source.head_dim;self.key_dim=source.key_dim
        self.dim=source.dim;self.scale=self.key_dim**-.5
        with torch.random.fork_rng(devices=[]):
            fused=Attention(self.dim,self.num_heads,attn_ratio=self.key_dim/self.head_dim)
        self.qkv=fused.qkv
        self.pe=copy.deepcopy(source.pe);self.proj=copy.deepcopy(source.proj)
        self.normalize=copy.deepcopy(source.normalize)
        parts=(source.qkv.q,source.qkv.k,source.qkv.v)
        assert all(isinstance(p.act,nn.Identity) for p in parts)
        assert all(p.bn.eps==parts[0].bn.eps and p.bn.momentum==parts[0].bn.momentum for p in parts)
        assert all(torch.equal(p.bn.num_batches_tracked,parts[0].bn.num_batches_tracked) for p in parts)
        self.qkv.bn.eps=parts[0].bn.eps;self.qkv.bn.momentum=parts[0].bn.momentum
        indices=qkv_channel_indices(num_heads=self.num_heads,key_dim=self.key_dim,head_dim=self.head_dim)
        with torch.no_grad():
            for p,ix in zip(parts,indices):
                self.qkv.conv.weight.index_copy_(0,ix,p.conv.weight)
                for n in ('weight','bias','running_mean','running_var'):
                    getattr(self.qkv.bn,n).index_copy_(0,ix,getattr(p.bn,n))
            self.qkv.bn.num_batches_tracked.copy_(parts[0].bn.num_batches_tracked)
        self.eval()

    def forward(self,x):
        b,c,h,w=x.shape;n=h*w
        q,k,v=self.qkv(x).view(b,self.num_heads,2*self.key_dim+self.head_dim,n).split(
            [self.key_dim,self.key_dim,self.head_dim],dim=2)
        # AMP 可用於投影；score／PWL 採 FP32 避免溢位，scale 為固定常數。
        with torch.autocast(device_type=x.device.type,enabled=False):
            probability=self.normalize((q.float()*self.scale).transpose(-2,-1)@k.float())
            aggregated=(v.float()@probability.transpose(-2,-1)).reshape(b,c,h,w)
        return self.proj(aggregated.to(v.dtype)+self.pe(v.reshape(b,c,h,w)))


def transform(model,arm):
    paths=[n for n,m in model.named_modules() if type(m).__name__=='HardwareFriendlyAttention']
    assert len(paths)==2
    for name in paths:
        m=model.get_submodule(name)
        assert m.progressive is None
        if arm=='native_qk':model.set_submodule(name,NativePWLAttention(m))
        elif arm=='scale_bias':m.score=StaticDyadicScore(m.score)
        else:
            m.score.__class__=TrainableBinaryScore;m.score.use_ste=True
    model.eval();verify_pwl(model)
    if arm=='native_qk':
        assert not any(type(m).__name__ in ('HardwareFriendlyAttention','BinaryScore','RelativePositionBias') for m in model.modules())
    return model


class RecoverySource(SelectedSource):
    arm='binary_control'
    def __init__(self,*args,arm=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.arm=arm or type(self).arm
        assert self.arm in ARMS

    def verify_manifest(self):
        assert sha256(SELECTED)==SELECTED_SHA
        return ManifestReport(files=1,bytes=SELECTED.stat().st_size)

    def provenance(self,kind='float'):
        return {**super().provenance(kind),'source_kind':'attention_recovery_0913',
            'arm':self.arm,'score_override':'native QK, binary branches and relative bias removed' if self.arm=='native_qk' else self.arm,
            'scale_quantization':'m/1024 fixed across images' if self.arm=='scale_bias' else 'unchanged or native constant',
            'native_relative_bias_removed':self.arm=='native_qk','pwl':[-10,0,20]}

    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair=super().build_task_models(kind,pose_head_checkpoint=pose_head_checkpoint)
        for model in (pair.detect,pair.pose):transform(model,self.arm)
        return replace(pair,transfer=replace(pair.transfer,compatible_tensors=sum(len(m.state_dict()) for m in pair.detect.model[:23])))


def build(arm,kind='float'):
    source=RecoverySource(arm=arm);pair=source.build_task_models(kind)
    model,report=assemble_graph_shared_model(pair.detect,pair.pose)
    assert report.complete
    return model.eval(),source


def coefficients(model):
    return {n:m.quantized().cpu().tolist() for n,m in model.named_modules() if isinstance(m,StaticDyadicScore)}
