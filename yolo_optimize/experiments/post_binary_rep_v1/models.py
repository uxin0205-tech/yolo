"""BinaryQK E5 後的 RepConv 三組；保留已訓 QK 與兩套 MASF。"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pose_masf_training_v1'))
import training_b as b
from modeling import StaticDyadicScore,NativePWLAttention,SelectedSource
from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.relative_bias import RelativePositionBias
from yolo_attention.config import BiasKind
from yolo_optimize.repconv import from_conv,to_eval_conv
from ultralytics.nn.modules.conv import Conv,RepConv
PARENT=Path('/home/uxin/yolo/yolo_optimize/experiments/post_masf_hardware_v1/artifacts/scale_bias/train/epochs/e5/inference.pt')
PARENT_SHA='c8dcaf439b68c77d45a4f213972e57d51912028b558418511badee55fafbf681'
ARMS=('rep17','rep20','rep17_20')
LAYERS={'rep17':(17,), 'rep20':(20,), 'rep17_20':(17,20), 'parent_reference':()}
SITES=('model.10.m.0.attn','model.22.m.0.1.attn')

class DyadicRelativeBias(RelativePositionBias):
    """每個 offset 固定 signed m/1024 表；不以影像選 scale，不存完整 NxN 表。"""
    def forward(self,scores,*,height,width):
        assert self.kind is BiasKind.DECOMPOSED_2D
        assert height<=self.max_size and width<=self.max_size
        def table(p):
            clipped=p.clamp(-32,32-1/1024)
            quantized=(clipped*1024).round()/1024
            return clipped+(quantized-clipped).detach() if self.training and torch.is_grad_enabled() else quantized
        y,x=torch.meshgrid(torch.arange(height,device=scores.device),torch.arange(width,device=scores.device),indexing='ij')
        y,x=y.flatten(),x.flatten();offset=self.max_size-1
        bias=table(self.table_y)[:,y[:,None]-y[None,:]+offset]+table(self.table_x)[:,x[:,None]-x[None,:]+offset]
        return scores+bias.unsqueeze(0).to(scores.dtype)

def replace_attention(model,template):
    for name in SITES:
        native=model.get_submodule(name);old=template.get_submodule(name)
        assert isinstance(native,NativePWLAttention) and isinstance(old,HardwareFriendlyAttention)
        hardware=HardwareFriendlyAttention(native,old.config,normalizer=copy.deepcopy(native.normalize))
        hardware.score=StaticDyadicScore(old.score)
        hardware.bias.load_state_dict(old.bias.state_dict(),strict=True)
        hardware.bias.__class__=DyadicRelativeBias
        hardware.eval();model.set_submodule(name,hardware)

class Source(b.TrainingSource):
    def __init__(self,*args,arm='parent_reference',**kwargs):
        super().__init__(*args,**kwargs);self.variant=arm
    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair=super().build_task_models(kind,pose_head_checkpoint=pose_head_checkpoint)
        original=SelectedSource().build_task_models(kind)
        replace_attention(pair.detect,original.detect);replace_attention(pair.pose,original.pose)
        return pair
    def provenance(self,kind='float'):
        return {**super().provenance(kind),'experiment':'post_binary_rep_v1','arm':self.variant,
            'fixed_parent':str(PARENT),'fixed_parent_sha256':PARENT_SHA,'no_unchanged_model_additional_training':True,
            'scale':'16 learned constants, unsigned m/1024, [1/1024,1]',
            'bias':'decomposed relative position signed 16-bit m/1024; loaded from completed scale_bias E5',
            'deployment_claim':'software reference, no target-board latency/energy measured'}

def build(arm):
    assert arm in (*ARMS,'parent_reference')
    assert b.sha256(PARENT)==PARENT_SHA
    model,_=b.build()
    template=SelectedSource().build_task_models('float').detect
    replace_attention(model.graph,template)
    model.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    # strict 載入後刷新非 persistent 的量化尺度快取。
    model.eval()
    for index in LAYERS[arm]:
        old=model.graph.model[index]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(index);model.graph.model[index]=from_conv(old)
        model.graph.model[index].np=sum(p.numel() for p in model.graph.model[index].parameters())
    configure(model,arm)
    return model,Source(arm=arm)

def configure(model,arm):
    for n,p in model.named_parameters():
        head=('.detect_head.' in n or '.pose_head.' in n) and '.p3_masf.' not in n
        special=any(n.startswith(f'graph.model.{i}.') for i in LAYERS[arm])
        p.requires_grad_(arm!='parent_reference' and (head or special))
    model.train()
    for m in model.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
    model.detect_head.p3_masf.eval();model.pose_head.p3_masf.eval()
    # template 僅供 reference 評估，不為未變架構加訓。
    if arm=='parent_reference':model.eval()

def folded(model):
    result=copy.deepcopy(model).cpu().eval()
    for index in (17,20):
        rep=result.graph.model[index]
        if not isinstance(rep,RepConv):continue
        c=rep.conv1.conv
        with torch.random.fork_rng(devices=[]):
            template=Conv(c.in_channels,c.out_channels,3,2,g=c.groups,act=copy.deepcopy(rep.act)).eval()
        template.bn.eps=rep.conv1.bn.eps;template.bn.momentum=rep.conv1.bn.momentum
        for n in ('i','f','type','np'):
            if hasattr(rep,n):setattr(template,n,getattr(rep,n))
        result.graph.model[index]=to_eval_conv(rep,template)
    return result
