"""單點 RepConv 的初始化與 eval-only 原 Conv 格式轉換。

不修改 live 模組；尚未接入正式 trainer／checkpoint factory。
"""
import copy
import math
import torch
from . import runtime
from ultralytics.nn.modules.conv import Conv, RepConv

def from_conv(original):
    c = original.conv
    if type(original) is not Conv or c.kernel_size != (3,3) or c.stride != (2,2):
        raise ValueError('只接受 layer17／20 型態的原生 stride2 3×3 Conv')
    if c.dilation != (1,1) or c.padding != (1,1) or c.bias is not None:
        raise ValueError('不支援改變 dilation／padding／bias 的 Conv')
    rep = RepConv(c.in_channels,c.out_channels,3,2,g=c.groups,
                  act=copy.deepcopy(original.act),bn=False).to(c.weight)
    rep.conv1.load_state_dict(original.state_dict(),strict=True)
    for branch in (rep.conv1,rep.conv2):
        branch.bn.eps = original.bn.eps
        branch.bn.momentum = original.bn.momentum
    with torch.no_grad():
        rep.conv2.bn.weight.zero_()
        rep.conv2.bn.bias.zero_()
    for name in ('i','f','type','np'):
        if hasattr(original,name): setattr(rep,name,getattr(original,name))
    rep.train(original.training)
    return rep

def to_eval_conv(rep, template):
    """保留原 state keys，BN 編碼融合 bias；只供 eval 副本，不可接續訓練。"""
    if not isinstance(rep,RepConv) or rep.training:
        raise ValueError('只可轉換 eval 模式的未融合 RepConv')
    c = template.conv
    if rep.conv1.conv.weight.shape != c.weight.shape:
        raise ValueError('template channels 不符')
    result=copy.deepcopy(template).eval()
    with torch.no_grad():
        weight,bias=rep.get_equivalent_kernel_bias()
        result.conv.weight.copy_(weight)
        result.bn.running_mean.zero_()
        result.bn.running_var.fill_(1)
        result.bn.weight.fill_(math.sqrt(1+result.bn.eps))
        result.bn.bias.copy_(bias)
        result.bn.num_batches_tracked.zero_()
    result.requires_grad_(False)
    return result

def install_layer17(base):
    """只在原 BEST 載入後 graft；隨機新分支不消耗共同比較的 RNG。"""
    if hasattr(base, '_direction1_repconv'):
        raise ValueError('禁止重複安裝 RepConv')
    old=base.graph.model[17]
    devices=[old.conv.weight.device.index] if old.conv.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1702)
        base.graph.model[17]=from_conv(old)
    base._direction1_repconv={'kind':'repconv', 'layer':17, 'schema_version':1,
        'bn_eps':old.bn.eps, 'bn_momentum':old.bn.momentum, 'identity_branch':False}
    return dict(base._direction1_repconv)

def evaluation_copy(base):
    """原圖直接回傳；RepConv 只融合深拷貝的 eval 圖，不改 live／EMA。"""
    if not hasattr(base,'_direction1_repconv'):
        return base
    result=copy.deepcopy(base).eval()
    rep=result.graph.model[17]
    c=rep.conv1.conv
    with torch.random.fork_rng(devices=[]):
        template=Conv(c.in_channels,c.out_channels,3,2,g=c.groups,
                      act=copy.deepcopy(rep.act)).to(c.weight).eval()
    template.bn.eps=rep.conv1.bn.eps
    template.bn.momentum=rep.conv1.bn.momentum
    for name in ('i','f','type','np'):
        if hasattr(rep,name): setattr(template,name,getattr(rep,name))
    result.graph.model[17]=to_eval_conv(rep,template)
    del result._direction1_repconv
    return result
