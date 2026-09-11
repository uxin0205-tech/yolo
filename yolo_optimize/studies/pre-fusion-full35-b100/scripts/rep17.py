"""只重用已測試的 Conv↔RepConv 轉換，不導入融合後 runner 或改來源路徑。"""
import copy
import sys
from common import ROOT
from ultralytics.nn.modules.conv import Conv,RepConv

_paths=list(sys.path)
try:
    from yolo_optimize.repconv import from_conv,to_eval_conv
finally:
    sys.path[:]=_paths


def install(model):
    old=model.model[17]
    import torch
    devices=[old.conv.weight.device.index] if old.conv.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1702)
        model.model[17]=from_conv(old)
    model.model[17].np=sum(p.numel() for p in model.model[17].parameters())
    model.model[17].type='ultralytics.nn.modules.conv.RepConv'


def folded(model):
    result=copy.deepcopy(model).eval()
    rep=result.model[17]
    if not isinstance(rep,RepConv):return result
    import torch
    c=rep.conv1.conv
    with torch.random.fork_rng(devices=[]):
        template=Conv(c.in_channels,c.out_channels,3,2,g=c.groups,act=copy.deepcopy(rep.act)).to(c.weight).eval()
    template.bn.eps=rep.conv1.bn.eps;template.bn.momentum=rep.conv1.bn.momentum
    template.i=rep.i;template.f=rep.f;template.type='ultralytics.nn.modules.conv.Conv'
    template.np=sum(p.numel() for p in template.parameters())
    result.model[17]=to_eval_conv(rep,template)
    return result
