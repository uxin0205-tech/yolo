"""增量雙層 RepConv；不改執行中的三臂模型／runner。"""
import sys
import torch
from torch import nn
import models as core
import run_arm as runner
ARM='rep17_20'
LAYERS=(17,20)

def configure(model,arm):
    if arm!=ARM:return core.configure(model,arm)
    for name,param in model.named_parameters():
        head=('.detect_head.' in name or '.pose_head.' in name) and '.p3_masf.' not in name
        rep=any(name.startswith(f'graph.model.{i}.') for i in LAYERS)
        param.requires_grad_(head or rep)
    model.train()
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
    model.detect_head.p3_masf.eval();model.pose_head.p3_masf.eval()

def build(arm):
    if arm!=ARM:return core.build(arm)
    model,_=core.build('native_qk_reference')
    for index in LAYERS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(index)
            model.graph.model[index]=core.from_conv(model.graph.model[index])
        model.graph.model[index].np=sum(p.numel() for p in model.graph.model[index].parameters())
    configure(model,arm)
    return model,core.Source(arm=arm)

def optimizer_for(model,arm):
    assert arm==ARM
    groups={};modules=dict(model.named_modules())
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        role='rep' if any(name.startswith(f'graph.model.{i}.') for i in LAYERS) else 'heads'
        no_decay=name.endswith('.bias') or isinstance(modules[name.rsplit('.',1)[0]],nn.modules.batchnorm._BatchNorm)
        key=role+('/no_decay' if no_decay else '/decay')
        group=groups.setdefault(key,dict(params=[],param_names=[],group_name=key,role=role,
            lr=runner.CFG['lrs'][role],weight_decay=0 if no_decay else runner.CFG['weight_decay']))
        group['params'].append(param);group['param_names'].append(name)
    optimizer=torch.optim.AdamW(list(groups.values()),betas=tuple(runner.CFG['betas']))
    if sys.argv[-1:] == ['smoke']:
        def check_both(opt,args,kwargs):
            for index in LAYERS:
                grad=model.graph.model[index].conv2.bn.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0,f'layer{index}新支路未收到task gradient'
        optimizer.register_step_pre_hook(check_both)
    return optimizer

def main():
    runner.ARMS=(ARM,)
    runner.build=build
    runner.configure=configure
    runner.optimizer_for=optimizer_for
    runner.main()

if __name__=='__main__':main()
