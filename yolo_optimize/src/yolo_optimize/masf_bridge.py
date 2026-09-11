"""原 Full35 shared MASF 的零殘差診斷；不移位、不刪除 context。"""
import math
import torch

def disable_shared_masf(base):
    module=base.graph.model[16].p3_masf
    if type(module).__name__!='P3MASFFull35' or module.alpha.numel()!=1:
        raise ValueError('只接受已核對的 Full35 scalar residual MASF')
    original=float(module.alpha.detach().cpu())
    if not math.isfinite(original):
        raise ValueError('原 MASF alpha 非有限')
    with torch.no_grad(): module.alpha.zero_()
    module.alpha.requires_grad_(False)
    return {'kind':'shared_masf_residual_zero','layer':16,
            'state_key':'graph.model.16.p3_masf.alpha','original_alpha':original,
            'effective_alpha':0.0,'context_preserved':True,
            'relocation_performed':False}
