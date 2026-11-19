"""獨立 CPU 算子稽核；inference_mode 會略過部分 dispatcher，不用它計 MAC。"""
import argparse,gc,json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from measure import HERE,build,forward,finite,initialize

class MatrixMAC(TorchDispatchMode):
    def __init__(self):super().__init__();self.floating=0;self.integer=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func)
        if name in ('aten.mm.default','aten.bmm.default','aten.addmm.default'):
            a,b=args[-2:] if name=='aten.addmm.default' else args[:2]
            mac=a.numel()*b.shape[-1]
            if a.is_floating_point() and b.is_floating_point():self.floating+=mac
            else:self.integer+=mac
        return func(*args,**(kwargs or {}))

def count(model,x,row):
    totals={'conv_mac':0,'binary_qk_bit_products':0};sites=[];hooks=[]
    def conv(m,args,out):totals['conv_mac']+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
    def binary(m,args):
        q,k=args[:2]
        if str(getattr(m,'basis','')).lower() not in ('fp','basiskind.fp'):
            n=q.shape[0]*q.shape[1]*q.shape[2]*q.shape[3]*k.shape[3]
            totals['binary_qk_bit_products']+=int(n)
            sites.append({'q':list(q.shape),'k':list(k.shape),'bit_products':int(n)})
    for m in model.modules():
        if isinstance(m,torch.nn.Conv2d):hooks.append(m.register_forward_hook(conv))
        if type(m).__name__=='BinaryScore':hooks.append(m.register_forward_pre_hook(binary))
    counter=MatrixMAC()
    with torch.inference_mode(False),torch.no_grad(),counter:out=forward(model,x,row)
    assert finite(out)
    for h in hooks:h.remove()
    totals.update(float_matrix_mac=counter.floating,integer_matrix_mac=counter.integer,
        dense_mac_subtotal=totals['conv_mac']+counter.floating+counter.integer,
        float_flops_subtotal=2*(totals['conv_mac']+counter.floating),
        integer_ops_subtotal=2*counter.integer,binary_sites=sites,
        excludes='BN/activation/PWL/reciprocal/pooling/decode/sort/NMS/reductions/memory traffic',
        accounting_mode='eval, no_grad, inference_mode=False to preserve dispatcher visibility')
    return totals

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case');args=parser.parse_args()
    initialize();torch.set_num_threads(4)
    cases=json.loads((HERE/'manifest.json').read_text())['cases']
    out=HERE/'artifacts/accounting-v2';out.mkdir(exist_ok=True,parents=True)
    for row in cases:
        if args.case and row['id']!=args.case:continue
        p=out/(row['id']+'.json')
        if p.exists():continue
        model,info=build(row);x=torch.rand(1,3,640,640,generator=torch.Generator().manual_seed(50900912))
        result={'status':'passed','case':row['id'],'checkpoint_sha256':row['sha256'],'cost':count(model,x,row),
            'no_gpu':True,'note':'補正 inference_mode dispatcher 漏算矩陣乘法；不重跑 latency／energy。'}
        p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('JOB_DONE accounting '+row['id'],flush=True);del model,x;gc.collect()

if __name__=='__main__':main()
