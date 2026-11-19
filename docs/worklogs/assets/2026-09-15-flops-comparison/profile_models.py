"""CPU 計算實際 640 前向的 Conv／矩陣乘加；不將二元運算冒充 FLOPs。"""
import sys,json,hashlib,gc
from pathlib import Path
ROOT=Path('/home/uxin/yolo')
sys.path.insert(0,str(ROOT/'yolo_optimize/model'))
from load_model import load_model,tensors
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from ultralytics import YOLO

torch.set_num_threads(4)
class Counter(TorchDispatchMode):
    def __init__(self):super().__init__();self.fp=0;self.integer=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if str(func) in ('aten.mm.default','aten.bmm.default','aten.addmm.default'):
            a,b=args[-2:] if str(func)=='aten.addmm.default' else args[:2]
            n=a.numel()*b.shape[-1]
            if a.is_floating_point() and b.is_floating_point():self.fp+=n
            else:self.integer+=n
        return func(*args,**(kwargs or {}))

def count(m,task=None):
    totals={'conv_mac':0,'binary_bit_products':0};hooks=[]
    def conv(mod,args,out):totals['conv_mac']+=out.numel()*(mod.in_channels//mod.groups)*mod.kernel_size[0]*mod.kernel_size[1]
    def binary(mod,args):
        q,k=args[:2]
        totals['binary_bit_products']+=2*q.shape[0]*q.shape[1]*q.shape[2]*q.shape[3]*k.shape[3]
    for mod in m.modules():
        if isinstance(mod,torch.nn.Conv2d):hooks.append(mod.register_forward_hook(conv))
        if type(mod).__name__=='StaticDyadicScore':hooks.append(mod.register_forward_pre_hook(binary))
    c=Counter()
    x=torch.rand(1,3,640,640,generator=torch.Generator().manual_seed(1))
    with torch.inference_mode(False),torch.no_grad(),c:
        out=m(x,task=task) if task else m(x)
    for h in hooks:h.remove()
    assert all(torch.isfinite(t).all() for t in tensors(out))
    totals.update(float_matrix_mac=c.fp,integer_matrix_mac=c.integer,float_flops_subtotal=2*(totals['conv_mac']+c.fp),integer_matrix_ops=2*c.integer)
    return totals

out={'input':[1,3,640,640],'convention':'1 MAC = 2 operations; CPU eval no_grad, no inference_mode; actual forward', 'excludes':'BN, bias additions, activations, PWL, reciprocal, Hadamard, pooling, decode, sort, NMS, reductions, memory traffic', 'results':{}}
for name,path in [('yolo26m',ROOT/'yolo_attention/weights/yolo26m.pt'),('yolo11m',Path('/home/uxin/single_view_pipeline/single_view_pipeline/weights/yolo11m.pt')),('yolo11m_bat',Path('/home/uxin/single_view_pipeline/single_view_pipeline/weights/yolo11m_bat.pt'))]:
    m=YOLO(path).model.float().eval()
    out['results'][name]={'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**count(m)}
    print(name,out['results'][name],flush=True);del m;gc.collect()
m=load_model(kind='bittrue')
for task in ['detect','pose','both']:
    out['results']['rep17_20_E5_'+task]=count(m,task)
    print(task,out['results']['rep17_20_E5_'+task],flush=True)
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2))
print('ALL_COMPLETE',flush=True)
